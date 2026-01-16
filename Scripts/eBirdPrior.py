import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from pyproj import Transformer, CRS


def _week_to_band(week_idx: int) -> int:
    """Map week index (1..52) to 1..52. Clamps out-of-range values."""
    return int(np.clip(int(week_idx), 1, 52))


def _meters_per_degree(lat_deg: float) -> float:
    """Approximate meters-per-degree latitude for geographic CRS fallback."""
    return 111_132.954 - 559.822 * np.cos(2 * np.radians(lat_deg)) + 1.175 * np.cos(4 * np.radians(lat_deg))


class EBirdCOGPrior:
    """
    Optimized eBird S&T prior reader with band caching and vectorized operations.
    
    File naming expected: <species>_<product>_<res>_uint8_cog.{tif,json}
    Example: 'norcar_abundance_27km_uint8_cog.tif'
    """
    
    def __init__(self, priors_dir: str, product: str = "abundance", resolution: str = "27km",
                 band_cache_size: int = 128):
        """
        Args:
            priors_dir: Path to directory containing prior TIF and JSON files
            product: Product type (e.g., "abundance")
            resolution: Resolution string (e.g., "27km")
            band_cache_size: Max number of (species, band) combinations to cache in memory
        """
        self.dir = Path(priors_dir)
        self.product = product
        self.resolution = resolution
        self._ds_cache: Dict[str, Tuple] = {}  # species -> (dataset, meta, transformer, crs_info)
        self._band_cache_size = band_cache_size
        self._band_cache: Dict[Tuple[str, int], np.ndarray] = {}  # (species, band) -> array
        self._band_cache_order: List[Tuple[str, int]] = []  # LRU tracking

    def _paths(self, species: str) -> Tuple[Path, Path]:
        stem = f"{species}_{self.product}_{self.resolution}_uint8_cog"
        return self.dir / f"{stem}.tif", self.dir / f"{stem}.json"

    def _open(self, species: str):
        """Open and cache dataset, metadata, transformer, and CRS info for a species."""
        if species in self._ds_cache:
            return self._ds_cache[species]
        
        tif, js = self._paths(species)
        if not tif.exists():
            raise FileNotFoundError(f"Missing prior raster for {species}: {tif}")
        if not js.exists():
            raise FileNotFoundError(f"Missing prior metadata for {species}: {js}")
        
        with open(js, "r") as f:
            meta = json.load(f)
        
        ds = rasterio.open(tif)
        raster_crs = CRS.from_wkt(ds.crs.to_wkt()) if ds.crs else CRS.from_epsg(4326)
        to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        
        # Pre-compute CRS info for faster area calculations
        crs_info = {
            "is_projected": raster_crs.is_projected if raster_crs else False,
            "px": abs(ds.transform.a),
            "py": abs(ds.transform.e),
        }
        
        self._ds_cache[species] = (ds, meta, to_raster, crs_info)
        return self._ds_cache[species]

    def _get_band(self, species: str, band: int) -> np.ndarray:
        """Get a full band array, using cache when available."""
        cache_key = (species, band)
        
        if cache_key in self._band_cache:
            # Move to end of LRU list
            self._band_cache_order.remove(cache_key)
            self._band_cache_order.append(cache_key)
            return self._band_cache[cache_key]
        
        ds, meta, _, _ = self._open(species)
        arr = ds.read(band)  # Read entire band
        
        # Cache with LRU eviction
        if len(self._band_cache) >= self._band_cache_size:
            evict_key = self._band_cache_order.pop(0)
            del self._band_cache[evict_key]
        
        self._band_cache[cache_key] = arr
        self._band_cache_order.append(cache_key)
        return arr

    def _read_point(self, ds, meta, to_raster, lat: float, lon: float, band: int) -> float:
        """Read a single point (kept for backwards compatibility)."""
        x, y = to_raster.transform(lon, lat)
        r, c = rowcol(ds.transform, x, y, op=round)
        
        if r < 0 or c < 0 or r >= ds.height or c >= ds.width:
            return 0.0
        
        arr = ds.read(band, window=Window(c, r, 1, 1), boundless=True, fill_value=0)
        v = float(arr[0, 0]) * meta.get("scale", 1.0/255.0) + meta.get("offset", 0.0)
        return float(np.clip(v, 0.0, 1.0))

    def _read_area(self, ds, meta, to_raster, crs_info, lat: float, lon: float, 
                   band: int, radius_km: float) -> float:
        """Read area average around a point."""
        x, y = to_raster.transform(lon, lat)
        px, py = crs_info["px"], crs_info["py"]
        
        if crs_info["is_projected"]:
            radius_units = radius_km * 1000.0
        else:
            radius_units = (radius_km * 1000.0) / _meters_per_degree(lat)
        
        pix_rad_x = radius_units / (px if px != 0 else 1e-6)
        pix_rad_y = radius_units / (py if py != 0 else 1e-6)
        pr = int(max(1, round(max(pix_rad_x, pix_rad_y))))
        
        r0, c0 = rowcol(ds.transform, x, y, op=round)
        r1, r2 = max(0, r0 - pr), min(ds.height, r0 + pr + 1)
        c1, c2 = max(0, c0 - pr), min(ds.width, c0 + pr + 1)
        
        if r1 >= r2 or c1 >= c2:
            return 0.0
        
        arr = ds.read(band, window=Window(c1, r1, c2 - c1, r2 - r1), boundless=True, fill_value=0)
        
        yy, xx = np.ogrid[r1:r2, c1:c2]
        mask = (yy - r0)**2 + (xx - c0)**2 <= (pr**2)
        vals = arr[mask]
        
        if vals.size == 0:
            return 0.0
        
        v = float(np.nanmean(vals)) * meta.get("scale", 1.0/255.0) + meta.get("offset", 0.0)
        return float(np.clip(v, 0.0, 1.0))

    def prob(self, species: str, lat: float, lon: float, week_idx: int,
             method: str = "point", radius_km: float = 5.0) -> float:
        """
        Get p(species | lat, lon, week).
        
        Args:
            species: eBird species code
            lat, lon: Coordinates in WGS84
            week_idx: Week of year (1-52)
            method: "point" for single pixel, "area" for radius average
            radius_km: Radius for area method
        """
        ds, meta, to_raster, crs_info = self._open(species)
        band = _week_to_band(week_idx)
        
        if method == "point":
            return self._read_point(ds, meta, to_raster, lat, lon, band)
        elif method == "area":
            return self._read_area(ds, meta, to_raster, crs_info, lat, lon, band, radius_km)
        else:
            raise ValueError("method must be 'point' or 'area'")

    def probs_batch(self, species: str, coords: np.ndarray, week_idx: int,
                    method: str = "point", radius_km: float = 5.0) -> np.ndarray:
        """
        Vectorized probability lookup for multiple coordinates.
        
        Args:
            species: eBird species code
            coords: Array of shape (N, 2) with [lat, lon] rows
            week_idx: Week of year (1-52)
            method: "point" or "area"
            radius_km: Radius for area method
            
        Returns:
            Array of shape (N,) with probabilities
        """
        ds, meta, to_raster, crs_info = self._open(species)
        band = _week_to_band(week_idx)
        scale = meta.get("scale", 1.0/255.0)
        offset = meta.get("offset", 0.0)
        
        coords = np.asarray(coords)
        n = len(coords)
        
        if method == "point":
            # Vectorized coordinate transformation
            lats = coords[:, 0]
            lons = coords[:, 1]
            xs, ys = to_raster.transform(lons, lats)
            xs, ys = np.asarray(xs), np.asarray(ys)
            
            # Vectorized row/col calculation using affine transform
            inv_transform = ~ds.transform
            cols = (inv_transform.a * xs + inv_transform.b * ys + inv_transform.c)
            rows = (inv_transform.d * xs + inv_transform.e * ys + inv_transform.f)
            cols = np.round(cols).astype(np.int64)
            rows = np.round(rows).astype(np.int64)
            
            # Get full band (cached) - returns 2D array (height, width)
            band_data = self._get_band(species, band)
            
            # Bounds mask
            valid = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
            
            out = np.zeros(n, dtype=np.float32)
            valid_rows = rows[valid]
            valid_cols = cols[valid]
            
            # Vectorized pixel lookup - band_data is 2D (height, width)
            out[valid] = band_data[valid_rows, valid_cols] * scale + offset
            out = np.clip(out, 0.0, 1.0)
            return out
        
        else:  # method == "area"
            # Area method still requires per-point processing due to variable windows
            out = np.zeros(n, dtype=np.float32)
            for i, (lat, lon) in enumerate(coords):
                out[i] = self._read_area(ds, meta, to_raster, crs_info, 
                                         float(lat), float(lon), band, radius_km)
            return out

    def probs_batch_season(self, species: str, coords: np.ndarray, season: int,
                           method: str = "point", radius_km: float = 5.0) -> np.ndarray:
        """
        Get abundance averaged across all weeks in a season.
        
        Args:
            species: eBird species code
            coords: Array of shape (N, 2) with [lat, lon] rows
            season: Season index (0=Winter, 1=Spring, 2=Summer, 3=Fall)
            method: "point" or "area"
            radius_km: Radius for area method
        
        Returns:
            Array of shape (N,) with averaged probabilities across season
        """
        # Map season to weeks
        if season == 0:  # Winter
            weeks = list(range(1, 14))
        elif season == 1:  # Spring
            weeks = list(range(14, 27))
        elif season == 2:  # Summer
            weeks = list(range(27, 40))
        else:  # Fall
            weeks = list(range(40, 53))
        
        # Preload all bands for this season into cache (much faster!)
        # This ensures all bands are cached before we start querying
        for week in weeks:
            try:
                self._get_band(species, week)  # Preload into cache
            except:
                pass
        
        # Average across all weeks in season
        # Always use "point" method for speed - temporal averaging (13 weeks) compensates for spatial averaging
        all_probs = []
        for week in weeks:
            # Use point method for speed - much faster than area method
            # Temporal averaging (13 weeks) provides smoothing, so spatial averaging less critical
            probs = self.probs_batch(species, coords, week_idx=week, method="point", radius_km=radius_km)
            all_probs.append(probs)
        
        # Average across weeks (mean of abundance across season)
        return np.mean(all_probs, axis=0).astype(np.float32)
    
    def probs_multi_species(self, species_list: List[str], lat: float, lon: float,
                            week_idx: int, method: str = "point", 
                            radius_km: float = 5.0) -> Dict[str, float]:
        """
        Get probabilities for multiple species at a single location.
        
        Args:
            species_list: List of eBird species codes
            lat, lon: Coordinates
            week_idx: Week of year
            method: "point" or "area"
            radius_km: Radius for area method
            
        Returns:
            Dict mapping species code to probability
        """
        return {sp: self.prob(sp, lat, lon, week_idx, method, radius_km) 
                for sp in species_list}

    def probs_multi_species_batch(self, species_list: List[str], coords: np.ndarray,
                                   week_idx: int, method: str = "point",
                                   radius_km: float = 5.0) -> Dict[str, np.ndarray]:
        """
        Get probabilities for multiple species across multiple coordinates.
        
        Args:
            species_list: List of eBird species codes
            coords: Array of shape (N, 2) with [lat, lon] rows
            week_idx: Week of year
            method: "point" or "area"
            radius_km: Radius for area method
            
        Returns:
            Dict mapping species code to probability array of shape (N,)
        """
        return {sp: self.probs_batch(sp, coords, week_idx, method, radius_km)
                for sp in species_list}
    
    def clear_band_cache(self):
        """Clear the band cache to free memory."""
        self._band_cache.clear()
        self._band_cache_order.clear()
    
    def preload_species(self, species_list: List[str], weeks: Optional[List[int]] = None):
        """
        Preload bands for given species into cache.
        
        Args:
            species_list: Species codes to preload
            weeks: Week indices to preload (default: all 52 weeks)
        """
        weeks = weeks or list(range(1, 53))
        for sp in species_list:
            for wk in weeks:
                try:
                    self._get_band(sp, _week_to_band(wk))
                except FileNotFoundError:
                    pass  # Skip missing species


if __name__ == "__main__":
    import time
    
    # Update this path for your local setup
    priors_dir = "../Data/priors"
    prior = EBirdCOGPrior(priors_dir, product="abundance", resolution="27km")

    # Single query (Austin-ish), week 18 (~May)
    p = prior.prob("rewbla", 30.27, -97.74, week_idx=18, method="area", radius_km=5.0)
    print(f"p(rewbla | Austin, week 18) = {p:.4f}")

    # Batch over several points (lat, lon)
    coords = np.array([
        [30.27, -97.74],  # Austin
        [29.76, -95.37],  # Houston
        [32.78, -96.80],  # Dallas
    ])
    ps = prior.probs_batch("rewbla", coords, week_idx=26, method="point")
    print(f"ps(rewbla | coords, week 26) = {ps}")

    # Benchmark: vectorized batch vs hypothetical sequential
    print("\n--- Benchmark: Batch Point Lookup ---")
    n_points = 1000
    np.random.seed(42)
    # Random coords in Texas-ish region
    test_coords = np.column_stack([
        np.random.uniform(26, 36, n_points),   # lat
        np.random.uniform(-106, -93, n_points)  # lon
    ])
    
    # Warmup (loads band into cache)
    _ = prior.probs_batch("rewbla", test_coords[:10], week_idx=18, method="point")
    
    start = time.perf_counter()
    ps = prior.probs_batch("rewbla", test_coords, week_idx=18, method="point")
    elapsed = time.perf_counter() - start
    print(f"Batch lookup of {n_points} points: {elapsed*1000:.2f} ms ({n_points/elapsed:.0f} points/sec)")
    print(f"Mean prob: {ps.mean():.4f}, Non-zero: {(ps > 0).sum()}/{n_points}")
    
    # Multi-species test
    print("\n--- Multi-Species Lookup ---")
    species_list = ["rewbla", "ribgul", "amerob"]
    result = prior.probs_multi_species(species_list, 30.27, -97.74, week_idx=18)
    for sp, prob in result.items():
        print(f"  p({sp} | Austin) = {prob:.4f}")