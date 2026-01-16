# Plan: Group Prior by Seasons Instead of Weeks

## Current State
- Prior data has 52 bands (one per week)
- Cache computation groups samples by (week, species) for efficiency
- Each sample's prior is computed using its specific week
- Cache size: (num_samples, num_species) - doesn't depend on weeks

## Goal
- Group weeks into 4 seasons (Winter, Spring, Summer, Fall)
- Average abundance across all weeks in a season when querying
- Reduce computation time (4 seasons vs 52 weeks)
- Cache size remains the same, but computation is faster

## Season Mapping
- **Winter**: weeks 1-13 (Jan-Mar, ~13 weeks)
- **Spring**: weeks 14-26 (Apr-Jun, ~13 weeks)
- **Summer**: weeks 27-39 (Jul-Sep, ~13 weeks)
- **Fall**: weeks 40-52 (Oct-Dec, ~13 weeks)

## Implementation Steps

### 1. Add season mapping function
**File**: `Scripts/train_weighted_fusion.py`

Add helper function:
```python
def week_to_season(week: int) -> int:
    """Map week (1-52) to season (0=Winter, 1=Spring, 2=Summer, 3=Fall)."""
    if week <= 13:
        return 0  # Winter
    elif week <= 26:
        return 1  # Spring
    elif week <= 39:
        return 2  # Summer
    else:
        return 3  # Fall

def season_to_weeks(season: int) -> list:
    """Map season to list of weeks."""
    if season == 0:  # Winter
        return list(range(1, 14))
    elif season == 1:  # Spring
        return list(range(14, 27))
    elif season == 2:  # Summer
        return list(range(27, 40))
    else:  # Fall
        return list(range(40, 53))
```

### 2. Add season-averaging method to EBirdCOGPrior
**File**: `Scripts/eBirdPrior.py`

Add method to average across multiple weeks:
```python
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
        Array of shape (N,) with averaged probabilities
    """
    weeks = season_to_weeks(season)
    # Average across all weeks in season
    all_probs = []
    for week in weeks:
        probs = self.probs_batch(species, coords, week_idx=week, method=method, radius_km=radius_km)
        all_probs.append(probs)
    # Average across weeks
    return np.mean(all_probs, axis=0)
```

### 3. Update get_prior_probs_batch to use seasons
**File**: `Scripts/train_weighted_fusion.py` (lines 591-655)

Change:
- Convert dates to seasons instead of weeks
- Group by season instead of week
- Use `probs_batch_season` instead of `probs_batch`

### 4. Update cache computation to group by season
**File**: `Scripts/precompute_priors.py` (lines 130-152)

Change:
- Extract season instead of week from dates
- Group by (season, species) instead of (week, species)

### 5. Update cache key to use season
**File**: `Scripts/train_weighted_fusion.py` (line 528)

Change `_cache_key` to use season instead of week.

## Benefits
- **Faster computation**: 4 seasons vs 52 weeks = ~13x fewer queries
- **Similar informativeness**: Season averages still capture temporal patterns
- **Smaller cache hits**: More samples share same season (better caching)
- **Same cache size**: Still (num_samples, num_species)

## Trade-offs
- **Slight loss of temporal resolution**: Week-level vs season-level
- **Still informative**: Seasonal patterns are the main signal anyway

