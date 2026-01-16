"""
Profile training to identify bottlenecks.
Run this to see where time is spent during training.
"""

import argparse
import warnings
import time
import torch
import torch.nn as nn
import torch.profiler
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress audio warnings and MP3 decoding notes
warnings.filterwarnings("ignore", message=".*Xing stream size.*")
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")
warnings.filterwarnings("ignore", message=".*trying to resync.*")
warnings.filterwarnings("ignore", message=".*hit end of available data.*")
warnings.filterwarnings("ignore", message=".*AUDIO-MPEG-HEADER.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Import from train_weighted_fusion
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import (
    CBIBirdDataset, NatureLMAudioEncoder, WeightedFusionModel,
    EBirdPriorWrapper, collate_fn
)
from sklearn.model_selection import train_test_split


def profile_training(args):
    """Profile a few training batches to identify bottlenecks."""
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load minimal data
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")
    
    # Create label mapping
    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    num_classes = len(all_species)
    
    # Small subset for profiling
    train_split, _ = train_test_split(
        train_csv, test_size=0.9, stratify=train_csv["species"], random_state=42
    )
    
    audio_dir = data_dir / "train_audio"
    # Make sure dataset preserves original indices for cache lookup
    train_dataset = CBIBirdDataset(audio_dir, train_split, label_to_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    # Initialize model
    print("Loading model...")
    start = time.time()
    audio_encoder = NatureLMAudioEncoder(use_qformer=args.use_qformer, pooling="mean")
    model = WeightedFusionModel(
        audio_encoder=audio_encoder,
        num_classes=num_classes,
        use_gating=False,  # Stage A
        w_max=2.0,
    )
    model = model.to(device)
    model.train()
    model.audio_encoder.eval()  # Freeze encoder
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    
    if device == "cuda":
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
              f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # Initialize prior model (with optional cache)
    print("Loading prior model...")
    prior_start = time.time()
    cache_path = Path(args.priors_cache) if args.priors_cache else None
    prior_model = EBirdPriorWrapper(
        priors_dir=Path(args.priors_dir).resolve() if args.priors_dir else None,
        species_list=all_species,
        cbi_metadata_path=data_dir / "train.csv",
        cache_path=cache_path,
        df_index=train_csv.index,  # Pass original dataframe index for cache lookup
    )
    prior_load_time = time.time() - prior_start
    print(f"Prior model loaded in {prior_load_time:.2f}s")
    if prior_model.use_cache:
        print(f"  ✓ Using pre-computed cache (fast lookups)")
    else:
        print(f"  ⚠️  No cache found - using on-the-fly computation (slow)")
        print(f"     Run precompute_priors.py to create cache")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    
    # Profile training batches (with REAL prior - this is what we actually use during training)
    print("\n" + "="*60)
    print("PROFILING TRAINING BATCHES (real prior - actual training regime)")
    print("="*60)
    
    train_batch_times = {
        "data_loading": [],
        "to_device": [],
        "prior_compute": [],
        "encoder_forward": [],
        "classifier_forward": [],
        "loss_compute": [],
        "backward": [],
        "optimizer_step": [],
        "total": [],
    }
    
    # Profile validation batches (with real prior - slower)
    print("\n" + "="*60)
    print("PROFILING VALIDATION BATCHES (real prior)")
    print("="*60)
    
    val_batch_times = {
        "data_loading": [],
        "to_device": [],
        "prior_compute": [],
        "encoder_forward": [],
        "classifier_forward": [],
        "loss_compute": [],
        "total": [],
    }
    
    num_profile_batches = min(5, len(train_loader))
    
    # Profile TRAINING batches (uniform prior)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_profile_batches:
            break
        
        total_start = time.time()
        
        # Data loading
        data_start = time.time()
        audio = batch["audio"]
        labels = batch["label"]
        metadata = batch["metadata"]
        data_time = time.time() - data_start
        train_batch_times["data_loading"].append(data_time)
        
        # To device
        device_start = time.time()
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        device_time = time.time() - device_start
        train_batch_times["to_device"].append(device_time)
        
        # REAL prior computation (this is what we use during actual training)
        # The model needs to see real priors to learn optimal fusion weights!
        # Fast if using cache, slow if on-the-fly
        prior_start = time.time()
        latitudes = [m["latitude"] for m in metadata]
        longitudes = [m["longitude"] for m in metadata]
        dates = [m["date"] for m in metadata]
        sample_indices = batch.get("sample_indices", None)
        prior_probs_np = prior_model.get_prior_probs_batch(
            latitudes, longitudes, dates, sample_indices=sample_indices
        )
        prior_probs = torch.from_numpy(prior_probs_np).float().to(device)
        prior_time = time.time() - prior_start
        train_batch_times["prior_compute"].append(prior_time)
        
        # Encoder forward
        encoder_start = time.time()
        with torch.no_grad():
            features = model.audio_encoder(audio)
        encoder_time = time.time() - encoder_start
        train_batch_times["encoder_forward"].append(encoder_time)
        
        # Classifier forward
        classifier_start = time.time()
        audio_logits = model.audio_classifier(features)
        classifier_time = time.time() - classifier_start
        train_batch_times["classifier_forward"].append(classifier_time)
        
        # Fusion forward
        fusion_start = time.time()
        eps = torch.clamp(model.epsilon, min=1e-8, max=0.5)
        uniform_prior = torch.ones_like(prior_probs) / num_classes
        prior_robust = (1 - eps) * prior_probs + eps * uniform_prior
        log_prior = torch.log(prior_robust + 1e-10)
        T = model.temperature.abs() + 1e-8
        w = model.w_weight
        final_logits = audio_logits / T + w * log_prior
        fusion_time = time.time() - fusion_start
        train_batch_times["classifier_forward"][-1] += fusion_time  # Add to classifier time
        
        # Loss
        loss_start = time.time()
        loss = nn.functional.cross_entropy(final_logits, labels)
        loss_time = time.time() - loss_start
        train_batch_times["loss_compute"].append(loss_time)
        
        # Backward
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - backward_start
        train_batch_times["backward"].append(backward_time)
        
        # Optimizer step
        step_start = time.time()
        optimizer.step()
        step_time = time.time() - step_start
        train_batch_times["optimizer_step"].append(step_time)
        
        total_time = time.time() - total_start
        train_batch_times["total"].append(total_time)
        
        print(f"\nTraining Batch {batch_idx + 1}/{num_profile_batches}:")
        print(f"  Data loading:     {data_time*1000:6.1f} ms ({100*data_time/total_time:5.1f}%)")
        print(f"  To device:        {device_time*1000:6.1f} ms ({100*device_time/total_time:5.1f}%)")
        print(f"  Prior compute:    {prior_time*1000:6.1f} ms ({100*prior_time/total_time:5.1f}%) ⚠️")
        print(f"  Encoder forward:   {encoder_time*1000:6.1f} ms ({100*encoder_time/total_time:5.1f}%)")
        print(f"  Classifier+fusion: {(classifier_time+fusion_time)*1000:6.1f} ms ({100*(classifier_time+fusion_time)/total_time:5.1f}%)")
        print(f"  Loss compute:     {loss_time*1000:6.1f} ms ({100*loss_time/total_time:5.1f}%)")
        print(f"  Backward:         {backward_time*1000:6.1f} ms ({100*backward_time/total_time:5.1f}%)")
        print(f"  Optimizer step:   {step_time*1000:6.1f} ms ({100*step_time/total_time:5.1f}%)")
        print(f"  TOTAL:            {total_time*1000:6.1f} ms")
        print(f"  Throughput:       {args.batch_size/total_time:.1f} samples/sec")
    
    # Profile VALIDATION batches (with real prior)
    # Use a small subset from the training split for profiling
    val_subset = train_split.iloc[:min(100, len(train_split))].copy()
    val_dataset = CBIBirdDataset(audio_dir, val_subset, label_to_idx)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    model.eval()
    num_val_batches = min(3, len(val_loader))
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_val_batches:
            break
        
        total_start = time.time()
        
        # Data loading
        data_start = time.time()
        audio = batch["audio"]
        labels = batch["label"]
        metadata = batch["metadata"]
        data_time = time.time() - data_start
        val_batch_times["data_loading"].append(data_time)
        
        # To device
        device_start = time.time()
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        device_time = time.time() - device_start
        val_batch_times["to_device"].append(device_time)
        
        # Real prior computation (fast if using cache, slow if on-the-fly)
        prior_start = time.time()
        latitudes = [m["latitude"] for m in metadata]
        longitudes = [m["longitude"] for m in metadata]
        dates = [m["date"] for m in metadata]
        sample_indices = batch.get("sample_indices", None)
        prior_probs_np = prior_model.get_prior_probs_batch(
            latitudes, longitudes, dates, sample_indices=sample_indices
        )
        prior_probs = torch.from_numpy(prior_probs_np).float().to(device)
        prior_time = time.time() - prior_start
        val_batch_times["prior_compute"].append(prior_time)
        
        # Encoder forward
        encoder_start = time.time()
        with torch.no_grad():
            features = model.audio_encoder(audio)
        encoder_time = time.time() - encoder_start
        val_batch_times["encoder_forward"].append(encoder_time)
        
        # Classifier forward
        classifier_start = time.time()
        with torch.no_grad():
            audio_logits = model.audio_classifier(features)
        classifier_time = time.time() - classifier_start
        val_batch_times["classifier_forward"].append(classifier_time)
        
        # Fusion forward
        fusion_start = time.time()
        with torch.no_grad():
            eps = torch.clamp(model.epsilon, min=1e-8, max=0.5)
            uniform_prior = torch.ones_like(prior_probs) / num_classes
            prior_robust = (1 - eps) * prior_probs + eps * uniform_prior
            log_prior = torch.log(prior_robust + 1e-10)
            T = model.temperature.abs() + 1e-8
            w = model.w_weight
            final_logits = audio_logits / T + w * log_prior
        fusion_time = time.time() - fusion_start
        val_batch_times["classifier_forward"][-1] += fusion_time
        
        # Loss
        loss_start = time.time()
        with torch.no_grad():
            loss = nn.functional.cross_entropy(final_logits, labels)
        loss_time = time.time() - loss_start
        val_batch_times["loss_compute"].append(loss_time)
        
        total_time = time.time() - total_start
        val_batch_times["total"].append(total_time)
        
        print(f"\nValidation Batch {batch_idx + 1}/{num_val_batches}:")
        print(f"  Data loading:     {data_time*1000:6.1f} ms ({100*data_time/total_time:5.1f}%)")
        print(f"  To device:        {device_time*1000:6.1f} ms ({100*device_time/total_time:5.1f}%)")
        print(f"  Prior compute:    {prior_time*1000:6.1f} ms ({100*prior_time/total_time:5.1f}%) ⚠️")
        print(f"  Encoder forward:   {encoder_time*1000:6.1f} ms ({100*encoder_time/total_time:5.1f}%)")
        print(f"  Classifier+fusion: {(classifier_time+fusion_time)*1000:6.1f} ms ({100*(classifier_time+fusion_time)/total_time:5.1f}%)")
        print(f"  Loss compute:     {loss_time*1000:6.1f} ms ({100*loss_time/total_time:5.1f}%)")
        print(f"  TOTAL:            {total_time*1000:6.1f} ms")
        print(f"  Throughput:       {args.batch_size/total_time:.1f} samples/sec")
    
    # Summary - Training
    print("\n" + "="*60)
    print("TRAINING SUMMARY (averaged, real prior - actual training regime)")
    print("="*60)
    
    for key, times in train_batch_times.items():
        if times:
            avg = np.mean(times)
            pct = 100 * avg / np.mean(train_batch_times["total"])
            print(f"{key:20s}: {avg*1000:7.2f} ms ({pct:5.1f}%)")
    
    avg_train_total = np.mean(train_batch_times["total"])
    print(f"\nAverage training batch time: {avg_train_total:.3f}s")
    print(f"Estimated time per epoch ({len(train_loader)} batches): {avg_train_total * len(train_loader) / 60:.1f} minutes")
    print(f"Training throughput: {args.batch_size / avg_train_total:.1f} samples/sec")
    
    # Summary - Validation
    print("\n" + "="*60)
    print("VALIDATION SUMMARY (averaged, real prior)")
    print("="*60)
    
    for key, times in val_batch_times.items():
        if times:
            avg = np.mean(times)
            pct = 100 * avg / np.mean(val_batch_times["total"])
            print(f"{key:20s}: {avg*1000:7.2f} ms ({pct:5.1f}%)")
    
    avg_val_total = np.mean(val_batch_times["total"])
    print(f"\nAverage validation batch time: {avg_val_total:.3f}s")
    print(f"Validation throughput: {args.batch_size / avg_val_total:.1f} samples/sec")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    encoder_pct = 100 * np.mean(train_batch_times["encoder_forward"]) / np.mean(train_batch_times["total"])
    data_pct = 100 * np.mean(train_batch_times["data_loading"]) / np.mean(train_batch_times["total"])
    prior_pct = 100 * np.mean(val_batch_times["prior_compute"]) / np.mean(val_batch_times["total"]) if val_batch_times["prior_compute"] else 0
    
    if encoder_pct > 70:
        print("⚠️  ENCODER IS THE BOTTLENECK (>70% of training time)")
        print("   - This is expected for large models like NatureLM")
        print("   - Solutions:")
        print("     * Use HPC with better GPUs (A100, H100)")
        print("     * Increase batch size if GPU memory allows")
        print("     * Use gradient checkpointing (if fine-tuning)")
        print("     * Consider using a smaller/faster encoder")
    
    if prior_pct > 10:
        if prior_model.use_cache:
            print(f"\n⚠️  PRIOR COMPUTATION IS STILL SLOW ({prior_pct:.1f}% of time) EVEN WITH CACHE")
            print("   - This suggests cache loading or lookup is inefficient")
            print("   - Check if cache file is on fast storage (SSD/NVMe)")
        else:
            print(f"\n⚠️  PRIOR COMPUTATION IS SLOW ({prior_pct:.1f}% of time)")
            print("   - This is expected without cache (on-the-fly COG file reads)")
            print("   - Solutions:")
            print("     * Run precompute_priors.py to create cache")
            print("     * Use --priors_cache argument to load cache")
    else:
        if prior_model.use_cache:
            print(f"\n✓ PRIOR COMPUTATION IS FAST ({prior_pct:.1f}% of time) WITH CACHE")
            print("   - Cache is working well!")
        else:
            print(f"\n✓ PRIOR COMPUTATION IS FAST ({prior_pct:.1f}% of time)")
            print("   - Consider using cache for even better performance")
    
    if data_pct > 20:
        print("\n⚠️  DATA LOADING IS SLOW (>20% of time)")
        print("   - Solutions:")
        print("     * Increase --num_workers (current: {})".format(args.num_workers))
        print("     * Use faster storage (SSD, NVMe)")
        print("     * Pre-process audio to reduce I/O")
    
    if encoder_pct <= 70 and data_pct <= 20 and prior_pct <= 10:
        print("✓ Training pipeline looks balanced")
        print("   - Consider increasing batch size for better GPU utilization")
    
    if device == "cuda":
        print(f"\nGPU Utilization: Check with 'nvidia-smi' during training")
        print(f"Current GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../Data/cbi")
    parser.add_argument("--priors_dir", type=str, default="../Data/priors")
    parser.add_argument("--priors_cache", type=str, default=None,
                       help="Path to pre-computed HDF5 cache (use precompute_priors.py to create)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_qformer", action="store_true")
    args = parser.parse_args()
    
    profile_training(args)

