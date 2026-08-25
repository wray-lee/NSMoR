"""Quick factory integration test - verify backward compatibility."""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nsmor.dataloader_factory import create_optimized_dataloader, create_dataloaders_from_config
from nsmor.nsmor_dataloader import NSMoRDataset
from nsmor.config_parser import ExperimentConfig
from nsmor.config import DEFAULT_FEATURE

print("[1/5] Testing imports...")
print("  - create_optimized_dataloader: OK")
print("  - create_dataloaders_from_config: OK")

print("\n[2/5] Creating synthetic dataset...")
# Create tiny synthetic dataset
n_seqs = 10
sequences = []
for i in range(n_seqs):
    T = np.random.randint(20, 50)
    X = np.random.randn(T, 8).astype(np.float32)
    Y = np.random.randn(T).astype(np.float32)
    sequences.append((X, Y, 0))

priors = np.random.rand(n_seqs, 4).astype(np.float32)
priors /= priors.sum(axis=1, keepdims=True)

dataset = NSMoRDataset(sequences=sequences, mcmc_priors=priors)
print(f"  - Created dataset with {len(dataset)} sequences")

print("\n[3/5] Testing create_optimized_dataloader...")
loader = create_optimized_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=-1,  # Auto-scale
)
print(f"  - DataLoader created: {len(loader)} batches")

# Test one batch
X_batch, Y_batch, lengths = next(iter(loader))
print(f"  - Batch shape: X={X_batch.shape}, Y={Y_batch.shape}, lengths={lengths.shape}")
assert X_batch.ndim == 3 and X_batch.shape[2] == 8, "X_batch shape mismatch"
assert Y_batch.ndim == 2, "Y_batch shape mismatch"
assert lengths.ndim == 1, "lengths shape mismatch"
print("  - Batch assertions: PASSED")

print("\n[4/5] Testing create_dataloaders_from_config...")
config = ExperimentConfig()
config.training.batch_size = 4
config.training.num_workers = -1

# Split dataset
train_dataset = NSMoRDataset(sequences=sequences[:7], mcmc_priors=priors[:7])
val_dataset = NSMoRDataset(sequences=sequences[7:], mcmc_priors=priors[7:])

train_loader, val_loader, test_loader = create_dataloaders_from_config(
    config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)
print(f"  - Train loader: {len(train_loader)} batches")
print(f"  - Val loader: {len(val_loader)} batches")
print(f"  - Test loader: {test_loader}")
assert train_loader is not None, "train_loader is None"
assert val_loader is not None, "val_loader is None"
assert test_loader is None, "test_loader should be None"
print("  - Config-driven loaders: PASSED")

print("\n[5/5] Testing worker auto-scaling logic...")
from nsmor.dataloader_factory import compute_num_workers, SMALL_DATASET_THRESHOLD

# Small dataset should get 0 workers
small_ds = NSMoRDataset(sequences=sequences[:5], mcmc_priors=priors[:5])
nw_small = compute_num_workers(small_ds, num_workers=-1)
print(f"  - Small dataset (n={len(small_ds)}): {nw_small} workers")
assert nw_small == 0, f"Small dataset should use 0 workers, got {nw_small}"

# Large dataset should get auto-scaled workers
large_seqs = sequences * 30  # 300 sequences
large_priors = np.tile(priors, (30, 1))
large_ds = NSMoRDataset(sequences=large_seqs, mcmc_priors=large_priors)
nw_large = compute_num_workers(large_ds, num_workers=-1)
print(f"  - Large dataset (n={len(large_ds)}): {nw_large} workers")
assert nw_large > 0, f"Large dataset should use >0 workers, got {nw_large}"

# Manual override should be honored
nw_override = compute_num_workers(small_ds, num_workers=2)
print(f"  - Manual override (num_workers=2): {nw_override} workers")
assert nw_override == 2, f"Override should be honored, got {nw_override}"

print("  - Worker auto-scaling: PASSED")

print("\n" + "="*60)
print("ALL TESTS PASSED - Factory integration verified!")
print("="*60)
