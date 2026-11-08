"""Seeded, stratified train/validation split of a labelled dataset.

Use the validation subset for early stopping, LR scheduling and hyperparameter
choices; keep the official test set untouched until those choices are frozen.
"""
import torch
from torch.utils.data import Subset


def stratified_split_indices(labels, val_frac=0.1, seed=0):
    """Return (train_idx, val_idx) as sorted LongTensors; each class keeps its share."""
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    labels = torch.as_tensor(labels).cpu()
    generator = torch.Generator().manual_seed(seed)
    train_parts, val_parts = [], []
    for cls in torch.unique(labels):
        idx = torch.nonzero(labels == cls, as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n_val = round(idx.numel() * val_frac)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return torch.sort(torch.cat(train_parts)).values, torch.sort(torch.cat(val_parts)).values


def train_val_split(dataset, val_frac=0.1, seed=0):
    """Split an MNISTDataset (or any dataset with a `.label` tensor) into Subsets."""
    train_idx, val_idx = stratified_split_indices(dataset.label, val_frac, seed)
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())
