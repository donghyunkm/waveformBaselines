"""
Patient-grouped batch sampler for efficient I/O during training.

Groups windows by patient so consecutive batches read from the same
memory-mapped numpy file. Shuffles patient order each epoch, and shuffles
windows within each patient for training diversity.

This ensures:
- Minimal file switching (one .npy file serves many consecutive batches)
- Training diversity (patient order + intra-patient shuffle per epoch)
- Balanced GPU utilization (no wasted time on file I/O mid-batch)
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class PatientGroupedSampler(Sampler[list[int]]):
    """
    Yields batches where all samples come from a minimal set of patients.
    
    Strategy:
    1. Shuffle patient order each epoch
    2. For each patient, shuffle its window indices
    3. Fill batches sequentially from the patient-ordered stream
    
    This means each batch draws from 1-2 patients max (since each patient
    has 100-2000 windows and batch_size is typically 32-128).

    Parameters
    ----------
    patient_boundaries : list of (start_idx, end_idx)
        Window index ranges per patient in the dataset.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle patient order and intra-patient windows each epoch.
    drop_last : bool
        Whether to drop the last incomplete batch.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        patient_boundaries: list[tuple[int, int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.patient_boundaries = patient_boundaries
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
        
        # Total number of samples
        self._total_samples = sum(end - start for start, end in patient_boundaries)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)

        # Get patient order
        n_patients = len(self.patient_boundaries)
        patient_order = np.arange(n_patients)
        if self.shuffle:
            rng.shuffle(patient_order)

        # Build the full ordered index stream
        # For each patient (in shuffled order), emit its window indices (shuffled within)
        all_indices = []
        for p_idx in patient_order:
            start, end = self.patient_boundaries[p_idx]
            window_indices = np.arange(start, end)
            if self.shuffle:
                rng.shuffle(window_indices)
            all_indices.append(window_indices)
        
        all_indices = np.concatenate(all_indices)

        # Yield batches
        n = len(all_indices)
        for i in range(0, n, self.batch_size):
            batch = all_indices[i:i + self.batch_size].tolist()
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self._total_samples // self.batch_size
        return math.ceil(self._total_samples / self.batch_size)


class PatientGroupedSamplerIndividual(Sampler[int]):
    """
    Individual-index variant (for use with DataLoader batch_size parameter).
    
    Yields individual indices in patient-grouped order. Use this with
    DataLoader's built-in batch_size parameter instead of batch_sampler.

    Parameters
    ----------
    patient_boundaries : list of (start_idx, end_idx)
        Window index ranges per patient in the dataset.
    shuffle : bool
        Whether to shuffle patient order and intra-patient windows each epoch.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        patient_boundaries: list[tuple[int, int]],
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.patient_boundaries = patient_boundaries
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._total_samples = sum(end - start for start, end in patient_boundaries)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)

        n_patients = len(self.patient_boundaries)
        patient_order = np.arange(n_patients)
        if self.shuffle:
            rng.shuffle(patient_order)

        for p_idx in patient_order:
            start, end = self.patient_boundaries[p_idx]
            window_indices = np.arange(start, end)
            if self.shuffle:
                rng.shuffle(window_indices)
            yield from window_indices.tolist()

    def __len__(self) -> int:
        return self._total_samples
