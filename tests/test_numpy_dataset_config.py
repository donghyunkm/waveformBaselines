from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from waveform_baselines.normalization import (
    compute_training_channel_stats,
    save_training_channel_stats,
)
from waveform_baselines.numpy_dataset import NumpyWaveformDataset


class NumpyWaveformDatasetNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.waveform_dir = root / "waveforms"
        self.waveform_dir.mkdir()
        self.splits_path = root / "splits.json"

        sig_len = 200_000
        anchor = 100_000
        self.anchor = anchor
        self.seq_len = 75_000
        self.channels = ["II", "ABP", "PLETH", "RESP"]

        patient_arrays = {
            "train_pid": np.vstack(
                [
                    np.full(sig_len, 70.0, dtype=np.float32),
                    np.full(sig_len, 80.0, dtype=np.float32),
                    np.full(sig_len, 10.0, dtype=np.float32),
                    np.full(sig_len, -2.0, dtype=np.float32),
                ]
            ),
            "val_pid": np.vstack(
                [
                    np.full(sig_len, 110.0, dtype=np.float32),
                    np.full(sig_len, 120.0, dtype=np.float32),
                    np.full(sig_len, 50.0, dtype=np.float32),
                    np.full(sig_len, 3.0, dtype=np.float32),
                ]
            ),
            "test_pid": np.vstack(
                [
                    np.full(sig_len, 60.0, dtype=np.float32),
                    np.full(sig_len, 90.0, dtype=np.float32),
                    np.full(sig_len, 40.0, dtype=np.float32),
                    np.full(sig_len, 0.0, dtype=np.float32),
                ]
            ),
        }
        patient_arrays["train_pid"][0, 0] = np.nan
        patient_arrays["train_pid"][1, 1] = np.inf

        for pid, arr in patient_arrays.items():
            np.save(self.waveform_dir / f"{pid}.npy", arr)

        metadata = {
            "channels": self.channels,
            "fs": 125,
            "ctx_samples": 150_000,
            "anchor_stride": 18_750,
            "min_windows": 100,
            "patients": {
                pid: {
                    "sig_len": sig_len,
                    "seg_start_secs": 0.0,
                    "n_anchors": 1,
                    "anchors": [anchor],
                    "channels": self.channels,
                    # Deliberately wrong to prove these are not used.
                    "channel_stats": [
                        {"mean": 999.0, "std": 999.0},
                        {"mean": 999.0, "std": 999.0},
                        {"mean": 999.0, "std": 999.0},
                        {"mean": 999.0, "std": 999.0},
                    ],
                }
                for pid in patient_arrays
            },
        }
        (self.waveform_dir / "metadata.json").write_text(json.dumps(metadata))
        self.splits_path.write_text(
            json.dumps(
                {
                    "train": ["train_pid"],
                    "val": ["val_pid"],
                    "test": ["test_pid"],
                }
            )
        )

        stats = compute_training_channel_stats(
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
        )
        save_training_channel_stats(
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            stats=stats,
        )
        self.stats = stats

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_stats_use_train_only_and_ignore_nonfinite(self):
        channel_stats = self.stats["channels"]
        self.assertAlmostEqual(channel_stats["II"]["mean"], 70.0)
        self.assertAlmostEqual(channel_stats["ABP"]["mean"], 80.0)
        self.assertAlmostEqual(channel_stats["PLETH"]["mean"], 10.0)
        self.assertAlmostEqual(channel_stats["RESP"]["mean"], -2.0)
        self.assertEqual(channel_stats["II"]["n_excluded_nonfinite"], 1)
        self.assertEqual(channel_stats["ABP"]["n_excluded_nonfinite"], 1)
        self.assertEqual(channel_stats["PLETH"]["n_excluded_nonfinite"], 0)
        self.assertEqual(channel_stats["RESP"]["n_excluded_nonfinite"], 0)

    def test_train_val_test_use_same_saved_stats(self):
        train_ds = NumpyWaveformDataset(
            split="train",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=True,
            channels=("II", "ABP", "PLETH", "RESP"),
            seq_len=self.seq_len,
        )
        val_ds = NumpyWaveformDataset(
            split="val",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=True,
            channels=("II", "ABP", "PLETH", "RESP"),
            seq_len=self.seq_len,
        )
        test_ds = NumpyWaveformDataset(
            split="test",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=True,
            channels=("II", "ABP", "PLETH", "RESP"),
            seq_len=self.seq_len,
        )
        self.assertEqual(train_ds._normalization_stats, val_ds._normalization_stats)
        self.assertEqual(train_ds._normalization_stats, test_ds._normalization_stats)

    def test_absolute_offsets_are_preserved_with_shared_stats(self):
        val_ds = NumpyWaveformDataset(
            split="val",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=True,
            channels=("II",),
            seq_len=self.seq_len,
        )
        test_ds = NumpyWaveformDataset(
            split="test",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=True,
            channels=("II",),
            seq_len=self.seq_len,
        )
        val_waveform = val_ds[0]["waveform"].numpy()[0]
        test_waveform = test_ds[0]["waveform"].numpy()[0]
        self.assertTrue(np.all(val_waveform > test_waveform))
        self.assertFalse(np.allclose(val_waveform, test_waveform))

    def test_no_normalize_returns_raw_values(self):
        ds = NumpyWaveformDataset(
            split="val",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=False,
            channels=("RESP", "II"),
            seq_len=self.seq_len,
        )
        sample = ds[0]["waveform"].numpy()
        self.assertAlmostEqual(float(sample[0, 0]), 3.0)
        self.assertAlmostEqual(float(sample[1, 0]), 110.0)

    def test_patient_metadata_channel_stats_are_not_used(self):
        ds = NumpyWaveformDataset(
            split="train",
            waveform_dir=self.waveform_dir,
            splits_path=self.splits_path,
            normalize=True,
            channels=("II",),
            seq_len=self.seq_len,
        )
        sample = ds[0]["waveform"].numpy()[0]
        # If patient metadata stats were used, this would be near (70-999)/999.
        self.assertLess(abs(float(sample[100]) - 0.0), 1e-5)


if __name__ == "__main__":
    unittest.main()
