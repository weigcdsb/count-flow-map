"""Chronological, memory-mapped spike histories for neural forecasting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class SpikeSession:
    def __init__(self, root, index, history_bins=10):
        self.root, self.index = Path(root), int(index)
        self.history_bins = int(history_bins)
        metadata_path = self.root / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        sessions = metadata['sessions']
        self.metadata = sessions[self.index] if isinstance(sessions, list) else sessions[str(self.index)]
        self.path = self.root / f'session_{self.index:03d}.npy'
        self.counts = np.load(self.path, mmap_mode='r', allow_pickle=False)
        if self.counts.ndim != 2 or not np.issubdtype(self.counts.dtype, np.integer):
            raise ValueError(f'{self.path}: expected integer counts [neurons, bins].')
        self.dim, self.n_bins = self.counts.shape
        if self.dim < 1 or self.history_bins < 1:
            raise ValueError('Empty population or invalid history length.')
        self.bin_width_ms = float(metadata.get('bin_width_ms', 50))
        if self.bin_width_ms != 50:
            raise ValueError('This experiment specifies 50-ms bins; reconfigure the protocol for other bins.')
        split = self.metadata['split_boundaries']
        train_end, val_end = int(split['train_end']), int(split['val_end'])
        if not 0 < train_end < val_end < self.n_bins:
            raise ValueError('Invalid chronological split boundaries.')
        self.bounds = {'train': (0, train_end), 'val': (train_end, val_end),
                       'test': (val_end, self.n_bins)}
        for lo, hi in self.bounds.values():
            if hi - lo <= self.history_bins:
                raise ValueError('Split is shorter than a history plus target.')
        # Validate storage, but estimate all model statistics on training bins only.
        sums, squares = np.zeros(self.dim), np.zeros(self.dim)
        for start in range(0, self.n_bins, 8192):
            block = np.asarray(self.counts[:, start:start + 8192])
            if np.any(block < 0):
                raise ValueError('Negative spike counts.')
            end = min(start + 8192, train_end)
            if start < end:
                x = np.asarray(self.counts[:, start:end], dtype=np.float64)
                sums += x.sum(1)
                squares += np.square(x).sum(1)
        self.mean = sums / train_end
        self.std = np.sqrt(np.maximum(squares / train_end - self.mean ** 2, 0)).clip(0.1)
        self.signature = fingerprint({'counts': file_hash(self.path),
                                      'session_metadata': self.metadata,
                                      'history_bins': self.history_bins, 'bin_width_ms': self.bin_width_ms})

    def indices(self, split, limit=None, horizon=1):
        lo, hi = self.bounds[split]
        first, stop = lo + self.history_bins, hi - int(horizon) + 1
        if stop <= first:
            raise ValueError('Split is too short for the forecast horizon.')
        if limit is None or stop - first <= int(limit):
            return np.arange(first, stop, dtype=np.int64)
        if int(limit) < 1:
            raise ValueError('Evaluation case count must be positive.')
        return np.linspace(first, stop - 1, int(limit), dtype=np.int64)

    def batch(self, indices, device='cpu', split=None, horizon=1):
        indices = np.asarray(indices, dtype=np.int64)
        lo, hi = self.bounds[split] if split is not None else (0, self.n_bins)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError('Expected a nonempty one-dimensional target-index array.')
        if np.any(indices - self.history_bins < lo) or np.any(indices + horizon > hi):
            raise ValueError('History or forecast crosses a split boundary.')
        positions = indices[:, None] + np.arange(-self.history_bins, 0)[None, :]
        history = np.asarray(self.counts[:, positions].transpose(1, 2, 0), dtype=np.float32)
        future_positions = indices[:, None] + np.arange(horizon)[None, :]
        future = np.asarray(self.counts[:, future_positions].transpose(1, 2, 0), dtype=np.int64)
        return torch.from_numpy(history).to(device), torch.from_numpy(future).to(device)

    def random_batch(self, size, rng, device):
        first, last = self.bounds['train']
        index = rng.integers(first + self.history_bins, last, size=int(size))
        history, future = self.batch(index, device, split='train')
        return history[:, -1].long(), future[:, 0], history
