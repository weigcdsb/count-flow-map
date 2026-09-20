"""Validation selection of EMA checkpoints without changing the training RNG."""
import random

import numpy as np
import torch


class ValidationSelector:
    def __init__(self, score, *, every, steps, verbose=True):
        if every < 1 or steps < 1:
            raise ValueError("Validation interval and training steps must be positive.")
        self.score = score
        self.every = int(every)
        self.steps = int(steps)
        self.verbose = verbose
        self.records = []
        self.best_score = float("inf")
        self.best_step = None
        self.best_state = None

    def __call__(self, ema, step):
        if step % self.every and step != self.steps:
            return
        # Scoring must not change subsequent training pairs, times, or CK samples.
        py_state, np_state = random.getstate(), np.random.get_state()
        was_training = ema.training
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                ema.eval()
                value = float(self.score(ema))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            ema.train(was_training)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite validation score at update {step}.")
        self.records.append((int(step), value))
        if value < self.best_score:  # Exact ties retain the earlier checkpoint.
            self.best_score, self.best_step = value, int(step)
            self.best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        if self.verbose:
            print(f"[validation] update={step} SW2={value:.6f}; best update={self.best_step}", flush=True)

    def add_history(self, history):
        if self.best_state is None:
            raise RuntimeError("No validation checkpoint was scored.")
        history.update(validation_step=[r[0] for r in self.records],
                       validation_sliced_w2=[r[1] for r in self.records],
                       selected_step=[self.best_step], selected_validation_sliced_w2=[self.best_score])

    def restore(self, ema):
        if self.best_state is None:
            raise RuntimeError("No validation checkpoint was scored.")
        ema.load_state_dict(self.best_state)
        ema.eval()
