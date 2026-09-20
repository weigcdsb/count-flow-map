"""Relative finite-time corrections preserving the original diagonal rates."""
import torch


def correction_duration(s: torch.Tensor, t: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "absolute":
        return t - s
    if mode == "remaining":
        if ((s < 0) | (t >= 1) | (t < s)).any():
            raise ValueError("Remaining-time corrections require 0 <= s <= t < 1.")
        return (t - s) / (1.0 - s)
    raise ValueError(f"Unknown correction_time_scale: {mode!r}")
