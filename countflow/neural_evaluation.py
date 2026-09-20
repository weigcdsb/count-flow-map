"""Conditional forecast scores, dependence controls, and autonomous rollouts."""
from __future__ import annotations

from contextlib import contextmanager
import math
import numpy as np
import torch

from .neural_models import forecast

LEVELS = (0.5, 0.8, 0.9, 0.95)


@contextmanager
def isolated_rng(seed, device):
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed(int(seed))
        yield


def sample_scores(samples, observations, rng):
    """Unbiased ensemble ES/CRPS; one observation for each conditioning history."""
    x = samples.float()
    y = observations.float()
    if x.ndim != 3 or y.shape != (len(x), x.shape[-1]):
        raise ValueError('Expected [cases, draws, neurons] and matching observations.')
    batch, draws, dim = x.shape
    if draws < 2 or draws % 2:
        raise ValueError('Use an even number of draws >= 2.')
    # Independent pairs avoid the self-pair bias of randomly permuting all draws.
    half = draws // 2
    es = (x - y[:, None]).norm(dim=-1).mean(1)
    es -= 0.5 * (x[:, :half] - x[:, half:]).norm(dim=-1).mean(1)
    a, observed = x.mean(-1), y.mean(-1)
    ordered = a.sort(dim=1).values
    weights = 2 * torch.arange(1, draws + 1, device=x.device) - draws - 1
    crps = (a - observed[:, None]).abs().mean(1)
    crps -= (ordered * weights).sum(1) / (draws * (draws - 1))
    mse = ((x.mean(1) - y).square() - x.var(1, unbiased=True) / draws).mean(-1)
    # Randomized ranks account for both discrete ties and finite ensemble size.
    less = (a < observed[:, None]).sum(1).cpu().numpy()
    equal = (a == observed[:, None]).sum(1).cpu().numpy()
    pit = (less + rng.random(batch) * (equal + 1)) / (draws + 1)
    result = {'energy_score': es.cpu().numpy() / math.sqrt(dim),
              'population_crps': crps.cpu().numpy(), 'mean_mse': mse.cpu().numpy(), 'pit': pit}
    for level in LEVELS:
        alpha = (1 - level) / 2
        result[f'coverage_{int(level*100)}'] = ((pit >= alpha) & (pit <= 1-alpha)).astype(float)
    result['width_90'] = (torch.quantile(a, 0.95, dim=1) - torch.quantile(a, 0.05, dim=1)).cpu().numpy()
    return result


def shuffle_neurons(samples, rng):
    """Permute draws independently per case and neuron; preserve all marginals."""
    x = samples.cpu().numpy()
    order = np.argsort(rng.random(x.shape), axis=1)
    return torch.from_numpy(np.take_along_axis(x, order, axis=1)).to(samples.device)


def combine_scores(pieces):
    arrays = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    summary = {key: float(value.mean()) for key, value in arrays.items() if key != 'pit'}
    summary['neuron_rmse'] = math.sqrt(max(0, summary.pop('mean_mse')))
    return summary, arrays


@torch.no_grad()
def evaluate(model, method, session, indices, *, device, draws, nfe=1, sampler='tau',
             tau=0.98, case_batch=16, seed=12345, split='val', shuffle=False):
    model.eval()
    score_rng, shuffle_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    parts, shuffled = [], []
    with isolated_rng(seed, device):
        for first in range(0, len(indices), case_batch):
            h, y = session.batch(indices[first:first+case_batch], device, split=split)
            samples = forecast(model, method, h, draws, nfe, sampler, tau)
            parts.append(sample_scores(samples, y[:, 0], score_rng))
            if shuffle:
                shuffled.append(sample_scores(shuffle_neurons(samples, shuffle_rng), y[:, 0], shuffle_rng))
    summary, arrays = combine_scores(parts)
    arrays['target_indices'] = indices
    if shuffled:
        summary_shuffle, array_shuffle = combine_scores(shuffled)
        summary.update({f'shuffled_{k}': v for k, v in summary_shuffle.items()})
        arrays.update({f'shuffled_{k}': v for k, v in array_shuffle.items()})
    return summary, arrays


@torch.no_grad()
def evaluate_rollouts(model, method, session, indices, horizons, *, device, draws,
                      nfe=1, sampler='tau', tau=0.98, case_batch=4, seed=23456):
    model.eval()
    rng = np.random.default_rng(seed)
    scores = {int(h): [] for h in horizons}
    last = max(scores)
    with isolated_rng(seed, device):
        for first in range(0, len(indices), case_batch):
            history, observed = session.batch(indices[first:first+case_batch], device,
                                               split='test', horizon=last)
            batch = len(history)
            histories = history.repeat_interleave(draws, dim=0)
            for horizon in range(1, last + 1):
                prediction = forecast(model, method, histories, 1, nfe, sampler, tau).squeeze(1)
                if horizon in scores:
                    samples = prediction.reshape(batch, draws, -1)
                    scores[horizon].append(sample_scores(samples, observed[:, horizon-1], rng))
                histories = torch.cat([histories[:, 1:], prediction[:, None].float()], dim=1)
    return {h: combine_scores(parts) for h, parts in scores.items()}
