"""Numerical repair for the existing log-link Poisson GLM, with narrow compatibility.

The statistical model, L2 penalty, inference, and scoring are unchanged.
Only the GLM learning rate, loss evaluation, and norm reduction are repaired.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F


# Exact reviewed files only: arbitrary subsequent edits still invalidate caches.
# These aliases apply only to unchanged neural training / unchanged evaluation.
_SOURCE_ALIASES = {
    'neural_training.py': (
        '75014ae6078c8371042b1c2445ad91ac54013758fbcdb684975945925fea1f4f',
        '2154f5cabf5eeea5eb04f7a2ab305e3256c68a512f00cc61ac864be9398aa070'),
    'neural_experiment.py': (
        '3663ed88b421f0f78a01db037c914ac8e4202c3e5cd914e281b7ebabe69b9986',
        'e72d8ca9dd5900fb6481c390128031f959ef145a33a0622eb8f3baee743617a0'),
}


def compatible_source_hashes(hashes):
    result = dict(hashes)
    for name, (fixed, original) in _SOURCE_ALIASES.items():
        if result.get(name) == fixed:
            result[name] = original
    return result


def glm_nll(model, history, target):
    """Canonical Poisson NLL: exp(eta) - y*eta + a target-only constant.

Working with log rates avoids differentiating log(exp(eta) + epsilon),
which loses the correct gradient at very small rates. No rate cap is used.
"""
    features = ((history - model.center) / model.scale).flatten(1)
    eta = model.linear(features)
    return F.poisson_nll_loss(eta, target.float(), log_input=True,
                              full=True, reduction='none').sum(-1).mean()


@torch.no_grad()
def clip_glm_grad_norm_(parameters, max_norm):
    """Same global L2 clipping, with a float64 norm reduction on overflow.

Keep the usual float32 path when it is finite. Never suppress actual NaN/Inf
gradients, zero them, or skip a bad update.
"""
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError:
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g, dtype=torch.float64) for g in grads]))
        if not torch.isfinite(norm):
            raise
        factor = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for grad in grads:
            grad.mul_(factor.to(grad.dtype))
        return norm


def migrate_glm_config(output, config):
    """Archive only old GLM runs for the exact shipped 1e-3 -> 1e-4 repair.

Called under the experiment's output lock. All other configuration differences
remain errors. Existing neural checkpoints and evaluations are never rewritten.
"""
    from .neural_data import file_hash
    from .neural_storage import storage_compatible_hash
    from .neural_training import atomic_json

    output = Path(output)
    snapshot = output / 'config_snapshot.json'
    if not snapshot.exists():
        return
    previous = json.loads(snapshot.read_text())
    if previous == config:
        return
    expected_previous = copy.deepcopy(config)
    expected_previous['training']['glm_learning_rate'] = 0.001
    if config['training']['glm_learning_rate'] != 0.0001 or previous != expected_previous:
        return  # The ordinary frozen-configuration check supplies the error.
    root = Path(__file__).parent
    if any(storage_compatible_hash(name, file_hash(root / name)) != fixed
           for name, (fixed, _) in _SOURCE_ALIASES.items()):
        raise ValueError('GLM repair files have additional edits; refusing automatic migration.')

    backup = output / 'glm_before_numerical_fix'
    old_snapshot = backup / 'config_snapshot.json'
    if old_snapshot.exists() and json.loads(old_snapshot.read_text()) != previous:
        raise ValueError('Conflicting GLM repair backup; no files were overwritten.')
    moves = []
    for group in ('checkpoints', 'evaluations'):
        for session in previous['sessions']:
            for seed in previous['seeds']:
                relative = Path(group) / f'session_{session:03d}' / f'seed_{seed}' / 'glm'
                source, target = output / relative, backup / relative
                if source.exists():
                    if target.exists():
                        raise ValueError(f'GLM archive already exists: {target}. No overwrite is allowed.')
                    moves.append((source, target))
    # Old aggregate exports, if any, include the old GLM and must be regenerated.
    for path in output.iterdir():
        if (path.name.startswith('paper_neural_') or path.name in {
                'figures', 'all_metrics.csv', 'matched_nfe_metrics.csv',
                'dependence_control.csv', 'rollout_metrics.csv',
                'figure_captions.txt', 'report_status.json'}):
            target = backup / path.name
            if target.exists():
                raise ValueError(f'Export archive already exists: {target}. No overwrite is allowed.')
            moves.append((path, target))
    atomic_json(previous, old_snapshot)
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    atomic_json({'reason': 'Poisson GLM optimization/numerical repair',
                 'old_learning_rate': 0.001, 'new_learning_rate': 0.0001,
                 'archived': [str(target.relative_to(output)) for _, target in moves],
                 'neural_checkpoints_modified': False}, backup / 'repair.json')
    atomic_json(config, snapshot)
    print('GLM numerical repair: old GLM runs archived in ' + str(backup) +
          '. GLM candidates will restart at learning rate 1e-4; other model fits/results are reused.',
          flush=True)
