"""Resumable fitting with atomic optimizer, EMA, and RNG checkpoints."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .bridge import conditional_birth_death_rates, sample_signed_binomial_bridge
from .losses import generalized_kl_rate_loss
from .neural_data import fingerprint, file_hash
from .neural_evaluation import evaluate
from .neural_models import build_model
from .neural_glm import compatible_source_hashes, glm_nll, clip_glm_grad_norm_
from .neural_original import original_compatible_hashes, validate_original_method, method_record
from .neural_storage import storage_compatible_hash, validate_selected, release_completed_last
from .utils import make_ema_copy, set_seed, update_ema


class TrainingPaused(RuntimeError):
    pass


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    # Only checkpoints produced locally by this experiment are loaded.
    return torch.load(path, map_location='cpu', weights_only=False)


def core_source_signature(method=None):
    root = Path(__file__).parent
    names = ['neural_data.py', 'neural_models.py', 'neural_training.py', 'neural_evaluation.py',
             'conditional_model.py', 'model.py', 'distributions.py', 'bridge.py',
             'losses.py', 'parameterization.py', 'utils.py']
    hashes = {name: file_hash(root / name) for name in names}
    if method in ('direct', 'transformer_nb', 'glm'):
        hashes = original_compatible_hashes(hashes)
    else:
        hashes['neural_original.py'] = file_hash(root / 'neural_original.py')
    hashes['neural_training.py'] = storage_compatible_hash('neural_training.py', hashes['neural_training.py'])
    if method in ('cfm', 'fm', 'direct', 'transformer_nb'):
        # This repair changes only GLM training; keep verified neural fits reusable.
        hashes = compatible_source_hashes(hashes)
    else:
        hashes['neural_glm.py'] = storage_compatible_hash('neural_glm.py', file_hash(root / 'neural_glm.py'))
    return fingerprint(hashes)


def candidate_configs(method, config):
    train = config['training']
    if method == 'glm':
        return [{'learning_rate': train['glm_learning_rate'], 'regularization': r}
                for r in train['glm_regularization']]
    return [{'learning_rate': lr, 'regularization': 0.0} for lr in train['learning_rates']]


def fit_signature(method, session, seed, candidate, config):
    training = copy.deepcopy(config['training'])
    model_config = dict(config['model'])
    if method != 'cfm':
        training.pop('endpoint_weight', None)
        training['validation_nfe']['cfm'] = 1
    if method not in ('cfm', 'fm'):
        model_config.pop('source_conditioned_support', None)
        # These settings never affect the unchanged direct predictors.
        model_config['correction_time_scale'] = 'remaining'
        training.pop('ck_teacher', None)
    if method != 'glm' and training['glm_learning_rate'] == 0.0001:
        # This GLM-only setting was unnecessarily included in every old fit hash.
        training['glm_learning_rate'] = 0.001
    identity = {'protocol': config['protocol_version'], 'data': session.signature,
                        'method': method, 'seed': seed, 'candidate': candidate,
                        'model': model_config, 'training': training,
                        'source': core_source_signature(method)}
    if method in ('cfm', 'fm'):
        identity['method_record'] = method_record(config)
    return fingerprint(identity)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def loss_for_batch(model, ema, method, session, rng, step, config, device, candidate):
    c = config['training']
    x0, x1, context = session.random_batch(c['batch_size'], rng, device)
    if method not in ('cfm', 'fm'):
        loss = (glm_nll(model, context, x1) if method == 'glm'
                else model.nll(context, x1)) / session.dim
        if method == 'glm':
            loss = loss + candidate['regularization'] * model.linear.weight.square().sum() / session.dim
        return loss, {'nll_per_neuron': float(loss.detach())}
    validate_original_method(config)
    time_point = torch.rand(len(x0), device=device) * c['tau']
    xt = sample_signed_binomial_bridge(x0, x1, time_point)
    target_birth, target_death = conditional_birth_death_rates(xt, x1, time_point)
    birth, death, _ = model.local_rates(xt, time_point, context)
    diagonal = generalized_kl_rate_loss(target_birth, target_death, birth, death) / session.dim
    if method == 'fm':
        return diagonal, {'diag_per_neuron': float(diagonal.detach())}
    x0, x1, context = session.random_batch(c['batch_size'], rng, device)
    progress = min(step / max(c['span_warmup_steps'], 1), 1)
    span = 0.1 + progress * (c['tau'] - 0.1)
    delta = torch.rand(len(x0), device=device) * span
    s = torch.rand(len(x0), device=device) * (c['tau'] - delta)
    u, t = s + delta / 2, s + delta
    xs = sample_signed_binomial_bridge(x0, x1, s)
    with torch.no_grad():
        # The manuscript target is stop-gradient K_theta, at the current weights.
        # EMA remains an evaluation/selection copy, not the training target.
        z = model.sample(xs, s, u, context)
        y = model.sample(z, u, t, context)
    ck = -model.log_prob(y, xs, s, t, context).mean() / session.dim
    alpha = c['ck_weight'] * min(step / max(c['ck_warmup_steps'], 1), 1)
    loss = diagonal + alpha * ck
    detail = {'diag_per_neuron': float(diagonal.detach()),
              'ck_per_neuron': float(ck.detach()), 'ck_weight': alpha}
    return loss, detail


def fit_candidate(method, session, seed, candidate, config, directory, device,
                  should_stop=lambda: False):
    validate_original_method(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = fit_signature(method, session, seed, candidate, config)
    c = config['training']
    selected_path, last_path = directory / 'selected.pt', directory / 'last.pt'
    if selected_path.exists():
        selected = load_checkpoint(selected_path)
        validate_selected(selected, signature, c['steps'])
        release_completed_last(directory, selected, signature, c['steps'], apply=True)
        print(f'[{method} session={session.index} seed={seed}] reuse completed fit', flush=True)
        return selected
    if last_path.exists():
        previous = load_checkpoint(last_path)
        if previous['signature'] != signature:
            raise ValueError(f'Incompatible checkpoint at {last_path}. Use a new output directory for changed settings/code/data.')
    else:
        previous = None
    set_seed(seed)
    model = build_model(method, session, config).to(device)
    ema = make_ema_copy(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'],
                                  weight_decay=0.0 if method == 'glm' else c['weight_decay'])
    rng = np.random.default_rng(seed + 103)
    best_score, best_step, best_state = float('inf'), 0, None
    step, elapsed, history = 0, 0.0, []
    if previous is not None:
        model.load_state_dict(previous['model'])
        ema.load_state_dict(previous['ema'])
        optimizer.load_state_dict(previous['optimizer'])
        step, elapsed, history = previous['step'], previous['elapsed'], previous['history']
        best_score, best_step, best_state = previous['best_score'], previous['best_step'], previous['best_state']
        rng.bit_generator.state = previous['data_rng']
        torch.set_rng_state(previous['torch_rng'])
        if torch.device(device).type == 'cuda' and previous['cuda_rng'] is not None:
            torch.cuda.set_rng_state(previous['cuda_rng'], device)
        print(f'[{method} session={session.index} seed={seed}] resume update {step}/{c["steps"]}', flush=True)
    else:
        print(f'[{method} session={session.index} seed={seed}] start {candidate}', flush=True)
    del previous
    val_indices = session.indices('val', c['validation_cases'])
    parameter_count = sum(p.numel() for p in model.parameters())
    start_clock, last_clock = time.perf_counter(), time.perf_counter()
    model.train()

    def save_last():
        atomic_torch_save({'signature': signature, 'step': step, 'model': cpu_state(model),
                          'ema': cpu_state(ema), 'optimizer': optimizer.state_dict(),
                          'best_state': best_state, 'best_score': best_score, 'best_step': best_step,
                          'data_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                          'cuda_rng': torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None,
                          'elapsed': elapsed + time.perf_counter() - start_clock, 'history': history}, last_path)

    # Save an initial checkpoint so an interruption before update 250 is restartable.
    if not last_path.exists():
        save_last()
    while step < c['steps']:
        if should_stop():
            save_last()
            raise TrainingPaused(f'Saved {last_path}; rerun the same command to continue.')
        next_step = step + 1
        loss, detail = loss_for_batch(model, ema, method, session, rng, next_step, config, device, candidate)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{method}: nonfinite loss at update {next_step}. Last valid checkpoint retained.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if method == 'glm':
            gradient_norm = clip_glm_grad_norm_(model.parameters(), c['grad_clip'])
        else:
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, c['ema_decay'])
        step = next_step
        if step == 1 or step % c['log_every'] == 0 or step == c['steps']:
            terms = ' '.join(f'{k}={v:.5f}' for k, v in detail.items())
            print(f'[{method} s={session.index} seed={seed}] {step}/{c["steps"]} '
                  f'loss={float(loss.detach()):.5f} grad={float(gradient_norm):.5f} {terms}', flush=True)
        validated = step % c['validate_every'] == 0 or step == c['steps']
        if validated:
            budgets = c['validation_nfe'].get(method, 1)
            budgets = [budgets] if isinstance(budgets, int) else budgets
            scores = {}
            for budget in budgets:
                metric, _ = evaluate(ema, method, session, val_indices, device=device,
                                     draws=c['validation_draws'], nfe=budget,
                                     tau=c['tau'], case_batch=c['validation_case_batch'],
                                     seed=c['validation_seed'], split='val')
                scores[str(budget)] = metric['energy_score']
            score = float(np.mean(list(scores.values())))
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite validation score.')
            if method == 'cfm':
                detail['validation_energy_by_nfe'] = scores
            history.append({'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(gradient_norm),
                            'validation_energy': score, **detail})
            if score < best_score:
                best_score, best_step, best_state = score, step, cpu_state(ema)
            by_budget = ' '.join(f'NFE={n}: {value:.6f}' for n, value in scores.items())
            print(f'[{method} s={session.index} seed={seed}] validation ES={score:.6f} '
                  f'({by_budget}); best update={best_step}', flush=True)
        now = time.perf_counter()
        if validated or step % c['checkpoint_every'] == 0 or now-last_clock >= c['checkpoint_seconds']:
            save_last()
            last_clock = time.perf_counter()
    selected = {'signature': signature, 'model': best_state, 'best_score': best_score,
                'best_step': best_step, 'completed_steps': step, 'candidate': candidate,
                'method': method, 'seed': seed, 'session': session.index,
                'parameters': parameter_count, 'training_seconds': elapsed + time.perf_counter() - start_clock,
                'history': history}
    if method in ('cfm', 'fm'):
        selected['method_record'] = method_record(config)
    atomic_torch_save(selected, selected_path)
    atomic_json({'best_score': best_score, 'best_step': best_step, 'completed_steps': step,
                 'candidate': candidate, 'history': history}, directory / 'training_history.json')
    release_completed_last(directory, selected, signature, c['steps'], apply=True)
    return selected
