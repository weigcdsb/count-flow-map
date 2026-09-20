"""One production entry point shared by the notebook and command line."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import signal
import time
import warnings

import numpy as np
import torch

from .neural_data import SpikeSession, fingerprint, file_hash
from .neural_evaluation import evaluate, evaluate_rollouts, isolated_rng
from .neural_models import build_model, forecast
from .neural_glm import compatible_source_hashes
from .neural_original import original_compatible_hashes, migrate_original_config, validate_original_method
from .neural_training import (TrainingPaused, atomic_json, atomic_torch_save, candidate_configs,
                              core_source_signature, fit_candidate, fit_signature, load_checkpoint)


METHOD_LABELS = {'cfm': 'Count Flow Map', 'fm': 'Count-FM', 'direct': 'Direct count mixture',
                 'transformer_nb': 'Transformer-NB', 'glm': 'Population Poisson GLM'}


def read_config(path):
    config = json.loads(Path(path).read_text())
    validate_original_method(config)
    if config.get('purpose') not in ('paper', 'diagnostic'):
        raise ValueError('Set purpose to paper for the formal run or diagnostic for private checks.')
    c, e = config['training'], config['evaluation']
    endpoint_weight = c.get('endpoint_weight', 0.0)
    if not isinstance(endpoint_weight, (int, float)) or not math.isfinite(endpoint_weight) or endpoint_weight < 0:
        raise ValueError('training.endpoint_weight must be finite and nonnegative.')
    if not isinstance(config['model'].get('source_conditioned_support', False), bool):
        raise ValueError('model.source_conditioned_support must be boolean.')
    for method, value in c['validation_nfe'].items():
        budgets = [value] if isinstance(value, int) else value
        if (not isinstance(budgets, list) or not budgets or
                any(type(n) is not int or n < 1 for n in budgets) or
                len(set(budgets)) != len(budgets)):
            raise ValueError(f'Invalid training.validation_nfe for {method}.')
    if config['protocol_version'] != 2:
        raise ValueError('Unsupported neural protocol version.')
    if not config['sessions'] or not config['seeds'] or not config['methods']:
        raise ValueError('Empty session, seed, or method list.')
    for key in ('sessions', 'seeds', 'methods'):
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f'Duplicate {key}.')
    if set(config['methods']) - set(METHOD_LABELS):
        raise ValueError('Unknown method.')
    if not 0.1 < c['tau'] < 1:
        raise ValueError('tau must lie in (0.1, 1).')
    for n in (c['validation_draws'], e['draws']):
        if n < 2 or n % 2:
            raise ValueError('Forecast sample counts must be positive and even.')
    for key in ('steps', 'batch_size', 'validate_every', 'checkpoint_every', 'log_every',
                'validation_cases', 'validation_case_batch', 'checkpoint_seconds'):
        if c[key] <= 0:
            raise ValueError(f'training.{key} must be positive.')
    if not c['learning_rates'] or not c['glm_regularization']:
        raise ValueError('Empty validation search.')
    for key in ('cfm_nfe', 'fm_nfe', 'rollout_horizons'):
        if not e[key] or min(e[key]) < 1 or len(set(e[key])) != len(e[key]):
            raise ValueError(f'Invalid {key}.')
    if config['model']['transformer_width'] % config['model']['transformer_heads']:
        raise ValueError('Transformer width must be divisible by head count.')
    return config


def download_data(config, root):
    from huggingface_hub import snapshot_download
    target = Path(root) / config['data_root']
    snapshot_download(repo_id=config['dataset_repo'], repo_type='dataset',
                      revision=config['dataset_revision'], local_dir=str(target),
                      allow_patterns=['metadata.json', 'README.md'] +
                      [f'session_{s:03d}.npy' for s in config['sessions']])
    return target


@contextmanager
def output_lock(output):
    # POSIX advisory locking is released by the OS even if an HPC job is killed.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another notebook/script is using this neural output directory.') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def stop_signals():
    state = {'stop': False}
    previous = {}
    def request_stop(signum, frame):
        state['stop'] = True
        print('Pause requested; finishing the current update before saving.', flush=True)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, request_stop)
        yield lambda: state['stop']
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def fit_directory(output, session, seed, method, candidate_index):
    return output / 'checkpoints' / f'session_{session:03d}' / f'seed_{seed}' / method / f'candidate_{candidate_index}'


def choose_fit(method, session, seed, config, output):
    selected = []
    candidates = candidate_configs(method, config)
    for i, candidate in enumerate(candidates):
        path = fit_directory(output, session.index, seed, method, i) / 'selected.pt'
        if not path.exists():
            raise FileNotFoundError(f'Incomplete fit: {path}. Run training first.')
        result = load_checkpoint(path)
        expected = fit_signature(method, session, seed, candidate, config)
        if result['signature'] != expected or result['completed_steps'] != config['training']['steps']:
            raise ValueError(f'Checkpoint configuration/data/source mismatch: {path}')
        selected.append((result['best_score'], i, result, path))
    _, candidate_index, best, path = min(selected, key=lambda row: (row[0], row[1]))
    best['candidate_index'] = candidate_index
    best['search_training_seconds'] = sum(row[2]['training_seconds'] for row in selected)
    return best, path


def synchronization(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def ensemble_latency(model, method, session, config, indices, device, nfe, sampler):
    e = config['evaluation']
    # Sequential independent forecast requests; each request produces one ensemble.
    h, _ = session.batch(indices[:e['timing_cases']], 'cpu', split='test')
    durations = []
    with isolated_rng(e['seed'] + 777, device):
        for repeat in range(e['timing_repeats'] + 1):
            synchronization(device)
            start = time.perf_counter()
            for history in h:
                samples = forecast(model, method, history[None].to(device), e['draws'], nfe,
                                   sampler, config['training']['tau'])
                samples.cpu()
            synchronization(device)
            seconds = (time.perf_counter() - start) / len(h)
            if repeat:
                durations.append(seconds)
    return 1000 * float(np.median(durations))


def evaluation_source_signature(method=None):
    root = Path(__file__).parent
    # Forecasting/scoring are unchanged by the GLM training repair.
    hashes = {name: file_hash(root / name) for name in
              ('neural_experiment.py', 'neural_evaluation.py', 'neural_models.py', 'neural_data.py')}
    if method in ('direct', 'transformer_nb', 'glm'):
        hashes = original_compatible_hashes(hashes)
    else:
        hashes['neural_original.py'] = file_hash(root / 'neural_original.py')
    return fingerprint(compatible_source_hashes(hashes))


def evaluate_fit(method, session, seed, config, output, device, should_stop):
    e = config['evaluation']
    selected, checkpoint = choose_fit(method, session, seed, config, output)
    destination = output / 'evaluations' / f'session_{session.index:03d}' / f'seed_{seed}' / method
    destination.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({'checkpoint': file_hash(checkpoint), 'evaluation': e,
                             'source': evaluation_source_signature(method), 'data': session.signature})
    complete = destination / 'complete.json'
    if complete.exists():
        old = json.loads(complete.read_text())
        if old['signature'] != signature:
            raise ValueError(f'Evaluation provenance mismatch: {destination}. Use a new output directory.')
        print(f'[{method} s={session.index} seed={seed}] reuse evaluated results', flush=True)
        return
    model = build_model(method, session, config).to(device).eval()
    model.load_state_dict(selected['model'])
    validation_indices = session.indices('val', e['validation_cases'])
    test_indices = session.indices('test', e['test_cases'])
    if method == 'cfm':
        variants = [(n, 'map') for n in e['cfm_nfe']]
    elif method == 'fm':
        variants = [(n, sampler) for sampler in e['fm_samplers'] for n in e['fm_nfe']]
    else:
        variants = [(1, 'direct')]
    metadata = {'session': session.index, 'seed': seed, 'method': method,
                'parameters': selected['parameters'], 'selected_update': selected['best_step'],
                'candidate_index': selected['candidate_index'], 'completed_steps': selected['completed_steps'],
                'training_seconds': selected['training_seconds'],
                'search_training_seconds': selected['search_training_seconds'], 'n_neurons': session.dim}
    rows = []
    for nfe, sampler in variants:
        if should_stop():
            raise TrainingPaused('Evaluation paused; completed variants will be reused.')
        path = destination / f'{sampler}_{nfe}.pt'
        if path.exists():
            stored = load_checkpoint(path)
            if stored['signature'] != signature:
                raise ValueError(f'Partial evaluation mismatch: {path}')
            rows.append(stored['row'])
            continue
        print(f'[{method} s={session.index} seed={seed}] evaluate {sampler}, NFE={nfe}', flush=True)
        kwargs = dict(device=device, draws=e['draws'], nfe=nfe, sampler=sampler,
                      tau=config['training']['tau'], case_batch=e['case_batch'], seed=e['seed'])
        val, val_arrays = evaluate(model, method, session, validation_indices, split='val', **kwargs)
        test, test_arrays = evaluate(model, method, session, test_indices, split='test',
                                     shuffle=method in ('cfm', 'direct'), **kwargs)
        row = {**metadata, 'nfe': nfe, 'sampler': sampler, **test,
               'validation_energy': val['energy_score'],
               'ensemble_ms': ensemble_latency(model, method, session, config, test_indices, device, nfe, sampler)}
        stored = {'signature': signature, 'row': row, 'validation': val_arrays, 'test': test_arrays}
        atomic_torch_save(stored, path)
        rows.append(row)
    # Validation selection is independent of all test metrics.
    fm_best = min(rows, key=lambda row: (row['validation_energy'], row['nfe'], row['sampler'])) if method == 'fm' else None
    for row in rows:
        row['reported'] = method != 'fm' or (row['nfe'], row['sampler']) == (fm_best['nfe'], fm_best['sampler'])
    if fm_best and fm_best['nfe'] == max(e['fm_nfe']):
        print('Count-FM selected the largest tested NFE; results will record this without a saturation claim.', flush=True)
    rollout_rows = []
    rollout_indices = session.indices('test', e['rollout_cases'], max(e['rollout_horizons']))
    for row in rows:
        if not row['reported']:
            continue
        if should_stop():
            raise TrainingPaused('Evaluation paused; rerun to continue remaining rollouts.')
        path = destination / f'rollout_{row["sampler"]}_{row["nfe"]}.pt'
        if path.exists():
            stored = load_checkpoint(path)
            if stored['signature'] != signature:
                raise ValueError('Rollout provenance mismatch.')
        else:
            print(f'[{method} s={session.index} seed={seed}] autonomous rollout, NFE={row["nfe"]}', flush=True)
            values = evaluate_rollouts(model, method, session, rollout_indices, e['rollout_horizons'],
                                       device=device, draws=e['draws'], nfe=row['nfe'], sampler=row['sampler'],
                                       tau=config['training']['tau'], case_batch=e['rollout_case_batch'],
                                       seed=e['seed'] + 1)
            entries, arrays = [], {}
            for horizon, (summary, detail) in values.items():
                entries.append({**metadata, 'nfe': row['nfe'], 'sampler': row['sampler'],
                                'horizon_bins': horizon, 'horizon_ms': horizon * session.bin_width_ms,
                                'total_nfe': horizon * row['nfe'], **summary})
                arrays[horizon] = detail
            stored = {'signature': signature, 'rows': entries, 'scores': arrays, 'target_indices': rollout_indices}
            atomic_torch_save(stored, path)
        rollout_rows.extend(stored['rows'])
    # The trace is fixed before fitting and illustrative; scores use the full frozen set.
    if seed == e['figure_seed'] and session.index == e['figure_session']:
        trace_indices = session.indices('test')[:e['trace_bins']]
        for row in rows:
            if not row['reported']:
                continue
            pieces = []
            with isolated_rng(e['seed'] + 2, device):
                for first in range(0, len(trace_indices), e['case_batch']):
                    h, observed = session.batch(trace_indices[first:first+e['case_batch']], device, split='test')
                    x = forecast(model, method, h, e['draws'], row['nfe'], row['sampler'], config['training']['tau'])
                    a = x.float().mean(-1)
                    pieces.append(torch.stack([observed[:, 0].float().mean(-1), a.mean(1),
                                               a.quantile(0.05, dim=1), a.quantile(0.95, dim=1)], -1).cpu())
            atomic_torch_save({'values': torch.cat(pieces).numpy(), 'indices': trace_indices,
                              'signature': signature}, destination / f'trace_{row["sampler"]}_{row["nfe"]}.pt')
    atomic_json({'signature': signature, 'rows': rows, 'rollouts': rollout_rows,
                 'selected_checkpoint': str(checkpoint.relative_to(output)),
                 'fm_selected_maximum_nfe': bool(fm_best and fm_best['nfe'] == max(e['fm_nfe']))}, complete)


def run_experiment(config_path, *, root=None, device=None, stage='all', sessions=None, seeds=None):
    config_path = Path(config_path).resolve()
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    config = read_config(config_path)
    if stage not in ('all', 'train', 'evaluate', 'report'):
        raise ValueError('stage must be all/train/evaluate/report.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output = root / config['output_dir']
    session_ids = config['sessions'] if sessions is None else sessions
    seed_ids = config['seeds'] if seeds is None else seeds
    if not set(session_ids) <= set(config['sessions']) or not set(seed_ids) <= set(config['seeds']):
        raise ValueError('Requested sessions/seeds must belong to the frozen configuration.')
    if stage != 'report':
        data_root = root / config['data_root']
        required = [data_root / 'metadata.json'] + [data_root / f'session_{s:03d}.npy' for s in session_ids]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing neural data. Run --download-only first: ' + ', '.join(missing))
    with output_lock(output):
        config_snapshot = output / 'config_snapshot.json'
        migrate_original_config(output, config, stage=stage)
        if config_snapshot.exists() and json.loads(config_snapshot.read_text()) != config:
            raise ValueError('This output directory belongs to different settings. Restore the configuration or choose a new output directory.')
        atomic_json(config, config_snapshot)
        if stage == 'report':
            from .neural_reporting import make_report
            make_report(output, config)
            return output
        atomic_json({'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'device': str(device),
                     'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                     'cuda': torch.version.cuda, 'host': platform.node(), 'training_source': core_source_signature()},
                    output / f'environment_{time.time_ns()}.json')
        with stop_signals() as should_stop:
            for session_id in session_ids:
                session = SpikeSession(root / config['data_root'], session_id, config['history_bins'])
                provenance = {'signature': session.signature, 'metadata': session.metadata,
                              'n_neurons': session.dim, 'n_bins': session.n_bins, 'bounds': session.bounds}
                atomic_json(provenance, output / 'data_manifest' / f'session_{session_id:03d}.json')
                for seed in seed_ids:
                    for method in config['methods']:
                        if stage in ('all', 'train'):
                            for i, candidate in enumerate(candidate_configs(method, config)):
                                fit_candidate(method, session, seed, candidate, config,
                                              fit_directory(output, session_id, seed, method, i), device, should_stop)
                        if stage in ('all', 'evaluate'):
                            evaluate_fit(method, session, seed, config, output, device, should_stop)
            if stage in ('all', 'evaluate'):
                from .neural_reporting import make_report
                make_report(output, config, require_complete=False)
    return output
