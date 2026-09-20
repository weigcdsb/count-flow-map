"""Keep resumable states only for unfinished neural fits; clean redundant files.

From the project root: python -m countflow.neural_storage [--apply] [--drop-old-glm]
The default is a preview. No fitting, sampling, or result rewriting is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import torch


_STORAGE_SOURCE_ALIASES = {
    'neural_training.py': ('cd3ab27c4341f61ede04221a9a422e9fc6f3f678ada0f2b1cac2325093c87075', '75014ae6078c8371042b1c2445ad91ac54013758fbcdb684975945925fea1f4f'),
    'neural_glm.py': ('854243721185084aa819d2e8a3276ee58970bbbe262f92c742c9c4f07791a6ad', '5b5008db5ddf3fda53ef5172371481594fa09ebefd292551d6cbbdf2b85aa0e0'),
}


def storage_compatible_hash(name, digest):
    # Only this exact storage-only edit is compatible. Training math/RNG are
    # unchanged; any other edit retains its own hash and fails normal checks.
    fixed, previous = _STORAGE_SOURCE_ALIASES.get(name, (None, None))
    return previous if digest == fixed else digest


def validate_selected(selected, signature, steps):
    if (selected.get('signature') != signature or selected.get('completed_steps') != steps
            or not 1 <= selected.get('best_step', 0) <= steps
            or not math.isfinite(selected.get('best_score', float('nan')))):
        raise ValueError('Selected checkpoint is incomplete or incompatible; existing files retained.')
    state = selected.get('model')
    if not isinstance(state, dict) or not state or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Selected checkpoint has invalid model weights; existing files retained.')


def release_completed_last(directory, selected, signature, steps, *, apply=False):
    """Remove last.pt only when its finished state agrees with valid selected.pt."""
    from .neural_training import load_checkpoint
    validate_selected(selected, signature, steps)
    path = Path(directory) / 'last.pt'
    if not path.exists():
        return 0
    if path.is_symlink():
        raise ValueError(f'Refusing to remove a symlink checkpoint: {path}')
    last = load_checkpoint(path)
    best = last.get('best_state')
    if (last.get('signature') != signature or last.get('step') != steps
            or last.get('best_step') != selected['best_step']
            or last.get('best_score') != selected['best_score']
            or not isinstance(best, dict) or best.keys() != selected['model'].keys()
            or any(not torch.equal(best[k], selected['model'][k]) for k in best)):
        raise ValueError(f'Last/selected checkpoint disagreement; retained {path}')
    size = path.stat().st_size
    if apply:
        path.unlink()
    return size


def _regular_inside(path, output):
    return path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(output)


def clean_output(output, *, apply=False, drop_old_glm=False):
    """Clean a registered run under its existing lock, without changing its config."""
    from .neural_experiment import output_lock
    from .neural_training import candidate_configs, fit_signature, load_checkpoint

    output = Path(output).resolve()
    if not (output / 'config_snapshot.json').is_file():
        raise FileNotFoundError(f'Missing {output / "config_snapshot.json"}; no files removed.')
    totals = {'completed_last': 0, 'temporary': 0, 'old_glm': 0}
    counts = dict.fromkeys(totals, 0)
    retained = 0
    with output_lock(output):
        config = json.loads((output / 'config_snapshot.json').read_text())
        for session_id in config['sessions']:
            manifest_path = output / 'data_manifest' / f'session_{session_id:03d}.json'
            if not manifest_path.is_file():
                continue
            session = SimpleNamespace(index=session_id,
                                      signature=json.loads(manifest_path.read_text())['signature'])
            for seed in config['seeds']:
                for method in config['methods']:
                    for i, candidate in enumerate(candidate_configs(method, config)):
                        directory = output / 'checkpoints' / f'session_{session_id:03d}' / f'seed_{seed}' / method / f'candidate_{i}'
                        last_path, selected_path = directory / 'last.pt', directory / 'selected.pt'
                        if not _regular_inside(last_path, output):
                            continue
                        if not _regular_inside(selected_path, output):
                            retained += 1
                            continue
                        try:
                            selected = load_checkpoint(selected_path)
                            if (selected.get('method'), selected.get('session'), selected.get('seed'),
                                    selected.get('candidate')) != (method, session_id, seed, candidate):
                                raise ValueError('Selected checkpoint labels do not match its directory.')
                            signature = fit_signature(method, session, seed, candidate, config)
                            size = release_completed_last(directory, selected, signature,
                                                          config['training']['steps'], apply=apply)
                        except Exception as error:
                            print(f'Kept {last_path}: {error}', flush=True)
                            retained += 1
                            continue
                        totals['completed_last'] += size
                        counts['completed_last'] += 1

        # Atomic writers use NAME.pt.PID.tmp or NAME.json.PID.tmp. With the run
        # lock held, these are abandoned temporary files, never resume states.
        for group in ('checkpoints', 'evaluations'):
            for path in (output / group).rglob('*.tmp'):
                if _regular_inside(path, output) and re.fullmatch(r'.+\.(?:pt|json)\.\d+\.tmp', path.name):
                    totals['temporary'] += path.stat().st_size
                    counts['temporary'] += 1
                    if apply:
                        path.unlink()

        if drop_old_glm:
            archive = output / 'glm_before_numerical_fix'
            snapshot = archive / 'config_snapshot.json'
            if snapshot.is_file():
                old = json.loads(snapshot.read_text())
                expected = copy.deepcopy(config)
                expected['training']['glm_learning_rate'] = 0.001
                if (config['training']['glm_learning_rate'] != 0.0001 or old != expected
                        or not (archive / 'repair.json').is_file()):
                    raise ValueError('Old GLM archive was not verified; its files were retained.')
                # Keep the tiny repair/config records. Only archived GLM tensors
                # are obsolete; active GLM checkpoints are outside this tree.
                for group in ('checkpoints', 'evaluations'):
                    for session_id in old['sessions']:
                        for seed in old['seeds']:
                            directory = archive / group / f'session_{session_id:03d}' / f'seed_{seed}' / 'glm'
                            for path in directory.rglob('*.pt'):
                                if _regular_inside(path, output):
                                    totals['old_glm'] += path.stat().st_size
                                    counts['old_glm'] += 1
                                    if apply:
                                        path.unlink()
        verb = 'Freed' if apply else 'Can free'
        for key, size in totals.items():
            print(f'{verb} {size / 2**30:.3f} GiB: {counts[key]} {key} files', flush=True)
        print(f'{verb} {sum(totals.values()) / 2**30:.3f} GiB in total. '
              f'Kept {retained} unfinished/unverified last.pt files. '
              'All active selected.pt files and evaluation results are retained.', flush=True)
    return {'bytes': totals, 'files': counts, 'retained_last': retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'outputs/neural_paper_v2')
    parser.add_argument('--apply', action='store_true', help='Delete only the verified redundant files.')
    parser.add_argument('--drop-old-glm', action='store_true', help='Also delete tensor files in the verified pre-repair GLM archive.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    clean_output(args.output, apply=args.apply, drop_old_glm=args.drop_old_glm)


if __name__ == '__main__':
    main()
