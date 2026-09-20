"""Restore the manuscript method and keep earlier experiments separate."""
from __future__ import annotations

import copy
import json
from pathlib import Path


METHOD_PROTOCOL = 'diagonal_ck_online_absolute_v1'
DIRECT_METHODS = ('direct', 'transformer_nb', 'glm')

# Filled with hashes of the reviewed shared-file edits. Only direct predictors
# are eligible; changed count-model fits must always restart.
_SOURCE_ALIASES = {'neural_training.py': ('a1f24dc2ddbcfd852b708ba11f791769f72d5a9bbc67e9ba169bfa36f1da5dbe', 'e74aaac04cc1e71e648db13638b83f294ce3dc9324e76501852180f63d180f53'), 'neural_models.py': ('8e87c83bd702ec0359ef3707568c6aca7258a502780de4af24333cb1d0988d1a', '3a22fb0d29f3e9e059d8c6f079e0c29ce3a17be7066b0d8fd4dde587264300c3'), 'neural_experiment.py': ('a399f3ff5a32dbdea4d8bc5749a361d8bec50337b9f655bb6a3b858fa6cf108c', '1b2eb4d79fe5d0106a5453114bb08ee1d7a4e1698125b6385cc8bc4e7f58e5af')}


def original_compatible_hashes(hashes):
    result = dict(hashes)
    # Preserve the direct predictors across the two preceding reviewed edits,
    # without requiring installation of the withdrawn endpoint/support modules.
    legacy_support = {
        'neural_training.py': ('e74aaac04cc1e71e648db13638b83f294ce3dc9324e76501852180f63d180f53', '9858bc60a6b606b75c22e65ef903fc209430d6b83804986a4866e7d411d88158'),
        'neural_models.py': ('3a22fb0d29f3e9e059d8c6f079e0c29ce3a17be7066b0d8fd4dde587264300c3', '652554c8062ce5fbb5b6cd7e8006e8784ca42eedcf0e3b71a0be7b86c8647b61'),
        'neural_experiment.py': ('1b2eb4d79fe5d0106a5453114bb08ee1d7a4e1698125b6385cc8bc4e7f58e5af', 'b01e05ce296e487e49c7e130def41dda38149eee5a213e27a9df8c8108b52d18'),
    }
    legacy_endpoint = {
        'neural_training.py': ('9858bc60a6b606b75c22e65ef903fc209430d6b83804986a4866e7d411d88158', 'cd3ab27c4341f61ede04221a9a422e9fc6f3f678ada0f2b1cac2325093c87075'),
        'neural_experiment.py': ('b01e05ce296e487e49c7e130def41dda38149eee5a213e27a9df8c8108b52d18', '3663ed88b421f0f78a01db037c914ac8e4202c3e5cd914e281b7ebabe69b9986'),
    }
    for aliases in (_SOURCE_ALIASES, legacy_support, legacy_endpoint):
        for name, (updated, previous) in aliases.items():
            if result.get(name) == updated:
                result[name] = previous
    return result


def validate_original_method(config):
    train, model = config['training'], config['model']
    if train.get('endpoint_weight', 0.0) != 0.0:
        raise ValueError('The manuscript objective has no endpoint loss. Set endpoint_weight=0.')
    if model.get('source_conditioned_support', False) is not False:
        raise ValueError('The manuscript transition family has no source-direction masks.')
    if model.get('correction_time_scale') != 'absolute':
        raise ValueError('Use correction_time_scale=absolute, as in the manuscript kernel.')
    if train.get('ck_teacher') != 'online_stop_gradient':
        raise ValueError('The manuscript CK target uses the current weights with gradients stopped.')


def method_record(config):
    validate_original_method(config)
    return {'method_protocol': METHOD_PROTOCOL, 'purpose': config['purpose'],
            'count_flow_map_objective': 'diagonal_rate_matching + alpha * CK_cross_entropy',
            'count_fm_objective': 'diagonal_rate_matching',
            'endpoint_weight': 0.0, 'source_conditioned_support': False,
            'correction_time_scale': 'absolute', 'ck_teacher': 'online_stop_gradient',
            'ema_use': 'checkpoint selection and inference only'}


def _previous_configs(config):
    """The three shipped predecessors, with all unrelated settings fixed."""
    base = copy.deepcopy(config)
    base.pop('purpose', None)
    base['training'].pop('ck_teacher', None)
    base['training'].pop('endpoint_weight', None)
    base['training']['validation_nfe']['cfm'] = 1
    base['model'].pop('source_conditioned_support', None)
    base['model']['correction_time_scale'] = 'remaining'
    endpoint = copy.deepcopy(base)
    endpoint['training']['endpoint_weight'] = 1.0
    support = copy.deepcopy(endpoint)
    support['model']['source_conditioned_support'] = True
    support['training']['validation_nfe']['cfm'] = [1, 16]
    return (base, endpoint, support)


def migrate_original_config(output, config, *, stage):
    """Move old count-model files, reuse the three unchanged direct predictors.

    Called under the output lock. The journal is committed before any rename;
    the active configuration is committed after all renames. No copies/deletes.
    """
    from .neural_data import file_hash
    from .neural_training import atomic_json

    validate_original_method(config)
    output = Path(output)
    snapshot = output / 'config_snapshot.json'
    if not snapshot.exists():
        return
    previous = json.loads(snapshot.read_text())
    if previous == config:
        return
    if config.get('purpose') != 'paper' or previous not in _previous_configs(config):
        return  # Ordinary configuration validation rejects unrelated changes.
    if stage not in ('all', 'train'):
        raise ValueError('Restoring the original method requires fresh CFM and Count-FM training. '
                         'Run --stage all or --stage train. Existing results were retained.')
    source_root = Path(__file__).parent
    if not _SOURCE_ALIASES or any(file_hash(source_root / name) != updated
                                for name, (updated, _) in _SOURCE_ALIASES.items()):
        raise ValueError('Shared source files have further edits; refusing automatic migration.')

    backup = output / 'count_models_before_original_method'
    if backup.is_symlink():
        raise ValueError('The count-model archive must not be a symlink.')
    journal_path = backup / 'migration.json'
    old_snapshot = backup / 'config_snapshot.json'
    if old_snapshot.exists() and json.loads(old_snapshot.read_text()) != previous:
        raise ValueError('Conflicting archive configuration; no files moved.')
    candidates = {
        str(Path(group) / f'session_{session:03d}' / f'seed_{seed}' / method)
        for group in ('checkpoints', 'evaluations')
        for session in previous['sessions'] for seed in previous['seeds']
        for method in ('cfm', 'fm')
    }
    export_names = {'figures', 'all_metrics.csv', 'matched_nfe_metrics.csv',
                    'dependence_control.csv', 'rollout_metrics.csv', 'figure_captions.txt',
                    'report_status.json', 'method_protocol.json'}
    export = lambda name: name.startswith('paper_neural_') or name in export_names
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get('old_config') != previous or journal.get('new_config') != config:
            raise ValueError('Conflicting migration journal; no files moved.')
        paths = journal['paths']
        if len(paths) != len(set(paths)) or any(
                rel not in candidates and not (len(Path(rel).parts) == 1 and export(rel))
                for rel in paths):
            raise ValueError('Invalid migration paths; no files moved.')
    else:
        paths = sorted(rel for rel in candidates if (output / rel).exists())
        paths += sorted(p.name for p in output.iterdir() if export(p.name))
        journal = {'old_config': previous, 'new_config': config, 'paths': paths}
    for rel in paths:
        source, target = output / rel, backup / rel
        if (source.is_symlink() or target.is_symlink() or
                not source.resolve().is_relative_to(output.resolve()) or
                not target.resolve().is_relative_to(backup.resolve())):
            raise ValueError(f'Unsafe migration path: {rel}')
        if source.exists() and target.exists():
            raise ValueError(f'Archive already exists: {target}; no overwrite is allowed.')
        if not source.exists() and (not journal_path.exists() or not target.exists()):
            raise ValueError(f'Missing migration source and archive: {rel}')
    atomic_json(journal, journal_path)
    atomic_json(previous, old_snapshot)
    for rel in paths:
        source, target = output / rel, backup / rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    atomic_json({'reason': 'Restore the manuscript loss, kernel, and stop-gradient CK target',
                 'archived': paths, 'retrained_methods': ['cfm', 'fm'],
                 'reused_methods': list(DIRECT_METHODS)}, backup / 'repair.json')
    atomic_json(config, snapshot)
    print('Archived previous count models and reports in ' + str(backup) +
          '. CFM and Count-FM restart from fresh initialization; compatible direct baselines are reused.',
          flush=True)
