"""Compact manuscript tables and figures from completed, verified evaluations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .neural_data import file_hash, fingerprint
from .neural_evaluation import LEVELS
from .neural_training import atomic_json, load_checkpoint

METRICS = ['energy_score', 'population_crps', 'neuron_rmse', 'coverage_90', 'width_90',
           'ensemble_ms', 'parameters', 'training_seconds', 'search_training_seconds']
ORDER = ['Count Flow Map (1 NFE)', 'Count Flow Map (16 NFE)', 'Count-FM (validation selected)',
         'Direct count mixture', 'Transformer-NB', 'Population Poisson GLM']
COLORS = ['#0072B2', '#56B4E9', '#D55E00', '#009E73', '#CC79A7', '#777777']
TABLE_DIGITS = {'energy_score': 4, 'population_crps': 5, 'neuron_rmse': 4, 'ensemble_ms': 2}


def label(row):
    from .neural_experiment import METHOD_LABELS
    if row['method'] == 'cfm':
        return f'Count Flow Map ({int(row["nfe"])} NFE)'
    if row['method'] == 'fm':
        return 'Count-FM (validation selected)'
    return METHOD_LABELS[row['method']]


def summarize(frame, group):
    metrics = [m for m in METRICS if m in frame]
    out = frame.groupby(group, sort=False)[metrics].agg(['mean', 'std'])
    out.columns = ['_'.join(c) for c in out.columns]
    return out.reset_index()


def fmt(mean, std, digits=3):
    return f'{mean:.{digits}f}' if pd.isna(std) else f'{mean:.{digits}f} $\\pm$ {std:.{digits}f}'


def tex_table(summary, budgets, caption, table_label, session_column=False):
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{' + ('r' if session_column else '') + 'llrrrr}', r'\toprule',
             ('Session & ' if session_column else '') +
             r'Method & NFE & Energy $\downarrow$ & Pop. CRPS $\downarrow$ & RMSE $\downarrow$ & Time (ms) $\downarrow$ \\',
             r'\midrule']
    for _, row in summary.iterrows():
        name = row['label']
        values = [fmt(row[f'{m}_mean'], row[f'{m}_std'], TABLE_DIGITS[m])
                  for m in ('energy_score', 'population_crps', 'neuron_rmse', 'ensemble_ms')]
        prefix = [str(int(row['session']))] if session_column else []
        lines.append(' & '.join(prefix + [name.replace('_', r'\_'), budgets[name]] + values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}', r'\caption{' + caption + '}',
                  r'\label{' + table_label + '}', r'\end{table}'])
    return '\n'.join(lines) + '\n'


def save_figure(fig, directory, name):
    for extension in ('pdf', 'png'):
        fig.savefig(directory / f'{name}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def rollout_figure(rollout, labels):
    """Full-range summaries and display-only zooms using identical input scores."""
    metrics = ['energy_score', 'population_crps']
    names = ['Energy score', 'Population CRPS']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    pooled = {metric: [] for metric in metrics}
    positive = {metric: [] for metric in metrics}
    all_positive = {metric: True for metric in metrics}
    for name in labels:
        sub = rollout[rollout.label == name]
        if sub.empty:
            continue
        by_seed = sub.groupby(['horizon_ms', 'seed'])[metrics].mean()
        means = by_seed.groupby('horizon_ms').mean()
        sds = by_seed.groupby('horizon_ms').std().fillna(0)
        color = COLORS[ORDER.index(name)] if name in ORDER else None
        x = means.index.to_numpy()
        for column, metric in enumerate(metrics):
            full, detail = axes[:, column]
            values = by_seed[metric].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite rollout scores for {name}: {metric}')
            pooled[metric].extend(means[metric].to_numpy())
            positive[metric].extend(values[values > 0])
            all_positive[metric] &= bool((values > 0).all())
            for _, run in by_seed[metric].groupby(level='seed'):
                run = run.droplevel('seed').sort_index()
                full.plot(run.index, run.values, color=color, alpha=.25, lw=.7)
            full.plot(x, means[metric], 'o-', ms=3, color=color, label=name)
            detail.plot(x, means[metric], 'o-', ms=3, color=color)
            detail.fill_between(x, means[metric]-sds[metric], means[metric]+sds[metric],
                                color=color, alpha=.12)
    for column, (metric, axis_name) in enumerate(zip(metrics, names)):
        full, detail = axes[:, column]
        if not pooled[metric]:
            raise ValueError(f'No rollout values for {metric}')
        if all_positive[metric]:
            full.set_yscale('log')
            scale_name = 'log scale'
        else:
            # Empirical score estimates can include zero; keep them in view.
            threshold = min(positive[metric]) if positive[metric] else 1.0
            full.set_yscale('symlog', linthresh=threshold)
            scale_name = 'symlog scale'
        q1, q3 = np.quantile(pooled[metric], [.25, .75])
        spread = max(q3-q1, abs(q3)*.01, 1e-12)
        lower, upper = q1-1.5*spread, q3+1.5*spread
        if min(pooled[metric]) >= 0:
            lower = max(0, lower)
        detail.set_ylim(lower, upper)
        full.set(title=f'{chr(65+column)}  {axis_name} ({scale_name})', ylabel=axis_name)
        detail.set(title=f'{chr(67+column)}  {axis_name} detail', ylabel=axis_name,
                   xlabel='Physical forecast horizon (ms)')
        detail.text(.02, .97, 'Same results · linear zoom', va='top', fontsize=8,
                    transform=detail.transAxes)
        for ax in (full, detail):
            ax.grid(axis='y', alpha=.15)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Autonomous neural forecasts', ha='center')
    return fig


def make_report(output, config, require_complete=True):
    from .neural_experiment import evaluation_source_signature
    from .neural_original import method_record
    output = Path(output)
    protocol = method_record(config)
    if config['purpose'] != 'paper':
        atomic_json({**protocol, 'status': 'diagnostic_only', 'paper_outputs_written': False},
                    output / 'diagnostic_status.json')
        print('Diagnostic run only. No paper tables or figures are exported.', flush=True)
        return
    rows, rollouts, missing, boundary = [], [], [], []
    for session in config['sessions']:
        for seed in config['seeds']:
            for method in config['methods']:
                path = output / 'evaluations' / f'session_{session:03d}' / f'seed_{seed}' / method / 'complete.json'
                if not path.exists():
                    missing.append(str(path.relative_to(output)))
                    continue
                data = json.loads(path.read_text())
                manifest = json.loads((output / 'data_manifest' / f'session_{session:03d}.json').read_text())
                expected = fingerprint({'checkpoint': file_hash(output / data['selected_checkpoint']),
                                        'evaluation': config['evaluation'], 'source': evaluation_source_signature(method),
                                        'data': manifest['signature']})
                if expected != data['signature']:
                    raise ValueError(f'Refusing a report from incompatible evaluations: {path}')
                if method in ('cfm', 'fm'):
                    selected = load_checkpoint(output / data['selected_checkpoint'])
                    if (selected.get('method_record') != protocol or
                            selected.get('completed_steps') != config['training']['steps']):
                        raise ValueError(f'Refusing a report from a different training method: {path}')
                    del selected
                if not data['rows'] or any((r['session'], r['seed'], r['method']) != (session, seed, method)
                                          for r in data['rows']):
                    raise ValueError(f'Mislabeled evaluation rows: {path}')
                rows.extend(data['rows'])
                rollouts.extend(data['rollouts'])
                if data['fm_selected_maximum_nfe']:
                    boundary.append({'session': session, 'seed': seed})
    atomic_json({**protocol, 'status': 'incomplete' if missing else 'complete', 'missing': missing,
                 'fm_selected_maximum_nfe': boundary, 'n_sessions': len(config['sessions']),
                 'n_seeds': len(config['seeds'])}, output / 'report_status.json')
    if missing:
        message = f'{len(missing)} evaluations remain; manuscript reports are produced only when all configured runs finish.'
        if require_complete:
            raise RuntimeError(message)
        print(message, flush=True)
        return
    atomic_json(protocol, output / 'method_protocol.json')
    table = pd.DataFrame(rows)
    table['label'] = table.apply(label, axis=1)
    table.to_csv(output / 'all_metrics.csv', index=False)
    reported = table[table['reported']].copy()
    keys = ['session', 'seed', 'label']
    if reported.duplicated(keys).any():
        raise ValueError('Duplicate reported run.')
    # Equal session weights, then SD across independently trained seeds.
    seed_means = reported.groupby(['label', 'seed'], sort=False)[METRICS].mean().reset_index()
    summary = summarize(seed_means, ['label'])
    labels = [name for name in ORDER if name in set(summary.label)]
    labels += [name for name in summary.label if name not in labels]
    rank = {name: i for i, name in enumerate(labels)}
    summary = summary.sort_values('label', key=lambda s: s.map(rank))
    per_session = summarize(reported, ['session', 'label']).sort_values(['session', 'label'])
    summary.to_csv(output / 'paper_neural_table.csv', index=False)
    per_session.to_csv(output / 'paper_neural_sessions.csv', index=False)
    seed_means.to_csv(output / 'paper_neural_seed_means.csv', index=False)
    budgets = {}
    for name, group in reported.groupby('label'):
        ns = group.nfe.to_numpy()
        budgets[name] = ('1-pass' if group.method.iloc[0] not in ('cfm', 'fm') else
                         str(int(ns[0])) if ns.min() == ns.max() else f'{int(ns.min())}--{int(ns.max())}')
    caption = ('Neural population forecasting from 500 ms of spike history to the next 50-ms bin. '
               f'Results average {len(config["sessions"])} sessions equally within each training seed, '
               f'then report mean and sample SD across {len(config["seeds"])} seeds. '
               'Energy score uses Euclidean distances divided by the square root of the neuron count; '
               'population CRPS uses mean count per neuron. RMSE estimates error of the predictive mean '
               'with a finite-ensemble variance correction. Count-FM sampler and NFE are selected on validation data; '
               'ranges reflect selections across sessions and seeds. Time is median latency per forecast request '
               f'for {config["evaluation"]["draws"]} joint samples, including history encoding and device transfers, '
               f'measured over {config["evaluation"]["timing_repeats"]} timed repetitions after warmup. SD measures training variability, not biological replication.')
    (output / 'paper_neural_table.tex').write_text(tex_table(summary, budgets, caption, 'tab:neural-forecast'))
    matched = table[table.nfe.isin([1, 16]) & table.method.isin(['cfm', 'fm'])].copy()
    matched.loc[matched.method == 'fm', 'label'] = matched[matched.method == 'fm'].apply(
        lambda r: f'Count-FM ({r["sampler"]}, {int(r["nfe"])} NFE)', axis=1)
    matched.to_csv(output / 'matched_nfe_metrics.csv', index=False)
    matched_seed = matched.groupby(['label', 'seed'], sort=False)[METRICS].mean().reset_index()
    matched_summary = summarize(matched_seed, ['label'])
    matched_summary.to_csv(output / 'paper_neural_matched_nfe.csv', index=False)
    matched_budgets = {name: str(int(group.nfe.iloc[0])) for name, group in matched.groupby('label')}
    appendix_caption = ('Matched-budget next-bin forecasting at 1 and 16 NFE. Unit denotes the '
                        'at-most-one-unit-jump sampler; tau denotes the binomial tau-leap sampler. '
                        'Both Count-FM samplers use the same selected network. Each method retains the '
                        'same selected weights across budgets. Aggregation and metrics match the main table.')
    (output / 'paper_neural_matched_nfe.tex').write_text(tex_table(
        matched_summary, matched_budgets, appendix_caption, 'tab:neural-matched-nfe'))
    dependence = reported[reported.method.isin(['cfm', 'direct'])].copy()
    dependence['crps_increase_after_shuffle'] = dependence.shuffled_population_crps - dependence.population_crps
    dependence.to_csv(output / 'dependence_control.csv', index=False)
    rollout = pd.DataFrame(rollouts)
    rollout['label'] = rollout.apply(label, axis=1)
    rollout.to_csv(output / 'rollout_metrics.csv', index=False)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
        e = config['evaluation']
        trace = output / 'evaluations' / f'session_{e["figure_session"]:03d}' / f'seed_{e["figure_seed"]}' / 'cfm' / 'trace_map_1.pt'
        if trace.exists():
            values = load_checkpoint(trace)['values']
            time_axis = np.arange(len(values)) * 0.05
            axes[0].fill_between(time_axis, values[:, 2], values[:, 3], color=COLORS[0], alpha=.2, label='90% interval')
            axes[0].plot(time_axis, values[:, 1], color=COLORS[0], lw=1, label='Forecast mean')
            axes[0].plot(time_axis, values[:, 0], color='black', lw=.7, label='Observed')
            axes[0].legend(fontsize=8, frameon=False)
        else:
            axes[0].text(.5, .5, 'Fixed illustrative run unavailable', ha='center', transform=axes[0].transAxes)
        axes[0].set(xlabel='Time in test segment (s)', ylabel='Mean count per neuron',
                    title='A  Next-bin forecast · 1 NFE')
        levels = np.asarray(LEVELS)
        axes[1].plot([.45, 1], [.45, 1], '--', color='black', lw=.8)
        for name in ['Count Flow Map (1 NFE)', 'Count Flow Map (16 NFE)', 'Count-FM (validation selected)', 'Direct count mixture', 'Transformer-NB']:
            sub = reported[reported.label == name]
            if sub.empty:
                continue
            columns = [f'coverage_{int(level*100)}' for level in levels]
            by_seed = sub.groupby('seed')[columns].mean()
            mean, sd = by_seed.mean().values, by_seed.std().fillna(0).values
            color = COLORS[ORDER.index(name)]
            axes[1].plot(levels, mean, 'o-', ms=3, color=color, label=name)
            axes[1].fill_between(levels, mean-sd, mean+sd, color=color, alpha=.12)
        axes[1].set(xlabel='Nominal coverage', ylabel='Randomized-rank coverage',
                    title='B  Population calibration')
        axes[1].legend(fontsize=7, frameon=False)
        axes[2].axhline(0, color='black', lw=.8)
        for i, name in enumerate(['Count Flow Map (1 NFE)', 'Count Flow Map (16 NFE)', 'Direct count mixture']):
            sub = dependence[dependence.label == name]
            if sub.empty:
                continue
            by_session = sub.groupby('session').crps_increase_after_shuffle.mean()
            offsets = np.linspace(-.12, .12, len(by_session))
            axes[2].scatter(i+offsets, by_session, color=COLORS[ORDER.index(name)], s=22)
            axes[2].plot([i-.18, i+.18], [by_session.mean()]*2, color='black', lw=2)
        axes[2].set_xticks([0, 1, 2], ['Flow Map\n1 NFE', 'Flow Map\n16 NFE', 'Direct\nmixture'])
        axes[2].set(ylabel='CRPS increase after shuffling', title='C  Does dependence help?')
        save_figure(fig, figures, 'paper_neural_main')
        fig = rollout_figure(rollout, labels)
        save_figure(fig, figures, 'paper_neural_appendix')
    captions = ('Main figure: A shows a fixed session and training run, forecasting each next bin from observed history; '
                'the band describes count uncertainty, not uncertainty in an estimated mean. '
                'B uses randomized ensemble ranks to handle count ties; lines average sessions equally, '
                'with SD across training seeds. C preserves each neuron\'s predictive marginal while independently '
                'permuting sample indices within each history. Each dot is one session averaged over seeds. '
                'Positive differences favor intact dependence; negative differences remain visible. '
                'Session means can be strongly affected by extreme forecasts, so a larger shuffle effect alone '
                'does not establish better calibrated or more accurate dependence.\n\n'
                'Appendix figure: Starting from observed history, all subsequent histories contain generated counts. '
                'Scores evaluate the population vector at each physical horizon, averaging sessions equally '
                'within each training seed. A and B show the full range for every method, with thin lines for '
                'individual seed averages and thick lines for their means. C and D show the same means on '
                'linear axes, with bands of one sample SD across training seeds. The detail limits use '
                'Q1 minus 1.5 interquartile ranges and Q3 plus 1.5 interquartile ranges, pooling all '
                'method-by-horizon means for each metric; a small minimum spread handles constant scores. '
                'These display-only zooms do '
                'not remove runs or alter any score. NFE labels refer to each 50-ms transition, so total '
                'rollout NFE grows with horizon.\n')
    (output / 'figure_captions.txt').write_text(captions)
    print(f'Paper tables and figures saved to {output}', flush=True)
