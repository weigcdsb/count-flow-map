import copy
import itertools
import random

import numpy as np
import pytest
import torch

from countflow.application_data import make_scrna_transport_surrogate
from countflow.application_metrics import raw_effect_cross_moments
from countflow.application_training import ApplicationTrainConfig, train_conditional_flow_map, train_conditional_count_fm
from countflow.conditional_model import ConditionalCountFlowMap, ConditionalCountRateModel, VectorContextEncoder
from countflow.validation import ValidationSelector


@pytest.mark.parametrize('kind', ['flow', 'fm'])
def test_validation_does_not_change_training_and_restores_best_ema(kind):
    torch.set_num_threads(2)
    bundle = make_scrna_transport_surrogate(dim=4, n_cell_lines=1, n_drugs=2, cells_per_condition=20)
    cls = ConditionalCountFlowMap if kind == 'flow' else ConditionalCountRateModel
    train = train_conditional_flow_map if kind == 'flow' else train_conditional_count_fm
    torch.manual_seed(77)
    model = cls(4, VectorContextEncoder(4, 8, 8), 8, hidden_dim=8, depth=2)
    other = copy.deepcopy(model)
    cfg = ApplicationTrainConfig(steps=6, batch_size=4, seed=71, log_every=6)
    online0, final0, hist0 = train(model, bundle.train, cfg, device='cpu', verbose=False)
    states = []
    def score(ema):
        # Validation is deliberately allowed to consume all three global RNGs.
        random.seed(998); random.random(); np.random.seed(998); np.random.random(20)
        torch.manual_seed(998); torch.rand(20)
        states.append({k: v.clone() for k, v in ema.state_dict().items()})
        return [3., 1., 2.][len(states)-1]
    selector = ValidationSelector(score, every=2, steps=6, verbose=False)
    online1, final1, hist1 = train(other, bundle.train, cfg, device='cpu', verbose=False, checkpoint_callback=selector)
    assert hist0 == hist1
    for a, b in [(online0, online1), (final0, final1)]:
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    selector.add_history(hist1); selector.restore(final1)
    assert hist1['selected_step'] == [4]
    assert all(torch.equal(v, final1.state_dict()[k]) for k, v in states[1].items())


def test_cross_moments_have_correct_expectations_with_shared_controls():
    # Enumerate two independent row triplets exactly. Shared C makes ordinary
    # observed/predicted products biased, while cross-row products are unbiased.
    expected = dict.fromkeys(raw_effect_cross_moments([[0],[0]], [[0],[0]], [[0],[0]]), 0.)
    probabilities = [.5, .75, .6]  # C, T, G are Bernoulli; both effects subtract C.
    for bits in itertools.product([0, 1], repeat=6):
        rows = np.array(bits).reshape(2, 3)
        prob = np.prod([[p if bit else 1-p for bit,p in zip(row, probabilities)] for row in rows])
        actual = raw_effect_cross_moments(rows[:,2:3], rows[:,1:2], rows[:,0:1])
        for key in expected:
            expected[key] += prob * actual[key]
    assert expected['raw_effect_cross_signal'] == pytest.approx(.25**2)
    assert expected['raw_effect_cross_alignment'] == pytest.approx(.25*.1)
    assert expected['raw_effect_cross_prediction'] == pytest.approx(.1**2)
    assert expected['raw_effect_cross_mse'] == pytest.approx((.25-.1)**2)
    negative = raw_effect_cross_moments([[1],[0]], [[1],[0]], [[0],[1]])
    assert negative['raw_effect_cross_signal'] == -1.  # Never clip noise away.
