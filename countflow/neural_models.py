"""Existing Count Flow Map/Count-FM plus direct probabilistic forecasters."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .conditional_model import ConditionalCountFlowMap, ConditionalCountRateModel, HistoryGRUEncoder
from .distributions import PoissonBinomialResidualMixture, ResidualMixtureParams
from .model import MLP


class CachedHistoryEncoder(HistoryGRUEncoder):
    def forward(self, history):
        # Only generation supplies already encoded context. Raw data are always 3D.
        if history.ndim == 2 and history.shape[1] == self.output_dim:
            return history
        return super().forward(history)


class DirectCountMixture(nn.Module):
    def __init__(self, dim, encoder, width, depth, mixtures):
        super().__init__()
        self.context_encoder, self.dim, self.mixtures = encoder, dim, mixtures
        self.state_encoder = MLP(2 * dim, width, width, depth=2)
        self.head = MLP(width + encoder.output_dim, width, mixtures * (2 * dim + 1), depth)
        self.distribution = PoissonBinomialResidualMixture(death_chunk_size=8)

    def params(self, history):
        x = history[:, -1]
        state = self.state_encoder(torch.cat([torch.log1p(x) / math.log(5), x / 4], -1))
        raw = self.head(torch.cat([state, self.context_encoder(history)], -1))
        logits, raw = raw[:, :self.mixtures], raw[:, self.mixtures:]
        raw = raw.reshape(-1, self.mixtures, 2, self.dim)
        return ResidualMixtureParams(logits, F.softplus(raw[:, :, 0]) + 1e-5,
                                    torch.sigmoid(raw[:, :, 1]).clamp(1e-7, 1 - 1e-7), torch.ones(len(x), device=x.device))

    def nll(self, history, target):
        return -self.distribution.log_prob(target, history[:, -1].long(), self.params(history)).mean()


class TransformerNB(nn.Module):
    def __init__(self, dim, history_bins, width, layers, heads, dropout):
        super().__init__()
        self.input = nn.Linear(2 * dim, width)
        self.position = nn.Parameter(torch.zeros(1, history_bins, width))
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        # Initialize cloned layers independently.
        for layer in self.transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, 2 * dim)
        self.register_buffer('mask', torch.ones(history_bins, history_bins, dtype=torch.bool).triu(1))

    def params(self, history):
        h = self.input(torch.cat([torch.log1p(history) / math.log(5), history / 4], -1))
        h = self.transformer(h + self.position[:, :h.shape[1]], mask=self.mask[:h.shape[1], :h.shape[1]])
        mean_raw, dispersion_raw = self.head(self.norm(h[:, -1])).chunk(2, -1)
        return F.softplus(mean_raw) + 1e-5, F.softplus(dispersion_raw) + 1e-4

    def nll(self, history, target):
        mean, dispersion = self.params(history)
        distribution = torch.distributions.NegativeBinomial(
            total_count=dispersion, logits=mean.log() - dispersion.log())
        return -distribution.log_prob(target).sum(-1).mean()


class PopulationGLM(nn.Module):
    def __init__(self, dim, history_bins, mean, std):
        super().__init__()
        self.register_buffer('center', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(std, dtype=torch.float32))
        self.linear = nn.Linear(history_bins * dim, dim)
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.linear.bias.copy_(self.center.clamp_min(1e-4).log())

    def params(self, history):
        return self.linear(((history - self.center) / self.scale).flatten(1)).exp()

    def nll(self, history, target):
        return F.poisson_nll_loss(self.params(history), target.float(), log_input=False,
                                  full=True, reduction='none').sum(-1).mean()


def build_model(method, session, config):
    m = config['model']
    if method in ('cfm', 'fm') and m.get('source_conditioned_support', False):
        raise ValueError('The manuscript protocol does not use source-direction masks.')
    if method in ('cfm', 'fm', 'direct'):
        encoder = CachedHistoryEncoder(session.dim, m['context_width'], m['context_width'],
                                       m['context_layers'], count_scale=4.0)
        args = dict(dim=session.dim, context_encoder=encoder, context_dim=m['context_width'],
                    hidden_dim=m['width'], depth=m['depth'], count_scale=4.0)
        if method == 'cfm':
            if m['correction_time_scale'] != 'absolute':
                raise ValueError('The manuscript kernel uses absolute interval corrections.')
            return ConditionalCountFlowMap(**args, n_mixtures=m['mixtures'], death_chunk_size=8,
                              correction_time_scale=m['correction_time_scale'])
        if method == 'fm':
            return ConditionalCountRateModel(**args)
        return DirectCountMixture(session.dim, encoder, m['width'], m['depth'], m['mixtures'])
    if method == 'transformer_nb':
        return TransformerNB(session.dim, session.history_bins, m['transformer_width'],
                             m['transformer_layers'], m['transformer_heads'], m['dropout'])
    if method == 'glm':
        return PopulationGLM(session.dim, session.history_bins, session.mean, session.std)
    raise ValueError(f'Unknown method {method}')


def expand_params(params, draws):
    return ResidualMixtureParams(*(v.repeat_interleave(draws, 0) for v in
                                  (params.mixture_logits, params.birth_mean, params.death_prob, params.delta)))


@torch.no_grad()
def forecast(model, method, history, draws, nfe=1, sampler='tau', tau=0.98):
    """Return integer samples [cases, draws, neurons], encoding each history once."""
    if draws < 1 or nfe < 1 or not 0 < tau < 1:
        raise ValueError('Invalid draw count, NFE, or endpoint cutoff.')
    batch, _, dim = history.shape
    x = history[:, -1].long().repeat_interleave(draws, 0)
    if method == 'direct':
        x = model.distribution.sample(x, expand_params(model.params(history), draws))
    elif method == 'transformer_nb':
        mean, dispersion = model.params(history)
        dist = torch.distributions.NegativeBinomial(total_count=dispersion,
                                                     logits=mean.log() - dispersion.log())
        x = dist.sample((draws,)).permute(1, 0, 2).long().reshape(batch * draws, dim)
    elif method == 'glm':
        x = torch.poisson(model.params(history)[:, None].expand(-1, draws, -1)).long().reshape(batch * draws, dim)
    elif method in ('cfm', 'fm'):
        context = model.context_encoder(history)
        context = context.repeat_interleave(draws, 0)
        h = tau / nfe
        for step in range(nfe):
            s = torch.full((len(x),), step * h, device=x.device)
            if method == 'cfm':
                x = model.sample(x, s, s + h, context)
            else:
                birth, death, coefficient = model.local_rates(x, s, context)
                if sampler == 'tau':
                    probability = (-torch.expm1(-h * coefficient)).clamp(0, 1)
                    x = x + torch.poisson(h * birth).long() - torch.distributions.Binomial(
                        total_count=x.float(), probs=probability).sample().long()
                elif sampler == 'unit':
                    total = birth + death
                    p = -torch.expm1(-h * total)
                    u = torch.rand_like(total)
                    x = x + (u > 1 - p * birth / total.clamp_min(1e-12)).long()
                    x = x - (u < p * death / total.clamp_min(1e-12)).long()
                else:
                    raise ValueError(sampler)
    else:
        raise ValueError(method)
    if (x < 0).any():
        raise FloatingPointError('Negative forecast counts.')
    return x.reshape(batch, draws, dim)
