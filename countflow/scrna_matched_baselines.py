"""Opt-in matched main MLPs and resumable baseline checkpoint selection.

The original baseline builders and all count-model definitions remain unchanged.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .model import MLP
from . import scrna_response_baselines as base

VERSION = "scrna-matched-baselines-v1"
KINDS = ("nbvae", "cpa", "scgen_vae")
LABELS = {"nbvae": (base.NBVAE_LABEL,), "cpa": (base.CPA_LABEL,),
          "scgen_vae": (base.SCGEN_LABEL, base.SCVIDR_LABEL)}
STEMS = {base.NBVAE_LABEL: "nbvae", base.CPA_LABEL: "cpa",
         base.SCGEN_LABEL: "scgen", base.SCVIDR_LABEL: "scvidr"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Resume this CUDA training run on its original device.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def scoring_mode(model):
    state, mode = rng_state(), model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        restore_rng(state)
        model.train(mode)


def hidden_mlp(input_dim, width, layers):
    modules = []
    for _ in range(layers):
        modules.extend([nn.Linear(input_dim, width), nn.SiLU()])
        input_dim = width
    return nn.Sequential(*modules)


def build_matched(kind, bundle, architecture, config):
    if (architecture["activation"] != "silu" or architecture["batch_norm"]
            or float(architecture["dropout"]) != 0):
        raise ValueError("Matched main blocks require SiLU, no BatchNorm, and zero dropout.")
    width, layers, latent = (int(architecture[k]) for k in ("hidden_dim", "hidden_layers", "latent_dim"))
    if min(width, layers, latent) < 1:
        raise ValueError("Architecture dimensions must be positive.")
    depth = layers + 1  # countflow.model.MLP includes the output layer in depth.
    if kind == "scgen_vae":
        model = base.ScGenVAE(bundle.dim, hidden_dim=width, latent_dim=latent,
                              depth=layers, dropout=0., kl_weight=float(config.get("kl_weight", 5e-5)))
        model.encoder = hidden_mlp(bundle.dim, width, layers)
        model.decoder_hidden = hidden_mlp(latent, width, layers)
    elif kind == "cpa":
        cells, drugs, _ = base._categorical_layout(bundle)
        model = base.CPAModel(bundle.dim, len(drugs), len(cells), latent_dim=latent,
                             autoencoder_width=width, autoencoder_depth=layers,
                             **{k: int(config[k]) for k in ("adversary_width", "adversary_depth", "doser_width", "doser_depth")})
        model.encoder = MLP(bundle.dim, width, latent, depth=depth)
        model.decoder = MLP(latent, width, 2 * bundle.dim, depth=depth)
    elif kind == "nbvae":
        context_dim = int(architecture["context_output_dim"])
        model = base.ConditionalNBVAE(bundle.dim, bundle.train.context.shape[1],
                                     hidden_dim=width, latent_dim=latent, depth=depth,
                                     context_hidden_dim=int(architecture["context_hidden_dim"]),
                                     context_output_dim=context_dim)
        model.prior_net = MLP(width + context_dim, width, 2 * latent, depth=depth)
        model.posterior_net = MLP(2 * width + context_dim, width, 2 * latent, depth=depth)
    else:
        raise ValueError("Only application baselines are supported: " + kind)
    return model


def total_steps(kind, train, config):
    if kind == "scgen_vae" and config.get("epochs") is not None:
        return int(config["epochs"]) * max(1, int(np.ceil(2 * train.n / int(config["batch_size"]))))
    return int(config["steps"])


class BaselineRun:
    def __init__(self, directory, identity, score, checkpoints=10, verbose=True):
        self.directory = Path(directory)
        self.identity = identity
        self.signature = signature(identity)
        self.score = score
        self.checkpoints = int(checkpoints)
        self.verbose = verbose
        self.records, self.best = [], {}
        self.steps = None
        self.last_step = 0

    @property
    def resume_path(self):
        return self.directory / "resume.pt"

    def completed(self):
        path = self.directory / "training_complete.json"
        if not path.is_file():
            return False
        data = json.loads(path.read_text())
        self._check_signature(data)
        for filename, digest in data["files"].items():
            file = self.directory / filename
            if not file.is_file() or sha256(file) != digest:
                raise ValueError(f"Completed baseline file is missing/changed: {file}. Restore it or use a new baseline output directory.")
        return True

    def _check_signature(self, data):
        if data.get("signature") != self.signature:
            raise ValueError(f"Incompatible baseline cache: {self.directory}. Use a new --baseline-output directory for changed settings/code.")

    def begin(self, model, optimizers, generator, history, steps):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.steps = int(steps)
        self.schedule = set(np.ceil(np.linspace(0, steps, min(self.checkpoints, steps) + 1)[1:]).astype(int).tolist())
        if self.resume_path.is_file():
            data = torch.load(self.resume_path, map_location="cpu", weights_only=False)
            self._check_signature(data)
            if int(data["steps"]) != self.steps:
                raise ValueError("The baseline training budget changed.")
            model.load_state_dict(data["state_dict"])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(data["optimizers"][name])
            generator.set_state(data["generator_state"].cpu())
            history.clear(); history.update(data["history"])
            self.records, self.best = data["records"], data["best"]
            self.last_step = int(data["step"])
            restore_rng(data["rng"])
            if self.verbose:
                print(f"[baseline] Resuming {self.directory.name} after update {self.last_step}/{steps}.", flush=True)
        else:
            self._save(model, optimizers, generator, history, 0)
        model.train()
        return self.last_step + 1

    def _save(self, model, optimizers, generator, history, step):
        atomic_torch(self.resume_path, {
            "signature": self.signature, "identity": self.identity, "steps": self.steps, "step": step,
            "state_dict": model.state_dict(), "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "generator_state": generator.get_state(), "rng": rng_state(), "history": history,
            "records": self.records, "best": self.best,
        })

    def after_step(self, model, optimizers, generator, history, step):
        self.last_step = step
        if step not in self.schedule:
            return
        with scoring_mode(model):
            scores = self.score(model)
        if set(scores) != set(LABELS[self.identity["kind"]]) or not all(np.isfinite(float(v)) for v in scores.values()):
            raise ValueError(f"Invalid validation scores at update {step}: {scores}")
        for label, value in scores.items():
            value = float(value)
            self.records.append({"method": label, "step": int(step), "validation_sliced_w2": value})
            if label not in self.best or value < self.best[label]["score"]:
                self.best[label] = {"score": value, "step": int(step),
                                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if self.verbose:
                print(f"[validation] {label}: update {step}/{self.steps}, SW2={value:.6f}, best update={self.best[label]['step']}", flush=True)
        self._save(model, optimizers, generator, history, step)
        atomic_json(self.directory / "validation_history.json", self.records)

    def finish(self, model, history):
        if self.last_step != self.steps or set(self.best) != set(LABELS[self.identity["kind"]]):
            raise RuntimeError("Baseline training or validation selection is incomplete.")
        files = {}
        for label, best in self.best.items():
            filename = STEMS[label] + "_selected.pt"
            atomic_torch(self.directory / filename, {
                "signature": self.signature, "identity": self.identity, "method": label,
                "state_dict": best["state_dict"], "selected_step": best["step"],
                "validation_sliced_w2": best["score"], "history": self.records,
            })
            files[filename] = sha256(self.directory / filename)
        atomic_torch(self.directory / "final.pt", {
            "signature": self.signature, "identity": self.identity,
            "state_dict": model.state_dict(), "step": self.steps, "history": history,
        })
        files["final.pt"] = sha256(self.directory / "final.pt")
        atomic_json(self.directory / "training_complete.json", {
            "status": "completed", "signature": self.signature, "identity": self.identity,
            "steps": self.steps, "validation_checkpoints": len(self.schedule), "files": files,
            "selections": {label: {k: best[k] for k in ("score", "step")} for label, best in self.best.items()},
        })
