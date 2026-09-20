from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class ConditionalPairSplit:
    x0: torch.Tensor
    x1: torch.Tensor
    context: torch.Tensor
    condition_id: torch.Tensor

    def __post_init__(self) -> None:
        n = self.x0.shape[0]
        if self.x0.ndim != 2 or self.x1.shape != self.x0.shape:
            raise ValueError("x0 and x1 must have the same shape [n,dim].")
        if self.context.shape[0] != n or self.condition_id.shape != (n,):
            raise ValueError("Context and condition_id must align with x0.")
        if (self.x0 < 0).any() or (self.x1 < 0).any():
            raise ValueError("Counts must be nonnegative.")
        self.x0 = self.x0.long()
        self.x1 = self.x1.long()
        self.context = self.context.float()
        self.condition_id = self.condition_id.long()

    @property
    def dim(self) -> int:
        return int(self.x0.shape[1])

    @property
    def n(self) -> int:
        return int(self.x0.shape[0])

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = torch.randint(
            self.n,
            (int(batch_size),),
            generator=generator,
            device=self.x0.device,
        )
        return (
            self.x0[index].to(device),
            self.x1[index].to(device),
            self.context[index].to(device),
            self.condition_id[index].to(device),
        )


@dataclass
class ConditionalPairBundle:
    train: ConditionalPairSplit
    val: ConditionalPairSplit
    test: ConditionalPairSplit
    metadata: Dict[str, object]

    @property
    def dim(self) -> int:
        return self.train.dim


def save_pair_bundle(path: Path, bundle: ConditionalPairBundle) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for name, split in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        arrays[f"x0_{name}"] = split.x0.cpu().numpy()
        arrays[f"x1_{name}"] = split.x1.cpu().numpy()
        arrays[f"context_{name}"] = split.context.cpu().numpy()
        arrays[f"condition_{name}"] = split.condition_id.cpu().numpy()
    arrays["metadata_json"] = np.asarray(json.dumps(bundle.metadata))
    np.savez_compressed(path, **arrays)


def load_pair_bundle(path: Path) -> ConditionalPairBundle:
    payload = np.load(Path(path), allow_pickle=False)

    def split(name: str) -> ConditionalPairSplit:
        return ConditionalPairSplit(
            torch.from_numpy(payload[f"x0_{name}"]),
            torch.from_numpy(payload[f"x1_{name}"]),
            torch.from_numpy(payload[f"context_{name}"]),
            torch.from_numpy(payload[f"condition_{name}"]),
        )

    metadata = json.loads(str(payload["metadata_json"].item()))
    return ConditionalPairBundle(split("train"), split("val"), split("test"), metadata)


def _split_indices(n: int, seed: int, fractions: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain three values summing to one.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def _one_hot(index: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((index.shape[0], int(size)), dtype=np.float32)
    result[np.arange(index.shape[0]), index] = 1.0
    return result


def make_scrna_transport_surrogate(
    *,
    dim: int = 48,
    n_cell_lines: int = 3,
    n_drugs: int = 4,
    cells_per_condition: int = 320,
    latent_rank: int = 5,
    seed: int = 42,
) -> ConditionalPairBundle:
    """Structured Gamma-Poisson control-to-treatment surrogate for smoke tests."""
    rng = np.random.default_rng(seed)
    dim = int(dim)
    loadings = rng.gamma(1.5, 0.5, size=(int(latent_rank), dim))
    base = rng.gamma(1.5, 0.7, size=(int(n_cell_lines), dim)) + 0.05
    drug_effect = rng.normal(0.0, 0.45, size=(int(n_drugs), dim))
    masks = rng.random((int(n_drugs), dim)) < 0.22
    drug_effect *= masks

    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    condition_names: List[str] = []
    condition = 0
    for cell_line in range(int(n_cell_lines)):
        for drug in range(int(n_drugs)):
            dose = 0.35 + 0.65 * (drug + 1) / int(n_drugs)
            latent_control = rng.gamma(1.5, 1.0, size=(cells_per_condition, latent_rank))
            latent_target = rng.gamma(1.5, 1.0, size=(cells_per_condition, latent_rank))
            control_rate = base[cell_line] + latent_control @ loadings / latent_rank
            log_fold = dose * drug_effect[drug]
            target_rate = (base[cell_line] + latent_target @ loadings / latent_rank) * np.exp(log_fold)
            target_rate += 0.08 * np.maximum(drug_effect[drug], 0.0)
            x0 = rng.poisson(np.clip(control_rate, 1e-4, None)).astype(np.int64)
            x1 = rng.poisson(np.clip(target_rate, 1e-4, None)).astype(np.int64)
            cell = np.full(cells_per_condition, cell_line, dtype=np.int64)
            drug_id = np.full(cells_per_condition, drug, dtype=np.int64)
            context = np.concatenate(
                [
                    _one_hot(cell, n_cell_lines),
                    _one_hot(drug_id, n_drugs),
                    np.full((cells_per_condition, 1), np.log1p(dose), dtype=np.float32),
                ],
                axis=1,
            )
            condition_names.append(f"cell{cell_line}:drug{drug}:dose{dose:.3f}")
            for i in range(cells_per_condition):
                rows.append((x0[i], x1[i], context[i], condition))
            condition += 1

    rng.shuffle(rows)
    x0 = np.stack([r[0] for r in rows])
    x1 = np.stack([r[1] for r in rows])
    context = np.stack([r[2] for r in rows])
    condition_id = np.asarray([r[3] for r in rows], dtype=np.int64)

    split_rows: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for c in range(condition):
        idx = np.flatnonzero(condition_id == c)
        train, val, test = _split_indices(len(idx), seed + c, (0.7, 0.15, 0.15))
        split_rows["train"].extend(idx[train].tolist())
        split_rows["val"].extend(idx[val].tolist())
        split_rows["test"].extend(idx[test].tolist())

    def build(name: str) -> ConditionalPairSplit:
        idx = np.asarray(split_rows[name], dtype=np.int64)
        return ConditionalPairSplit(
            torch.from_numpy(x0[idx]),
            torch.from_numpy(x1[idx]),
            torch.from_numpy(context[idx]),
            torch.from_numpy(condition_id[idx]),
        )

    metadata = {
        "dataset": "synthetic_scrna_transport_smoke",
        "dim": dim,
        "context_dim": int(context.shape[1]),
        "condition_names": condition_names,
        "n_cell_lines": n_cell_lines,
        "n_drugs": n_drugs,
        "smoke_only": True,
    }
    return ConditionalPairBundle(build("train"), build("val"), build("test"), metadata)


def make_neural_forecasting_surrogate(
    *,
    dim: int = 32,
    n_bins: int = 4800,
    history_bins: int = 10,
    latent_dim: int = 6,
    seed: int = 42,
) -> ConditionalPairBundle:
    """Autoregressive latent Poisson population used only for integration tests."""
    rng = np.random.default_rng(seed)
    transition = 0.82 * np.eye(latent_dim) + rng.normal(0.0, 0.04, (latent_dim, latent_dim))
    spectral = max(abs(np.linalg.eigvals(transition)))
    transition *= 0.94 / max(spectral, 0.94)
    loading = rng.normal(0.0, 0.28, (latent_dim, dim))
    bias = rng.normal(-1.45, 0.25, dim)
    latent = np.zeros((n_bins, latent_dim), dtype=np.float64)
    for t in range(1, n_bins):
        latent[t] = transition @ latent[t - 1] + rng.normal(0.0, 0.28, latent_dim)
    rate = np.exp(np.clip(latent @ loading + bias, -5.0, 2.5))
    counts = rng.poisson(rate).astype(np.int64)
    return build_forecasting_bundle_from_counts(
        counts.T,
        history_bins=history_bins,
        train_end=int(0.7 * n_bins),
        val_end=int(0.85 * n_bins),
        metadata={"dataset": "synthetic_neural_forecasting_smoke", "smoke_only": True},
    )


def _forecast_windows(
    counts_time_major: np.ndarray,
    start: int,
    end: int,
    history_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_target = max(int(start) + int(history_bins), int(history_bins))
    targets = np.arange(first_target, int(end), dtype=np.int64)
    if targets.size == 0:
        raise ValueError("Split is too short for the requested history length.")
    histories = np.stack(
        [counts_time_major[t - history_bins : t] for t in targets], axis=0
    )
    current = counts_time_major[targets - 1]
    future = counts_time_major[targets]
    return current, future, histories


def build_forecasting_bundle_from_counts(
    counts_units_by_time: np.ndarray,
    *,
    history_bins: int,
    train_end: int,
    val_end: int,
    metadata: Optional[Dict[str, object]] = None,
    max_units: Optional[int] = None,
) -> ConditionalPairBundle:
    counts = np.asarray(counts_units_by_time)
    if counts.ndim != 2:
        raise ValueError("counts must have shape [units,time].")
    counts = counts.astype(np.int64, copy=False)
    if max_units is not None and counts.shape[0] > int(max_units):
        train_var = counts[:, : int(train_end)].var(axis=1)
        keep = np.argsort(train_var)[-int(max_units) :]
        keep.sort()
        counts = counts[keep]
    time_major = counts.T
    boundaries = {
        "train": (history_bins, int(train_end)),
        "val": (int(train_end), int(val_end)),
        "test": (int(val_end), time_major.shape[0]),
    }

    def build(name: str) -> ConditionalPairSplit:
        current, future, history = _forecast_windows(
            time_major, boundaries[name][0], boundaries[name][1], int(history_bins)
        )
        return ConditionalPairSplit(
            torch.from_numpy(current),
            torch.from_numpy(future),
            torch.from_numpy(history),
            torch.zeros(current.shape[0], dtype=torch.long),
        )

    info = dict(metadata or {})
    info.update(
        {
            "dim": int(counts.shape[0]),
            "history_bins": int(history_bins),
            "train_end": int(train_end),
            "val_end": int(val_end),
            "num_bins": int(time_major.shape[0]),
        }
    )
    return ConditionalPairBundle(build("train"), build("val"), build("test"), info)


def load_spikeprophecy_session(
    root: Path,
    session_index: int,
    *,
    history_bins: Optional[int] = None,
    max_units: Optional[int] = None,
) -> ConditionalPairBundle:
    root = Path(root)
    meta = json.loads((root / "metadata.json").read_text())
    session_meta = meta["sessions"][int(session_index)]
    counts = np.load(root / f"session_{int(session_index):03d}.npy")
    split = session_meta["split_boundaries"]
    history = int(meta.get("history_bins", 10) if history_bins is None else history_bins)
    return build_forecasting_bundle_from_counts(
        counts,
        history_bins=history,
        train_end=int(split["train_end"]),
        val_end=int(split["val_end"]),
        max_units=max_units,
        metadata={
            "dataset": "spikeprophecy-steinmetz",
            "session_index": int(session_index),
            "session_metadata": session_meta,
            "bin_width_ms": int(meta.get("bin_width_ms", 50)),
        },
    )


def _parse_dose(value: str) -> float:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and parsed:
            item = parsed[0]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return float(item[1])
    except (SyntaxError, ValueError, TypeError):
        pass
    return 0.0


def prepare_tahoe_panel_from_huggingface(
    *,
    panel_path: Path,
    output_path: Path,
    max_cells_per_condition: int = 5000,
    n_hvg: int = 2000,
    pilot_cells: int = 50000,
    seed: int = 42,
) -> ConditionalPairBundle:
    """Stream a reproducible Tahoe-100M panel into the package NPZ contract.

    The panel JSON must contain ``conditions`` with ``cell_line_id`` and ``drug``;
    optional ``samples`` restricts a condition to explicit replicate IDs.  DMSO
    controls are matched by cell line and plate as recommended by the dataset card.
    This function requires the optional ``datasets`` package and network access.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as error:  # pragma: no cover - optional formal dependency
        raise RuntimeError(
            "Install application dependencies with requirements-applications.txt."
        ) from error

    panel = json.loads(Path(panel_path).read_text())
    conditions = list(panel["conditions"])
    if not conditions:
        raise ValueError("Panel contains no conditions.")
    selected = {(str(c["cell_line_id"]), str(c["drug"])) for c in conditions}
    allowed_samples = {
        (str(c["cell_line_id"]), str(c["drug"])): set(map(str, c.get("samples", [])))
        for c in conditions
    }
    sample_meta = load_dataset(
        "tahoebio/Tahoe-100M", "sample_metadata", split="train"
    )
    sample_to_dose = {
        str(row["sample"]): _parse_dose(str(row.get("drugname_drugconc", "")))
        for row in sample_meta
    }

    stream = load_dataset(
        "tahoebio/Tahoe-100M",
        "expression_data",
        streaming=True,
        split="train",
    )
    rng = np.random.default_rng(seed)
    pilot: List[Tuple[List[int], List[float]]] = []
    reservoirs: Dict[Tuple[str, str, str], List[dict]] = {}
    seen: Dict[Tuple[str, str, str], int] = {}

    for row in stream:
        cell_line = str(row["cell_line_id"])
        drug = str(row["drug"])
        if drug == "DMSO_TF":
            relevant = any(key[0] == cell_line for key in selected)
        else:
            relevant = (cell_line, drug) in selected
        if not relevant:
            continue
        sample = str(row["sample"])
        plate = str(row["plate"])
        if drug != "DMSO_TF":
            permitted = allowed_samples.get((cell_line, drug), set())
            if permitted and sample not in permitted:
                continue
        key = (cell_line, drug, plate)
        seen[key] = seen.get(key, 0) + 1
        bucket = reservoirs.setdefault(key, [])
        record = {
            "genes": list(map(int, row["genes"][1:])),
            "expressions": list(map(float, row["expressions"][1:])),
            "sample": sample,
            "plate": plate,
            "dose": float(sample_to_dose.get(sample, 0.0)),
        }
        if len(pilot) < int(pilot_cells):
            pilot.append((record["genes"], record["expressions"]))
        if len(bucket) < int(max_cells_per_condition):
            bucket.append(record)
        else:
            j = int(rng.integers(seen[key]))
            if j < int(max_cells_per_condition):
                bucket[j] = record
        ready = True
        for cell, treated_drug in selected:
            treated_count = sum(
                len(reservoirs.get((cell, treated_drug, p), []))
                for p in {k[2] for k in reservoirs if k[0] == cell}
            )
            control_count = sum(
                len(reservoirs.get((cell, "DMSO_TF", p), []))
                for p in {k[2] for k in reservoirs if k[0] == cell}
            )
            ready &= treated_count >= max_cells_per_condition and control_count >= max_cells_per_condition
        if ready and len(pilot) >= int(pilot_cells):
            break

    if not pilot:
        raise RuntimeError("No Tahoe cells matched the requested panel.")
    gene_sum: Dict[int, float] = {}
    gene_sq: Dict[int, float] = {}
    for genes, values in pilot:
        for gene, value in zip(genes, values):
            gene_sum[gene] = gene_sum.get(gene, 0.0) + value
            gene_sq[gene] = gene_sq.get(gene, 0.0) + value * value
    n_pilot = float(len(pilot))
    dispersion = []
    for gene, total in gene_sum.items():
        mean = total / n_pilot
        variance = max(gene_sq[gene] / n_pilot - mean * mean, 0.0)
        dispersion.append((variance / (mean + 1e-3), gene))
    token_ids = [gene for _, gene in sorted(dispersion, reverse=True)[: int(n_hvg)]]
    token_to_col = {token: i for i, token in enumerate(token_ids)}

    def dense(record: dict) -> np.ndarray:
        output = np.zeros(len(token_ids), dtype=np.int64)
        for gene, value in zip(record["genes"], record["expressions"]):
            col = token_to_col.get(int(gene))
            if col is not None:
                output[col] = int(round(value))
        return output

    cell_lines = sorted({key[0] for key in selected})
    drugs = sorted({key[1] for key in selected})
    cell_to_idx = {name: i for i, name in enumerate(cell_lines)}
    drug_to_idx = {name: i for i, name in enumerate(drugs)}
    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]] = []
    condition_names: List[str] = []
    condition_index = 0
    for cell_line, drug in sorted(selected):
        treated = [r for (c, d, _), rows in reservoirs.items() if c == cell_line and d == drug for r in rows]
        if not treated:
            raise RuntimeError(f"No treated cells for {(cell_line, drug)}.")
        controls_by_plate = {
            plate: rows
            for (c, d, plate), rows in reservoirs.items()
            if c == cell_line and d == "DMSO_TF"
        }
        condition_names.append(f"{cell_line}:{drug}")
        for i, target in enumerate(treated):
            controls = controls_by_plate.get(str(target["plate"]))
            if not controls:
                controls = [r for rows in controls_by_plate.values() for r in rows]
            source = controls[i % len(controls)]
            cell_vec = np.zeros(len(cell_lines), dtype=np.float32)
            drug_vec = np.zeros(len(drugs), dtype=np.float32)
            cell_vec[cell_to_idx[cell_line]] = 1.0
            drug_vec[drug_to_idx[drug]] = 1.0
            context = np.concatenate(
                [cell_vec, drug_vec, np.asarray([np.log1p(target["dose"])], dtype=np.float32)]
            )
            pairs.append((dense(source), dense(target), context, condition_index, str(target["sample"])))
        condition_index += 1

    split_rows: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for c in range(condition_index):
        idx = np.asarray([i for i, row in enumerate(pairs) if row[3] == c], dtype=np.int64)
        samples = sorted({pairs[i][4] for i in idx})
        if len(samples) >= 3:
            rng_local = np.random.default_rng(seed + c)
            rng_local.shuffle(samples)
            assignment = {}
            for j, sample in enumerate(samples):
                fraction = j / len(samples)
                assignment[sample] = "train" if fraction < 0.7 else ("val" if fraction < 0.85 else "test")
            for i in idx:
                split_rows[assignment[pairs[i][4]]].append(int(i))
        else:
            train, val, test = _split_indices(len(idx), seed + c, (0.7, 0.15, 0.15))
            for name, subset in zip(("train", "val", "test"), (train, val, test)):
                split_rows[name].extend(idx[subset].tolist())

    def build(name: str) -> ConditionalPairSplit:
        idx = split_rows[name]
        return ConditionalPairSplit(
            torch.from_numpy(np.stack([pairs[i][0] for i in idx])),
            torch.from_numpy(np.stack([pairs[i][1] for i in idx])),
            torch.from_numpy(np.stack([pairs[i][2] for i in idx])),
            torch.tensor([pairs[i][3] for i in idx], dtype=torch.long),
        )

    bundle = ConditionalPairBundle(
        build("train"),
        build("val"),
        build("test"),
        {
            "dataset": "Tahoe-100M",
            "panel": panel,
            "gene_token_ids": token_ids,
            "condition_names": condition_names,
            "cell_lines": cell_lines,
            "drugs": drugs,
            "context_dim": len(cell_lines) + len(drugs) + 1,
            "split_protocol": "replicate-aware when >=3 sample replicates; otherwise stratified cells",
        },
    )
    save_pair_bundle(Path(output_path), bundle)
    return bundle
