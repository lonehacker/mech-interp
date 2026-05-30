"""
Shared run-context for experiment scripts.

Each experiment script in this directory is a small, idempotent main() that:
- gets a ModelBundle from `get_model` (memoized per-process)
- writes its outputs to artifacts/runs/<step>/<timestamp>/
- writes a human-readable summary to results/<step>.md

This file owns the cross-cutting concerns (paths, run-dir creation, JSON
serialization) so the experiment scripts stay focused on their step.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

# Acknowledge the MPS-may-be-silently-wrong warning from transformer_lens.
# Justified for Phase 0/1 on gemma-2-2b-it where we're looking for clear
# signals (separation, refusal-rate deltas), not publication-grade precision.
# Verify any conclusion against a CPU spot-check before committing to it.
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

import torch

# Late-imported to keep import overhead off scripts that only want paths.
_BUNDLE: object | None = None

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"

ARTIFACTS_CACHE = ARTIFACTS / "cache"
ARTIFACTS_FIGURES = ARTIFACTS / "figures"
ARTIFACTS_RUNS = ARTIFACTS / "runs"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def get_model(name: str = "gemma-2-2b-it"):
    """Memoized model load. Subsequent calls in the same process reuse the
    bundle; cross-process callers pay the ~30s MPS load each time.
    """
    global _BUNDLE
    if _BUNDLE is not None and _BUNDLE.name.endswith(name):  # type: ignore[union-attr]
        return _BUNDLE
    from src.model import load_model
    _BUNDLE = load_model(name)
    return _BUNDLE


def new_run_dir(step: str) -> Path:
    """Create artifacts/runs/<step>/<timestamp>/ and return the path."""
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = ARTIFACTS_RUNS / step / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> None:
    """JSON-dump with dataclass + tensor coercion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, torch.Tensor):
        return o.detach().to("cpu").float().tolist()
    if isinstance(o, Path):
        return str(o)
    # numpy scalar types (bool_, int64, float32, ...) → native Python
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"not JSON-serializable: {type(o)}")


def content_hash(items: list[str], extra: str = "") -> str:
    """Stable hash over a list of strings + extra config. Used as a cache key
    for activation tensors."""
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    if extra:
        h.update(extra.encode("utf-8"))
    return h.hexdigest()[:16]


def cached_activations(
    key: str,
    compute_fn,
) -> torch.Tensor:
    """Disk-cache wrapper around an activation computation.

    Usage:
        acts = cached_activations(
            content_hash(prompts, extra=f"L{layer}"),
            lambda: cache_resid(bundle, prompts, layer=layer),
        )

    Saves to artifacts/cache/<key>.pt. Loads it on the next call with the same
    key (same prompts + same layer + same dtype assumption).
    """
    ARTIFACTS_CACHE.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_CACHE / f"{key}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    acts = compute_fn()
    torch.save(acts, path)
    return acts


def load_jsonl_pairs(path: Path) -> tuple[list[str], list[str]]:
    """Load a JSONL with {text, label} records; return (harmful, harmless)
    lists in file order. Used by every experiment that needs the contrastive
    set, so the load is exactly once and exactly the same everywhere.
    """
    harmful: list[str] = []
    harmless: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["label"] == "harmful":
                harmful.append(rec["text"])
            elif rec["label"] == "harmless":
                harmless.append(rec["text"])
            else:
                raise ValueError(f"unknown label in {path}: {rec['label']}")
    return harmful, harmless


@dataclass(frozen=True)
class StepResult:
    """Common return shape for an experiment step."""
    step: str
    run_dir: Path
    summary: dict


T = TypeVar("T")


def generate_batch(
    bundle,
    prompts: list[str],
    *,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    strip: bool = True,
) -> list[str]:
    """Greedy-generate over a list of prompts; return list of completions.
    Defaults match the dual-judge sweep convention (T=0, 160 tokens, .strip())."""
    from src.model import generate
    out = [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=temperature)
           for p in prompts]
    return [s.strip() for s in out] if strip else out


def train_test_split(
    records: list[T],
    *,
    seed: int,
    n_test: int,
) -> tuple[list[T], list[T]]:
    """Uniform shuffle + slice. Returns (train, test).

    Byte-identical to `rng = random.Random(seed); rng.shuffle(records);
    test, train = records[:n_test], records[n_test:]` over a fresh copy."""
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[n_test:], items[:n_test]


def stratified_split(
    records: list[T],
    *,
    key_fn: Callable[[T], Any],
    seed: int,
    n_test: int,
) -> tuple[list[T], list[T]]:
    """Stratified split preserving per-stratum fractions. Returns (train, test).

    Strata are grouped by `key_fn(record)` and listed in first-occurrence
    order. Each stratum is shuffled with `random.Random(seed)`; that single
    rng is consumed in stratum-iteration order, matching the inlined
    `rng.shuffle(hb); rng.shuffle(adv)` pattern in the Phase 2 runners.

    `n_test` is split across strata by `round(n_test * frac)`, with the
    final stratum absorbing the remainder so the total equals `n_test`."""
    items = list(records)
    n_total = len(items)
    if n_total == 0:
        return [], []
    # First-occurrence order for strata
    strata_order: list[Any] = []
    buckets: dict[Any, list[T]] = {}
    for r in items:
        k = key_fn(r)
        if k not in buckets:
            buckets[k] = []
            strata_order.append(k)
        buckets[k].append(r)

    rng = random.Random(seed)
    for k in strata_order:
        rng.shuffle(buckets[k])

    # Allocate n_test across strata; last stratum absorbs remainder.
    n_test_per: dict[Any, int] = {}
    allocated = 0
    for i, k in enumerate(strata_order):
        frac = len(buckets[k]) / n_total
        if i == len(strata_order) - 1:
            n_test_per[k] = n_test - allocated
        else:
            n_test_per[k] = round(n_test * frac)
            allocated += n_test_per[k]

    train: list[T] = []
    test: list[T] = []
    for k in strata_order:
        n_te = n_test_per[k]
        test.extend(buckets[k][:n_te])
        train.extend(buckets[k][n_te:])
    return train, test


def extract_d_hat(
    bundle,
    harmful: list[str],
    harmless: list[str],
    *,
    layer: int,
    format_fn: Callable[[str], str] | None = None,
    extra_tag: str,
    harmful_key_suffix: str = "harmful_train",
    harmless_key_suffix: str = "harmless",
):
    """Standard d̂ extraction pipeline. Returns (d_hat, H, L, meta) where
    `meta = {"natural_scale": float, "midpoint": float, "extra": str,
    "key_h": str, "key_l": str}`.

    The cache-key `extra` is built as
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{layer}|"
        f"resid_post|last_token|{extra_tag}"
    and combined with the suffix args, matching the inlined Phase 2 pattern
    byte-for-byte. `extra_tag` examples: "phase2", "matched_v2"."""
    from src.activations import cache_resid
    from src.directions import diff_of_means, project, unit

    extra = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{layer}|"
             f"resid_post|last_token|{extra_tag}")
    key_h = content_hash(harmful, extra=extra + f"|{harmful_key_suffix}")
    key_l = content_hash(harmless, extra=extra + f"|{harmless_key_suffix}")
    H = cached_activations(
        key_h,
        lambda: cache_resid(bundle, harmful, layer=layer,
                            show_progress=False, format_fn=format_fn),
    )
    L = cached_activations(
        key_l,
        lambda: cache_resid(bundle, harmless, layer=layer,
                            show_progress=False, format_fn=format_fn),
    )
    d_hat = unit(diff_of_means(H, L))
    h_mean = float(project(H, d_hat).mean())
    l_mean = float(project(L, d_hat).mean())
    meta = {
        "natural_scale": h_mean,
        "harmless_mean": l_mean,
        "midpoint": 0.5 * (h_mean + l_mean),
        "extra": extra,
        "key_h": key_h,
        "key_l": key_l,
    }
    return d_hat, H, L, meta
