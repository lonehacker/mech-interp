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
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any

# Acknowledge the MPS-may-be-silently-wrong warning from transformer_lens.
# Justified for Phase 0/1 on gemma-2-2b-it where we're looking for clear
# signals (separation, refusal-rate deltas), not publication-grade precision.
# Verify any conclusion against a CPU spot-check before committing to it.
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

import torch

# Late-imported to keep import overhead off scripts that only want paths.
_BUNDLE: "object | None" = None

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
