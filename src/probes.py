"""
Per-layer linear probes: a second, independent line of evidence for the
refusal representation.

Two probes always run together:
- the real probe: predicts the prompt label (harmful=1, harmless=0) from the
  residual activation at each layer.
- the shuffled-label control: same probe trained on labels permuted with a
  separate seed. It MUST sit at chance. If it doesn't, the train/test split
  is leaking and every probe number above is suspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ProbeResult:
    layer: int
    train_acc: float
    test_acc: float
    n_train: int
    n_test: int
    seed: int


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").float().numpy()


def train_probe(
    acts: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    test_size: float = 0.25,
    C: float = 1.0,
) -> ProbeResult:
    """Train a single logistic-regression probe on [n, d_model] activations.

    Parameters
    ----------
    acts: [n, d_model] — residual activations at one fixed layer.
    labels: [n] — 0/1.
    seed: drives the train/test split AND the LR solver. Same seed → same
        numbers across runs (a hard requirement for reporting).
    test_size: fraction held out.
    C: inverse regularization. The default 1.0 is the sklearn default; tune
        only with a held-out set, never on the test split.

    Returns
    -------
    ProbeResult with train and test accuracy.
    """
    X = _to_numpy(acts)
    y = _to_numpy(labels).astype(int)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"n mismatch: acts {X.shape[0]} vs labels {y.shape[0]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X_tr, y_tr)

    return ProbeResult(
        layer=-1,  # caller sets the real layer index
        train_acc=float(clf.score(X_tr, y_tr)),
        test_acc=float(clf.score(X_te, y_te)),
        n_train=int(X_tr.shape[0]),
        n_test=int(X_te.shape[0]),
        seed=seed,
    )


def probe_layer_sweep(
    acts_all_layers: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    test_size: float = 0.25,
    C: float = 1.0,
) -> list[ProbeResult]:
    """Train a probe at every layer.

    Parameters
    ----------
    acts_all_layers: [n, n_layers, d_model] — output of
        activations.cache_resid_all_layers.
    """
    if acts_all_layers.ndim != 3:
        raise ValueError(
            f"expected [n, n_layers, d_model], got shape {tuple(acts_all_layers.shape)}"
        )

    n_layers = acts_all_layers.shape[1]
    results: list[ProbeResult] = []
    for L in range(n_layers):
        r = train_probe(
            acts_all_layers[:, L, :], labels, seed=seed,
            test_size=test_size, C=C,
        )
        results.append(ProbeResult(
            layer=L,
            train_acc=r.train_acc,
            test_acc=r.test_acc,
            n_train=r.n_train,
            n_test=r.n_test,
            seed=r.seed,
        ))
    return results


def shuffled_control_sweep(
    acts_all_layers: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    shuffle_seed: int,
    test_size: float = 0.25,
    C: float = 1.0,
) -> list[ProbeResult]:
    """Layer sweep with labels permuted under shuffle_seed.

    Mandatory control (Step 4). Permuting under a SEPARATE seed from the
    split keeps the comparison apples-to-apples: same split, scrambled
    supervision. Test accuracy must sit at chance (~0.5 for balanced labels)
    at every layer. If it doesn't, suspect:

    - imbalanced labels (chance ≠ 0.5)
    - duplicate/near-duplicate prompts across train/test
    - bugs in cache_resid_all_layers (e.g. constant activations)
    """
    g = np.random.default_rng(shuffle_seed)
    shuffled = _to_numpy(labels).astype(int).copy()
    g.shuffle(shuffled)
    shuffled_t = torch.from_numpy(shuffled).long()
    return probe_layer_sweep(
        acts_all_layers, shuffled_t, seed=seed, test_size=test_size, C=C,
    )
