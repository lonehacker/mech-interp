"""Phase 2 Part 2 PIECE 2B — backwards-decomposition L0 cosine.

The empirical headline (AUC picks the wrong layer; bypass-gap picks the
causal one) is in. The mechanistic capstone is one cosine:
cos(d̂_L22_pos-1, d̂_L0_matched). If near zero, the causal direction at L22
is nearly perpendicular to the L0-embedding separability subspace — which
is *why* AUC at L0 saturates yet cannot point at the causal direction.

Extracts d̂ at L0 on the same matched contrast + same seed=1 / n_test=10
split as the original Part 2 run (so the cosine is comparable to the saved
d̂_L22 vectors in artifacts/runs/phase2_part2/20260530-134914/result.json).
Computes cosines against d̂_L22_pos-1, d̂_L22_pos-4, and the L14 references
already on disk.

Saves a small JSON summary under artifacts/runs/phase2_part2_l0_cosine/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import dotenv
import torch

dotenv.load_dotenv()

from experiments._runner import get_logger, get_model, new_run_dir, write_json  # noqa: E402
from mech_security.activations import cache_resid  # noqa: E402
from mech_security.directions import diff_of_means, unit  # noqa: E402
from mech_security.model import format_prompt_for_bundle  # noqa: E402

log = get_logger("phase2_part2_l0_cosine")


def split_matched(jsonl_path: Path, seed: int = 1, n_test: int = 10):
    """Byte-identical to the original Part 2 run's split."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    rng = random.Random(seed)
    rng.shuffle(harmful); rng.shuffle(harmless)
    return harmful[n_test:], harmless[n_test:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/code_contrastive_matched.jsonl")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    run_dir = new_run_dir("phase2_part2_l0_cosine")
    log.info("run_dir: %s", run_dir)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path
    h_train, l_train = split_matched(data_path)

    log.info("loading model: %s", args.model)
    bundle = get_model(args.model)
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)  # noqa: E731

    log.info("extracting d̂_L0_matched (last-token, on same matched train set)...")
    H = cache_resid(bundle, h_train, layer=0, position=-1, format_fn=fmt, show_progress=False)
    L = cache_resid(bundle, l_train, layer=0, position=-1, format_fn=fmt, show_progress=False)
    d_L0 = unit(diff_of_means(H, L))
    log.info("d̂_L0_matched: shape=%s norm=%.4f", tuple(d_L0.shape), d_L0.norm().item())

    # Load saved d̂_L22 vectors from the original Part 2 run
    part2 = json.load(open("artifacts/runs/phase2_part2/20260530-134914/result.json"))
    d_L22_p1 = torch.tensor(next(c["d_hat_cpu"] for c in part2["cells"] if c["name"] == "L22_pos-1"))
    d_L22_p4 = torch.tensor(next(c["d_hat_cpu"] for c in part2["cells"] if c["name"] == "L22_pos-4"))

    # Recompute d̂_L14_matched + d̂_L14_old from cached on-disk activations
    H_m14 = torch.load("artifacts/cache/c101587891347bd3.pt")
    L_m14 = torch.load("artifacts/cache/7048dd30241c42cc.pt")
    d_L14_matched = unit(diff_of_means(H_m14, L_m14))
    H_c14 = torch.load("artifacts/cache/031758e602174196.pt")
    L_c14 = torch.load("artifacts/cache/798173ea952ecf0d.pt")
    d_L14_old = unit(diff_of_means(H_c14, L_c14))

    cosines = {
        "cos(d_L22_pos-1, d_L0_matched)": float((d_L22_p1 * d_L0).sum().item()),
        "cos(d_L22_pos-4, d_L0_matched)": float((d_L22_p4 * d_L0).sum().item()),
        "cos(d_L14_matched, d_L0_matched)": float((d_L14_matched * d_L0).sum().item()),
        "cos(d_L14_old, d_L0_matched)": float((d_L14_old * d_L0).sum().item()),
        "cos(d_L22_pos-1, d_L22_pos-4)": float((d_L22_p1 * d_L22_p4).sum().item()),
        "cos(d_L22_pos-1, d_L14_matched)": float((d_L22_p1 * d_L14_matched).sum().item()),
        "cos(d_L22_pos-4, d_L14_matched)": float((d_L22_p4 * d_L14_matched).sum().item()),
        "cos(d_L22_pos-1, d_L14_old)": float((d_L22_p1 * d_L14_old).sum().item()),
        "cos(d_L22_pos-4, d_L14_old)": float((d_L22_p4 * d_L14_old).sum().item()),
        "cos(d_L14_matched, d_L14_old)": float((d_L14_matched * d_L14_old).sum().item()),
    }

    print("\n=== Phase 2 Part 2 PIECE 2B — backwards-decomposition cosines ===")
    print(f"d̂_L0_matched: norm={d_L0.norm().item():.4f}")
    print()
    print(f"{'cosine':<40} {'value':>8}")
    for name, val in cosines.items():
        print(f"{name:<40} {val:>+8.4f}")

    write_json(run_dir / "result.json", {
        "model": args.model,
        "data": str(data_path),
        "cosines": cosines,
        "d_L0_matched_cpu": d_L0.tolist(),
    })
    log.info("wrote %s", run_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
