"""Bridge mech-security matched set → geometry-of-refusal pipeline inputs.

SUPERSEDED by Part 2 (diff-of-means + bypass-gap-layer-selection in our
harness). This script was written when "Step 4b" meant "train d_rdo at L14
init"; the planner's Part-1 terminology audit + Part-2 redesign moved L14
off the table. Keep for reference; do not rerun. See PROJECT_STATE.md for
the current 2×2 and Step 4 plan.

Original purpose (Step 4b prep, pre-revision):
  1. Write data/matched_splits/{harmful,harmless}_{train,val}.json under the
     geometry-of-refusal repo using the same 30/10 split as Phase 2 step 3e
     (split_seed=1). Val files are 10-prompt held-out (also needed downstream
     for any --filter_data run; RDO --train_direction alone doesn't read val,
     but write them so downstream variants don't crash).
  2. Cache-hit extract d̂_matched at L14 (matched_v2 tag), then save the DIM
     init files RDO expects:
        artifacts/dim_results/Qwen2.5-3B-Instruct/direction.pt
        artifacts/dim_results/Qwen2.5-3B-Instruct/direction_metadata.json
        artifacts/dim_results/Qwen2.5-3B-Instruct/generate_directions/mean_diffs.pt
     direction.pt = d̂_matched * natural_scale, so .norm() = our natural alpha.
     mean_diffs.pt is a dummy zero tensor; rdo.py loads it but never reads it.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments._runner import extract_d_hat, get_model  # noqa: E402
from src.model import format_prompt_for_bundle  # noqa: E402

GOR_ROOT = Path.home() / "safe_ai" / "geometry-of-refusal"
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
MODEL_ID = MODEL_PATH.split("/")[-1]  # "Qwen2.5-3B-Instruct"
EXTRACT_LAYER = 14
SPLIT_SEED = 1
N_TEST = 10


def split_matched(jsonl_path: Path, seed: int = 1, n_test: int = 10):
    """Byte-faithful to experiments/phase2_step3e_matched_set_sweep._split."""
    recs = [json.loads(l) for l in jsonl_path.open()]
    harmful = [r["text"] for r in recs if r["label"] == "harmful"]
    harmless = [r["text"] for r in recs if r["label"] == "harmless"]
    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    return (
        harmful[n_test:], harmful[:n_test],
        harmless[n_test:], harmless[:n_test],
    )


def write_split_jsons(splits_dir: Path, h_train, h_test, l_train, l_test):
    splits_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("harmful_train.json", h_train),
        ("harmless_train.json", l_train),
        ("harmful_val.json", h_test),   # use the same 10-prompt held-out as our test
        ("harmless_val.json", l_test),
    ]
    for name, prompts in pairs:
        out = [{"instruction": p, "source": "matched_v4"} for p in prompts]
        (splits_dir / name).write_text(json.dumps(out, indent=2))
        print(f"wrote {splits_dir / name}: {len(out)} instructions")


def main() -> int:
    matched_path = REPO_ROOT / "data" / "code_contrastive_matched.jsonl"
    assert matched_path.exists(), matched_path

    h_train, h_test, l_train, l_test = split_matched(
        matched_path, seed=SPLIT_SEED, n_test=N_TEST,
    )
    print(f"split: harmful train={len(h_train)} test={len(h_test)}, "
          f"harmless train={len(l_train)} test={len(l_test)}")

    splits_dir = GOR_ROOT / "data" / "matched_splits"
    write_split_jsons(splits_dir, h_train, h_test, l_train, l_test)

    print("loading Qwen2.5-3B-Instruct (cache hit expected for activations)...")
    bundle = get_model(MODEL_PATH)
    fmt = lambda msg: format_prompt_for_bundle(bundle, msg)  # noqa: E731
    d_hat, _H, _L, meta = extract_d_hat(
        bundle, h_train, l_train,
        layer=EXTRACT_LAYER, format_fn=fmt,
        extra_tag="matched_v2", harmless_key_suffix="harmless_train",
    )
    natural_scale = meta["natural_scale"]
    print(f"d̂_matched at L{EXTRACT_LAYER}: natural_scale={natural_scale:.4f}")
    print(f"d_hat .norm() (should be ~1): {d_hat.norm().item():.4f}")
    print(f"cache keys: H={meta['key_h']} L={meta['key_l']}")

    dim_dir = GOR_ROOT / "artifacts" / "dim_results" / MODEL_ID
    dim_dir.mkdir(parents=True, exist_ok=True)
    (dim_dir / "generate_directions").mkdir(exist_ok=True)

    direction = (d_hat.detach().cpu().to(torch.float32) * float(natural_scale))
    torch.save(direction, dim_dir / "direction.pt")
    print(f"wrote {dim_dir / 'direction.pt'}: shape={tuple(direction.shape)} "
          f"norm={direction.norm().item():.4f}")

    metadata = {
        "layer": EXTRACT_LAYER,
        "pos": -1,
        "source": "mech-security d̂_matched (matched_v4 contrast, split_seed=1)",
        "natural_scale": natural_scale,
        "model": MODEL_PATH,
    }
    (dim_dir / "direction_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {dim_dir / 'direction_metadata.json'}")

    n_layers = bundle.model.cfg.n_layers
    d_model = bundle.model.cfg.d_model
    dummy_mean_diffs = torch.zeros((n_layers, 1, d_model), dtype=torch.float32)
    torch.save(dummy_mean_diffs, dim_dir / "generate_directions" / "mean_diffs.pt")
    print(f"wrote dummy mean_diffs.pt: shape={tuple(dummy_mean_diffs.shape)}")

    print("\n=== bridge complete ===")
    print(f"data:       {splits_dir}")
    print(f"DIM init:   {dim_dir}")
    print("next:       cd ~/safe_ai/geometry-of-refusal && "
          f"python rdo.py --train_direction --model {MODEL_PATH} --splits matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
