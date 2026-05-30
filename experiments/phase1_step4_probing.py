"""
Phase 1 — Step 4: per-layer linear probes (independent line of evidence).

The steering result (Step 3) is causal but lives in one method's frame:
ablating d_hat eliminates refusal at L13. A probe is the second leg of the
triangulation — it asks a different question (can a linear classifier read
the label off the residual stream at layer L?) with different machinery
(logistic regression on a held-out split). Two methods pointing at the same
layer band is strong; tension between them is research.

Mandatory control: shuffled-label probe. With balanced labels (n harmful =
n harmless = 150) the shuffled probe MUST sit at ~0.5 test accuracy at every
layer. If it doesn't, the split is leaking and every real-probe number above
it is suspect.

Reuses the disk-cached residual activations from Step 2 (same content_hash
key), so this script does no forward passes if Step 2 has already run.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_step4_probing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments._runner import (
    ARTIFACTS_FIGURES,
    RESULTS,
    cached_activations,
    content_hash,
    get_logger,
    get_model,
    load_jsonl_pairs,
    new_run_dir,
    write_json,
)
from mech_security.activations import cache_resid_all_layers
from mech_security.probes import probe_layer_sweep, shuffled_control_sweep

log = get_logger("phase1_step4")

# Steering band from Step 2 (LOO-CV AUC plateau within 0.02 of peak on the
# AdvBench-backed contrastive set). Hardcoded here so the figure overlay
# doesn't silently shift when Step 2 is re-tagged. If Step 2 is re-run with
# different data, update this list and the steering-injection layer below.
STEERING_BAND = list(range(7, 26))      # L7–L25 inclusive
STEERING_INJECTION_LAYER = 13           # the layer used in Step 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-layer probing.")
    ap.add_argument("--data", type=str, default="data/contrastive.jsonl")
    ap.add_argument("--seed", type=int, default=0,
                    help="Split + LR seed. Matches Step 2 convention.")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="Distinct seed for the label-shuffled control.")
    ap.add_argument("--test-size", type=float, default=0.25,
                    help="Held-out fraction (stratified).")
    ap.add_argument("--C", type=float, default=1.0,
                    help="Inverse regularization for logistic regression.")
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_step4")
    log.info("run_dir: %s | data=%s seed=%d shuffle_seed=%d test_size=%.2f",
             run_dir, args.data, args.seed, args.shuffle_seed, args.test_size)

    bundle = get_model()
    log.info("model: %s | n_layers=%d d_model=%d device=%s",
             bundle.name, bundle.n_layers, bundle.d_model, bundle.device)

    pairs_path = Path(args.data)
    if not pairs_path.is_absolute():
        pairs_path = Path(__file__).resolve().parent.parent / pairs_path
    harmful, harmless = load_jsonl_pairs(pairs_path)
    log.info("pairs: %d harmful, %d harmless from %s",
             len(harmful), len(harmless), pairs_path.name)

    # Match Step 2's cache key verbatim so we re-use its on-disk activation
    # tensor (no forward passes if Step 2 already ran).
    extra_all = (
        f"{bundle.name}|dtype={bundle.model.cfg.dtype}|all_layers|resid_post|last_token"
    )
    key_h = content_hash(harmful, extra=extra_all + "|harmful")
    key_l = content_hash(harmless, extra=extra_all + "|harmless")
    log.info("loading cached residuals across all %d layers...", bundle.n_layers)
    harmful_acts = cached_activations(
        key_h, lambda: cache_resid_all_layers(bundle, harmful, show_progress=False)
    )
    harmless_acts = cached_activations(
        key_l, lambda: cache_resid_all_layers(bundle, harmless, show_progress=False)
    )
    log.info("shapes: harmful %s, harmless %s",
             tuple(harmful_acts.shape), tuple(harmless_acts.shape))

    # Stack into [n_total, n_layers, d_model] with parallel labels (1=harmful).
    all_acts = torch.cat([harmful_acts, harmless_acts], dim=0)
    labels = torch.tensor(
        [1] * len(harmful) + [0] * len(harmless), dtype=torch.long
    )
    n_layers = all_acts.shape[1]

    log.info("training real probes across %d layers...", n_layers)
    real = probe_layer_sweep(
        all_acts, labels, seed=args.seed,
        test_size=args.test_size, C=args.C,
    )
    log.info("training shuffled-label control probes across %d layers...", n_layers)
    shuf = shuffled_control_sweep(
        all_acts, labels, seed=args.seed, shuffle_seed=args.shuffle_seed,
        test_size=args.test_size, C=args.C,
    )

    real_acc = np.array([r.test_acc for r in real])
    shuf_acc = np.array([r.test_acc for r in shuf])
    real_train_acc = np.array([r.train_acc for r in real])

    peak_layer = int(np.argmax(real_acc))
    peak_acc = float(real_acc[peak_layer])
    # Probe band: layers within 0.02 of peak (mirror Step 2's plateau definition).
    probe_band = [int(L) for L in range(n_layers) if real_acc[L] >= peak_acc - 0.02]

    # Convergence diagnostics (numbers, not verdicts — see runbook §7).
    steering_set = set(STEERING_BAND)
    probe_set = set(probe_band)
    overlap = sorted(steering_set & probe_set)
    only_steering = sorted(steering_set - probe_set)
    only_probe = sorted(probe_set - steering_set)
    probe_acc_at_inj = float(real_acc[STEERING_INJECTION_LAYER])
    shuf_acc_at_inj = float(shuf_acc[STEERING_INJECTION_LAYER])

    log.info("probe peak: L%d, test_acc=%.3f (band L%d–L%d)",
             peak_layer, peak_acc, min(probe_band), max(probe_band))
    log.info("at steering-injection layer L%d: probe test_acc=%.3f, shuffled=%.3f",
             STEERING_INJECTION_LAYER, probe_acc_at_inj, shuf_acc_at_inj)
    log.info("shuffled-control mean across layers: %.3f (must be ≈ 0.5)",
             float(shuf_acc.mean()))

    # Controls
    shuf_chance_overall = bool(abs(shuf_acc.mean() - 0.5) < 0.10)
    shuf_chance_at_peak = bool(abs(shuf_acc[peak_layer] - 0.5) < 0.15)
    real_beats_shuf_at_peak = bool(real_acc[peak_layer] - shuf_acc[peak_layer] > 0.30)

    # Figure: real probe + shuffled control + steering band + injection axvline.
    fig_path = ARTIFACTS_FIGURES / "phase1_step4_probe_by_layer.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    layers = np.arange(n_layers)
    ax.axvspan(min(STEERING_BAND), max(STEERING_BAND), alpha=0.10,
               color="tab:orange",
               label=f"Step 2 steering band (L{min(STEERING_BAND)}–L{max(STEERING_BAND)})")
    ax.axvline(STEERING_INJECTION_LAYER, color="tab:red", linestyle=":",
               linewidth=1.5, alpha=0.7,
               label=f"Step 3 injection L{STEERING_INJECTION_LAYER}")
    ax.plot(layers, real_acc, "o-", color="tab:blue", linewidth=2,
            label="real probe (test acc)")
    ax.plot(layers, real_train_acc, "o:", color="tab:blue", linewidth=1, alpha=0.4,
            label="real probe (train acc)")
    ax.plot(layers, shuf_acc, "^--", color="tab:green", alpha=0.7,
            label="shuffled-label control (test acc)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy")
    ax.set_title(
        f"Phase 1 Step 4 — per-layer logistic probe vs steering band\n"
        f"(peak L{peak_layer}, test_acc={peak_acc:.3f}; "
        f"shuffled mean={shuf_acc.mean():.3f})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log.info("figure -> %s", fig_path)

    # Persist
    record = {
        "step": "phase1_step4",
        "model": bundle.name,
        "device": bundle.device,
        "n_layers": n_layers,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "test_size": args.test_size,
        "C": args.C,
        "probe_peak_layer": peak_layer,
        "probe_peak_test_acc": peak_acc,
        "probe_band": probe_band,
        "steering_band": STEERING_BAND,
        "steering_injection_layer": STEERING_INJECTION_LAYER,
        "real_test_acc": real_acc.tolist(),
        "real_train_acc": real_train_acc.tolist(),
        "shuffled_test_acc": shuf_acc.tolist(),
        "probe_acc_at_injection_layer": probe_acc_at_inj,
        "shuffled_acc_at_injection_layer": shuf_acc_at_inj,
        "band_overlap": overlap,
        "only_steering_band": only_steering,
        "only_probe_band": only_probe,
        "controls_pass": {
            "shuffled_at_chance_overall": shuf_chance_overall,
            "shuffled_at_chance_at_peak": shuf_chance_at_peak,
            "real_beats_shuffled_at_peak": real_beats_shuf_at_peak,
        },
        "figure": str(fig_path),
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    summary_path = RESULTS / "phase1_step4.md"
    summary_path.write_text(_render_summary(record))
    log.info("summary -> %s", summary_path)

    print(
        f"\nphase1_step4 | probe peak L{peak_layer} acc={peak_acc:.3f} | "
        f"shuffled mean={shuf_acc.mean():.3f} | "
        f"band overlap with steering: {len(overlap)}/{len(STEERING_BAND)} layers"
    )
    return 0


def _render_summary(rec: dict) -> str:
    real = rec["real_test_acc"]
    train = rec["real_train_acc"]
    shuf = rec["shuffled_test_acc"]
    cp = rec["controls_pass"]
    band = rec["probe_band"]
    sband = rec["steering_band"]
    inj = rec["steering_injection_layer"]

    lines = [
        "# Phase 1 Step 4 — per-layer probing",
        "",
        "**Independent line of evidence for the refusal representation.** A linear "
        "probe at each layer answers a different question than steering: not "
        "*does intervening change behavior?* but *is the label linearly readable?* "
        "Agreement between probe and steering bands is strong; tension is research.",
        "",
        f"- Model: `{rec['model']}` on `{rec['device']}` | n_layers={rec['n_layers']}",
        f"- n harmful = {rec['n_harmful']}, n harmless = {rec['n_harmless']}",
        f"- Split: stratified, test_size={rec['test_size']}, seed={rec['seed']}",
        f"- Logistic regression: C={rec['C']}, solver=lbfgs",
        f"- Shuffled-label control: shuffle_seed={rec['shuffle_seed']}",
        "",
        "## Headline",
        "",
        f"- **Probe peak: L{rec['probe_peak_layer']}** (test_acc = {rec['probe_peak_test_acc']:.3f})",
        f"- **Probe band (within 0.02 of peak): L{min(band)}–L{max(band)}** "
        f"({len(band)} layers)",
        f"- **Steering band (from Step 2): L{min(sband)}–L{max(sband)}** "
        f"({len(sband)} layers)",
        f"- **Band overlap: {len(rec['band_overlap'])} layers** — {rec['band_overlap']}",
        f"- At steering-injection L{inj}: probe test_acc = {rec['probe_acc_at_injection_layer']:.3f}, "
        f"shuffled = {rec['shuffled_acc_at_injection_layer']:.3f}",
        "",
        "## Controls (gates, not extras)",
        "",
        "| Control | Numbers | Pass |",
        "|---|---|---|",
        f"| Shuffled at chance overall (\\|mean − 0.5\\| < 0.10) | "
        f"mean shuffled test acc = {np.mean(shuf):.3f} | "
        f"{'✅' if cp['shuffled_at_chance_overall'] else '❌'} |",
        f"| Shuffled at chance at probe peak (\\|acc − 0.5\\| < 0.15) | "
        f"shuffled at L{rec['probe_peak_layer']} = {shuf[rec['probe_peak_layer']]:.3f} | "
        f"{'✅' if cp['shuffled_at_chance_at_peak'] else '❌'} |",
        f"| Real beats shuffled at peak by ≥ 0.30 | "
        f"real = {real[rec['probe_peak_layer']]:.3f}, "
        f"shuffled = {shuf[rec['probe_peak_layer']]:.3f} | "
        f"{'✅' if cp['real_beats_shuffled_at_peak'] else '❌'} |",
        "",
        "## Per-layer table",
        "",
        "| layer | real test acc | real train acc | shuffled test acc | in steering band? |",
        "|---:|---:|---:|---:|:---:|",
    ]
    sband_set = set(sband)
    for L in range(rec["n_layers"]):
        flag = "✓" if L in sband_set else ""
        lines.append(
            f"| L{L} | {real[L]:.3f} | {train[L]:.3f} | {shuf[L]:.3f} | {flag} |"
        )
    lines.append("")
    lines.append(f"Figure: `{rec['figure']}`")
    lines.append("")
    lines.append("## (H) Interpretation — to be filled in by the human")
    lines.append("")
    lines.append(
        "The runbook §7 is explicit: if the probe is strong where steering is "
        "weak (or vice versa), that tension IS the research — do not paper "
        "over it. Read the per-layer table and answer:"
    )
    lines.append("")
    lines.append("1. Does the probe band overlap the steering band? "
                 f"({len(rec['band_overlap'])} / {len(sband)} layers do.)")
    lines.append("2. Layers in steering band but NOT probe band: "
                 f"{rec['only_steering_band']}")
    lines.append("3. Layers in probe band but NOT steering band: "
                 f"{rec['only_probe_band']}")
    lines.append("4. Is the steering-injection layer L"
                 f"{inj} a strong probe layer? "
                 f"(test_acc = {rec['probe_acc_at_injection_layer']:.3f})")
    lines.append("")
    lines.append("Write the answer here, in raw-numbers form, before claiming "
                 "convergence:")
    lines.append("")
    lines.append("> _(H) finding goes here_")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
