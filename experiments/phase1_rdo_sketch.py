"""
SUPERSEDED — parked, never run. Kept for posterity as a scope-escalation example
(homegrown RDO sketch written from prose, not the canonical Wollschläger repo).

⚠️  HOMEGROWN SKETCH — NOT WOLLSCHLÄGER'S RDO. DO NOT TREAT RESULTS FROM THIS
   SCRIPT AS AN RDO TEST OR A REPLICATION OF Wollschläger et al. ICML 2025.

This file was written from a planner's prose description of RDO, not from
the paper or the canonical Wollschläger repo (github.com/wollschlager/
geometry-of-refusal). It is missing the defining properties of RDO:
  - Monotonic scaling under ADDITION (not tested here; this script only
    optimizes for ablation effect).
  - Representational Independence (not implemented).
  - Validation that the output direction satisfies both properties before
    being treated as an RDO direction.

Before any version of this is run for a writeup-bound result:
  1. Adapt from the canonical repo, do not reimplement from prose.
  2. Check whether Wollschläger tested Gemma-2-2b-it; if yes, expect a
     specific published number and treat divergence as debuggable.
  3. Add a validation step verifying the output exhibits BOTH monotonic
     scaling under addition AND surgical ablation effect, before treating
     it as an RDO direction at all.

Parked here as a record of the homegrown sketch + scope-escalation in the
2026-05-25 session — see notes/scope_decisions.md. Do not delete.

----- ORIGINAL HOMEGROWN DOCSTRING BELOW -----

Phase 1 — RDO: Refusal Direction Optimization (Wollschläger 2025 method).

Gradient-based search for a unit vector d that, when ABLATED via the Arditi
multi-layer recipe, reduces the model's probability of producing
refusal-start tokens on a batch of harmful prompts. Optionally constrained
to be orthogonal to a previously-found direction (default: L13 diff-of-means)
so we test for *additional* causal directions beyond Arditi's.

Why this matters: our `phase1_subspace_ablation.py` showed that statistical
extraction (diff-of-means at non-peak layers, LDA from any bootstrap, LDA
top-5 subspace) does NOT recover additional causal directions on
Gemma-2-2b-it. But Wollschläger's multi-direction "polyhedral cone" claim
uses gradient extraction, not statistical extraction. The two methods can
disagree. RDO is the gradient method, and applying it here tells us:

  - If RDO finds an orthogonal-to-d_hat-L13 direction whose ablation also
    drops refusal: multi-D causal claim holds on Gemma-2-2b-it under
    gradient extraction. Convergent with Wollschläger.
  - If RDO fails to find any such direction (loss doesn't decrease, or
    decreases but the optimized direction's ablation behavioral test stays
    at baseline refusal): single-direction picture extends to gradient
    methods too on this model. Divergent from Wollschläger; a real
    methodological finding.

The objective:
  We want to find d (unit vector) such that ablating d reduces the
  probability of refusal-start tokens. Loss = sum_{harmful} log p(refusal-
  starter | prompt, ablate(d)). Minimize.

  Refusal-starter tokens: "I" (covers "I cannot", "I can't", "I'm sorry",
  "I am", "I won't"...) and "As" (covers "As an AI"). Sum their probabilities.

Constraints (applied after each gradient step):
  - unit norm: d ← d / ||d||
  - orthogonal to anchor direction (default = L13 diff-of-means)

Pre-registered prediction (logged 2026-05-25 before running):
  - Sanity check (no orthogonality constraint, d_init = d_hat_L13): loss
    decreases to baseline-comply level within ~50 steps. Confirms gradients
    flow correctly and RDO can rediscover diff-of-means.
  - Real test (orthogonal to d_hat_L13, d_init = random):
    (A) Loss decreases significantly + behavioral refusal drops ≤ 0.3 →
        multi-D causal claim holds on Gemma; convergent with Wollschläger.
    (B) Loss plateaus near baseline OR loss decreases but behavioral refusal
        stays ≥ 0.7 → no additional causal direction found by gradient
        extraction; single-direction picture extends to RDO on this model.

  My logged guess (weak prior): (B) is more likely on Gemma-2-2b-it given
  its small size and the strong statistical-extraction null result. But
  this is the empirical question; predicting wrongly here would be
  informative.

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.phase1_rdo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F

from experiments._runner import (
    RESULTS,
    cached_activations,
    content_hash,
    get_logger,
    get_model,
    load_jsonl_pairs,
    new_run_dir,
    write_json,
)
from mech_security.activations import cache_resid
from mech_security.directions import ablate_dir, diff_of_means, unit
from mech_security.eval import refusal_rate
from mech_security.model import format_prompt, generate, tokenize_prompt

log = get_logger("phase1_rdo")


# Tokens that start refusal-pattern responses. We want to MINIMIZE the sum
# of their probabilities at the first-response-token position (= LAST token
# of the chat-template-formatted prompt).
REFUSAL_STARTER_STRS = ["I", "As", "I'm", "Sorry"]


def find_refusal_starter_tokens(bundle) -> torch.Tensor:
    """Return tensor of token IDs that count as refusal-start markers."""
    tok = bundle.model.tokenizer
    ids = set()
    for s in REFUSAL_STARTER_STRS:
        # Try with and without leading space (Gemma tokenizer is sentencepiece-style)
        for variant in (s, " " + s):
            encoded = tok(variant, add_special_tokens=False)["input_ids"]
            for t in encoded:
                ids.add(int(t))
    out = torch.tensor(sorted(ids), dtype=torch.long)
    log.info("refusal-starter token IDs (%d total): %s",
             len(out),
             {int(t): tok.decode([int(t)]) for t in out[:20]})
    return out


def project_orthogonal(d: torch.Tensor, anchor: torch.Tensor | None) -> torch.Tensor:
    """Project d onto the orthogonal complement of `anchor`, then unit-normalize.
    If anchor is None, just unit-normalize."""
    if anchor is not None:
        d = d - torch.dot(d, anchor) * anchor
    n = d.norm()
    return d / (n + 1e-12)


def rdo_loss(bundle, d: torch.Tensor, prompt_tokens_list: list[torch.Tensor],
              refusal_tokens: torch.Tensor) -> torch.Tensor:
    """For each prompt, run the forward pass with d ablated at every residual
    hook, take the logit at the last position, compute the sum of probabilities
    on refusal-starter tokens. Return the mean over the batch.

    Lower loss = lower probability of starting with refusal = compliance.
    """
    d_dev = d.to(bundle.model.cfg.device).to(bundle.model.W_E.dtype)
    losses = []

    with ablate_dir(bundle.model, d_dev):
        for tokens in prompt_tokens_list:
            tokens_dev = tokens.to(bundle.model.cfg.device)
            logits = bundle.model(tokens_dev, return_type="logits")
            # logits: [1, seq_len, vocab]; we want the LAST position
            last_logits = logits[0, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)
            refusal_log_probs = log_probs[refusal_tokens.to(log_probs.device)]
            # logsumexp of the refusal-starter probabilities = log p(start with refusal)
            log_p_refusal_start = torch.logsumexp(refusal_log_probs, dim=0)
            losses.append(log_p_refusal_start)
    return torch.stack(losses).mean()


def eval_refusal_rate(bundle, d: torch.Tensor, prompts: list[str],
                       max_new_tokens: int = 96) -> tuple[float, int]:
    """Behavioral evaluation: refusal rate on `prompts` under ablation of d.
    Returns (rate, n_refused). Uses the substring scorer for speed; cross-
    check with LLM judge happens in the parent runner only on key cells.
    """
    with ablate_dir(bundle.model, d):
        gens = [generate(bundle, p, max_new_tokens=max_new_tokens, temperature=0.0)
                for p in prompts]
    r = refusal_rate(gens)
    return r.rate, r.n_refused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "orthogonal"], default="orthogonal",
                    help="'sanity': d_init = d_hat_L13, no orthogonality; "
                         "should rediscover diff-of-means. "
                         "'orthogonal': d_init = random in orthogonal complement "
                         "of d_hat_L13; test for additional causal direction.")
    ap.add_argument("--n-train", type=int, default=32,
                    help="Number of harmful prompts for the gradient loss batch.")
    ap.add_argument("--n-eval", type=int, default=12,
                    help="Number of held-out harmful prompts for behavioral eval.")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=50,
                    help="Behavioral eval frequency (in optimization steps).")
    ap.add_argument("--extract-layer", type=int, default=13)
    args = ap.parse_args()

    run_dir = new_run_dir("phase1_rdo")
    log.info("run_dir: %s | mode=%s n_steps=%d lr=%.3f n_train=%d n_eval=%d",
             run_dir, args.mode, args.n_steps, args.lr, args.n_train, args.n_eval)

    bundle = get_model()
    log.info("model: %s | device=%s d_model=%d",
             bundle.name, bundle.device, bundle.d_model)

    # === Load prompts and split ===
    pairs_path = Path(__file__).resolve().parent.parent / "data/contrastive.jsonl"
    harmful, harmless = load_jsonl_pairs(pairs_path)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(harmful))
    train_prompts = [harmful[i] for i in perm[:args.n_train]]
    eval_prompts = [harmful[i] for i in perm[args.n_train:args.n_train + args.n_eval]]
    log.info("train: %d, eval: %d (held out)", len(train_prompts), len(eval_prompts))

    # Pre-tokenize train prompts (we don't want to retokenize every step)
    train_tokens = []
    for p in train_prompts:
        text = format_prompt(p)
        ids = tokenize_prompt(bundle, text)
        train_tokens.append(ids)

    # === Anchor direction (d_hat_L13) ===
    extra = (f"{bundle.name}|dtype={bundle.model.cfg.dtype}|L{args.extract_layer}|"
             f"resid_post|last_token|advbench_full")
    key_h = content_hash(harmful, extra=extra + "|harmful")
    key_l = content_hash(harmless, extra=extra + "|harmless")
    log.info("extracting d_hat_L13 ...")
    H = cached_activations(key_h, lambda: cache_resid(bundle, harmful, layer=args.extract_layer, show_progress=False))
    L = cached_activations(key_l, lambda: cache_resid(bundle, harmless, layer=args.extract_layer, show_progress=False))
    d_hat = unit(diff_of_means(H, L))  # shape [d_model]
    log.info("d_hat_L13 extracted | norm=%.4f", float(d_hat.norm()))

    refusal_tokens = find_refusal_starter_tokens(bundle)

    # === Initialize d ===
    torch.manual_seed(args.seed)
    if args.mode == "sanity":
        d = d_hat.clone().detach().to(torch.float32).requires_grad_(True)
        anchor = None
        log.info("sanity mode: d_init = d_hat_L13; no orthogonality constraint")
    else:
        # Random init in orthogonal complement of d_hat
        d_init = torch.randn(bundle.d_model, dtype=torch.float32)
        d_init = d_init - torch.dot(d_init, d_hat.float()) * d_hat.float()
        d_init = d_init / d_init.norm()
        d = d_init.detach().requires_grad_(True)
        anchor = d_hat.float()
        log.info("orthogonal mode: d_init random in orthogonal complement of d_hat_L13; "
                 "initial cos(d, d_hat) = %.6f", float(torch.dot(d.detach(), anchor)))

    # === Baseline behavioral eval (no hook = ablation of zero vector) ===
    log.info("baseline behavioral eval (no hook) on n=%d harmful held-out ...",
             len(eval_prompts))
    base_gens = [generate(bundle, p, max_new_tokens=96, temperature=0.0)
                  for p in eval_prompts]
    base_rate = refusal_rate(base_gens).rate
    log.info("baseline refusal rate: %.2f", base_rate)

    # === Initial RDO loss ===
    with torch.no_grad():
        init_loss = rdo_loss(bundle, d.detach(), train_tokens, refusal_tokens)
    log.info("initial RDO loss (mean log p(refusal-start)): %.4f (lower = less refusal)",
             float(init_loss))

    # === Optimization loop ===
    history = []
    optimizer = torch.optim.SGD([d], lr=args.lr, momentum=0.9)
    for step in range(args.n_steps):
        optimizer.zero_grad()
        loss = rdo_loss(bundle, d, train_tokens, refusal_tokens)
        loss.backward()
        optimizer.step()
        # Apply constraints
        with torch.no_grad():
            d.data = project_orthogonal(d.data, anchor)
        history.append({"step": step + 1, "loss": float(loss)})
        if (step + 1) % args.eval_every == 0 or step == 0:
            log.info("step %d/%d | loss=%.4f | cos(d, d_hat)=%.4f",
                     step + 1, args.n_steps, float(loss),
                     float(torch.dot(d.detach(), d_hat.float())))
            # Periodic behavioral eval
            rate, n_ref = eval_refusal_rate(bundle, d.detach(), eval_prompts)
            history[-1]["eval_refusal_rate"] = rate
            log.info("  behavioral refusal rate at step %d: %.2f (%d/%d)",
                     step + 1, rate, n_ref, len(eval_prompts))

    # === Final behavioral eval ===
    final_d = d.detach()
    log.info("final cos(d, d_hat_L13) = %.6f", float(torch.dot(final_d, d_hat.float())))
    log.info("final RDO loss: %.4f (baseline was %.4f)", history[-1]["loss"], float(init_loss))
    log.info("final behavioral eval on n=%d harmful ...", len(eval_prompts))
    final_rate, final_n_ref = eval_refusal_rate(bundle, final_d, eval_prompts)
    log.info("FINAL refusal rate: %.2f (%d/%d), baseline was %.2f",
             final_rate, final_n_ref, len(eval_prompts), base_rate)

    record = {
        "step": "phase1_rdo",
        "model": bundle.name,
        "mode": args.mode,
        "extract_layer": args.extract_layer,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "seed": args.seed,
        "initial_loss": float(init_loss),
        "final_loss": history[-1]["loss"],
        "baseline_refusal_rate": base_rate,
        "final_refusal_rate": final_rate,
        "n_refused": final_n_ref,
        "delta_refusal": base_rate - final_rate,
        "final_cos_with_d_hat_L13": float(torch.dot(final_d, d_hat.float())),
        "training_history": history,
        # Save the direction in the record (small; d_model = 2304)
        "final_direction": final_d.cpu().tolist(),
    }
    write_json(run_dir / "result.json", record)
    log.info("run record -> %s", run_dir / "result.json")

    summary_path = RESULTS / f"phase1_rdo_{args.mode}.md"
    summary_path.write_text(_render_summary(record))
    log.info("summary -> %s", summary_path)

    print(f"\nphase1_rdo ({args.mode}) | "
          f"loss {float(init_loss):.3f} → {history[-1]['loss']:.3f} | "
          f"refusal {base_rate:.2f} → {final_rate:.2f} | "
          f"cos(d, d_hat_L13) = {float(torch.dot(final_d, d_hat.float())):.4f}")
    return 0


def _render_summary(rec):
    md = [
        f"# Phase 1 — RDO ({rec['mode']} mode)",
        "",
        "**Pre-registered prediction:** "
        + ("(sanity) loss decreases to comply baseline within ~50 steps; "
           "behavioral refusal drops to ≤0.2; cos(d, d_hat_L13) stays ≈ 1.0. "
           "Confirms gradients flow."
           if rec["mode"] == "sanity"
           else
           "(orthogonal-to-d_hat_L13 test for additional causal direction). "
           "(A) loss decreases AND refusal drops ≤ 0.3 → multi-D causal claim "
           "holds under gradient extraction. "
           "(B) loss plateaus OR refusal stays ≥ 0.7 → no additional causal "
           "direction; single-direction extends to gradient methods. Logged "
           "weak prior: (B) more likely."),
        "",
        f"- Model: `{rec['model']}` | extract_layer L{rec['extract_layer']}",
        f"- n_train (gradient batch): {rec['n_train']} | n_eval (held-out): {rec['n_eval']}",
        f"- Optimizer: SGD(lr={rec['lr']}, momentum=0.9), {rec['n_steps']} steps",
        f"- Constraint: unit norm{' + orthogonal to d_hat_L13' if rec['mode'] == 'orthogonal' else ''}",
        "",
        "## Results",
        "",
        "| | initial | final |",
        "|---|---:|---:|",
        f"| RDO loss (log p(refusal-start)) | {rec['initial_loss']:.4f} | {rec['final_loss']:.4f} |",
        f"| Behavioral refusal rate (held-out n={rec['n_eval']}) | {rec['baseline_refusal_rate']:.2f} | {rec['final_refusal_rate']:.2f} |",
        f"| cos(d, d_hat_L13) | — | {rec['final_cos_with_d_hat_L13']:.4f} |",
        "",
        f"**Δ refusal rate: {rec['delta_refusal']:+.2f}**",
        "",
        "## Verdict",
        "",
    ]
    if rec["mode"] == "sanity":
        if rec["final_refusal_rate"] < 0.2:
            md.append("✅ Sanity check passes. Gradients flow, RDO can drive refusal down "
                      "when seeded at the known causal direction.")
        else:
            md.append("❌ Sanity check fails. Even seeded at d_hat_L13, RDO did not drive "
                      "refusal below 0.2. The gradient implementation has a bug; do NOT "
                      "trust the orthogonal-mode result without fixing this first.")
    else:
        if rec["delta_refusal"] >= 0.3:
            md.append(
                "🚨 **Multi-D causal claim holds on Gemma-2-2b-it under gradient "
                f"extraction.** RDO found a direction orthogonal to d_hat_L13 "
                f"(cos = {rec['final_cos_with_d_hat_L13']:.4f}) whose ablation drops "
                f"refusal by {rec['delta_refusal']:.2f}. Convergent with Wollschläger "
                "et al. The 'single causal direction' picture from statistical "
                "extraction is incomplete; gradient methods recover additional "
                "causal directions on this model. The methodological-decoupling "
                "finding from `phase1_subspace_ablation.md` survives — statistical "
                "extraction misses these directions — but the underlying mechanism "
                "is multi-D.")
        else:
            md.append(
                "✅ **Single-direction picture extends to gradient extraction on "
                f"Gemma-2-2b-it.** RDO did not find an additional causal direction "
                f"orthogonal to d_hat_L13 (Δ refusal = {rec['delta_refusal']:+.2f}, "
                "below the 0.3 threshold). On this specific (small) model, the "
                "single-direction story holds under BOTH statistical and gradient "
                "extraction. Whether this generalizes to larger models (where "
                "Wollschläger established multi-D with RDO) is the natural Phase 2 "
                "test.")
    md.append("")
    md.append("## Method note")
    md.append("")
    md.append(
        "RDO objective: minimize mean over harmful prompts of log p(refusal-start | "
        "prompt, ablate(d) at every residual hook). Refusal-start tokens = first-token "
        "IDs of {'I', 'As', \"I'm\", 'Sorry'} variants. After each gradient step, "
        "project d onto orthogonal complement of d_hat_L13 (if orthogonal mode) and "
        "unit-normalize. Gradients flow through the differentiable ablation hook "
        "(x ← x − (x·d)*d, linear in d for fixed x)."
    )
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
