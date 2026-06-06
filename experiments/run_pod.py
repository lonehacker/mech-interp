"""Run a pod experiment end-to-end: provision → push code → run each step → pull results → terminate.

Thin, readable wrapper over `mech_security.pod`. Experiments are declared in JOBS below — to add one,
add an entry (no new script). Run in the background; watch it live with `bash scripts/pod_status.sh`.

    python experiments/run_pod.py <job_name>
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import dotenv_values

from mech_security import pod
from mech_security import track1_splits as ts

REPO = Path(__file__).resolve().parent.parent

# GPU preference order for an 8B run: A100-80 fits the TL fp32 peak; H100-80 fallback; community last.
A100_80 = [("NVIDIA A100 80GB PCIe", "SECURE"), ("NVIDIA A100-SXM4-80GB", "SECURE"),
           ("NVIDIA H100 80GB HBM3", "SECURE"), ("NVIDIA A100-SXM4-80GB", "COMMUNITY")]
_POD_ENV = "HF_HOME=/workspace/hf PYTORCH_ALLOC_CONF=expandable_segments:True"

# Shared Llama-8B attack args (vanilla-only; corrected extraction; general Alpaca harmless reference;
# LOW-k {1,2,3} per the reframe — k≥5 is uninterpretable; 128 tokens for known-ground-comparable S).
_LLAMA = ("experiments/phase3_track1.py --base meta-llama/Meta-Llama-3-8B-Instruct "
          "--vanilla NousResearch/Meta-Llama-3-8B-Instruct --spine --no-processing --device cuda "
          "--harmless data/alpaca_harmless.jsonl --ks 1 2 3 --seeds 42 1337 --max-new-tokens 128")
# Shared probe-after-ablation args (the Goal-B centerpiece): ablate clean k=3 at L18, probe leftover refusal.
_PROBE = ("experiments/phase3_probe_ablation.py --no-processing --device cuda --layer 18 --k 3 "
          "--n-extract 60 --n-probe 120 --n-harmless 60 --max-new-tokens 128")
# Shared d̂-convergence args (H-extract check): does the direction stabilize as extraction n grows? Cheap (no gen).
_CONVERGE = ("experiments/phase3_dhat_converge.py --no-processing --device cuda --layer 18 "
             "--ns 50 100 200 --n-extract 200")

# Each job: which GPUs to try, container disk, and an ordered list of (label, runner command) run on ONE pod.
JOBS: dict[str, dict] = {
    # Phase 3 Step 1 (H3): is Llama's 0.52 floor an extraction artifact? Position sweep + n_extract=200.
    "llama_h3": {
        "gpu_specs": A100_80,
        "disk_gb": 150,
        "steps": [
            ("llama_pos",  f"{_LLAMA} --layers 18 20 --positions -1 -2 -3 -4 -5 "
                           "--n-extract 100 --n-score 24 --out results/phase3_llama_pos.md"),
            ("llama_n200", f"{_LLAMA} --layers 18 20 --positions -1 "
                           "--n-extract 200 --n-score 24 --out results/phase3_llama_n200.md"),
        ],
    },
    # Goal B (2026-06-06): WHY does diff-of-means underperform on Llama? ONE pre-registered batched run:
    # n=200 denominator + low-k sweep, then probe-after-ablation (H-dim vs H-nonlinear) + Qwen positive control.
    "llama_goalb": {
        "gpu_specs": A100_80,
        "disk_gb": 150,
        "steps": [
            ("llama_attack",   f"{_LLAMA} --layers 18 20 --positions -1 --n-extract 200 --n-score 40 "
                               "--out results/phase3_llama_lowk_n200.md"),
            ("llama_converge", f"{_CONVERGE} --ckpt NousResearch/Meta-Llama-3-8B-Instruct "
                               "--base meta-llama/Meta-Llama-3-8B-Instruct --out results/phase3_dhat_converge_llama.json"),
            ("llama_probe",    f"{_PROBE} --ckpt NousResearch/Meta-Llama-3-8B-Instruct "
                               "--base meta-llama/Meta-Llama-3-8B-Instruct --out results/phase3_probe_llama.json"),
            ("qwen_probe",     f"{_PROBE} --ckpt Qwen/Qwen2.5-3B-Instruct "
                               "--base Qwen/Qwen2.5-3B-Instruct --out results/phase3_probe_qwen.json"),
        ],
    },
}


def preflight(job: dict) -> None:
    """Assert every step's data requirements are satisfiable from local files BEFORE we provision a
    paid pod (the n200 lesson: a run that needs more rows than the file has should fail here, in
    seconds, not after boot). Raises SystemExit naming the offending step."""
    for label, command in job["steps"]:
        a = pod._parse_step_data_args(command)
        if a["n_probe"]:  # phase3_probe_ablation: harmful = n_extract + n_probe (disjoint), harmless = n_harmless
            have_h = len(ts._advbench_goals(str(REPO / a["advbench"])))
            have_hl = len(ts._harmless(str(REPO / a["harmless"])))
            need_h, need_hl = a["n_extract"] + a["n_probe"], a["n_harmless"]
            ok = have_h >= need_h and have_hl >= need_hl
            reason = f"probe: harmful {have_h}>={need_h}, harmless {have_hl}>={need_hl}"
        elif a["matched"]:
            ok, reason = ts.feasibility(matched_path=str(REPO / a["matched"]),
                                        n_extract=a["n_extract"], n_score=a["n_score"])
        else:
            ok, reason = ts.feasibility(advbench_path=str(REPO / a["advbench"]), harmless_path=str(REPO / a["harmless"]),
                                        n_extract=a["n_extract"], n_score=a["n_score"])
        print(f"[run_pod] pre-flight {label}: {reason}")
        if not ok:
            raise SystemExit(f"[run_pod] PRE-FLIGHT FAIL ({label}): {reason} — NOT provisioning a pod.")


def main(job_name: str) -> int:
    job = JOBS[job_name]
    env = dotenv_values(REPO / ".env")
    heartbeat = REPO / "logs" / f"{job_name}.heartbeat"

    preflight(job)  # fail locally on infeasible data BEFORE spending on a pod
    p = pod.provision(job_name, job["gpu_specs"], job["disk_gb"], env["RUNPOD_API_KEY"])
    try:
        pod.push_and_setup(p, REPO, env["ANTHROPIC_API_KEY"])
        print("[run_pod] bootstrap done")
        for label, command in job["steps"]:
            print(f"[run_pod] starting step: {label}")
            p.start_background(f"cd /workspace/mech-security && {_POD_ENV} python -u {command}",
                               f"/workspace/{label}.log")
            outcome = pod.wait_for_run(p, f"/workspace/{label}.log", heartbeat, label)
            pod.pull_results(p, REPO)
            (REPO / "logs" / f"{label}.log").write_text(p.ssh(f"cat /workspace/{label}.log", timeout=120).stdout)
            print(f"[run_pod] step {label}: {outcome}")
            if outcome != "done":
                raise RuntimeError(f"step {label} ended '{outcome}' — see logs/{label}.log")
    finally:
        print(f"[run_pod] terminating pod {p.id} → HTTP {p.terminate()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
