"""Run an experiment on a throwaway RunPod GPU — provision, push code, run, pull results, terminate.

A pod run touches money and a remote machine, so the whole flow lives in ONE readable place instead of
scattered bash. `experiments/run_pod.py` is the thin CLI over this; a "job" is just a Python list of
(label, command) steps. Each step here is a small function so the run is auditable top-to-bottom.

Each bit of resilience below was learned from a real failure (so it's commented, not clever):
  * SSH uses keepalives → a stalled connection errors in ~60 s instead of hanging the run forever.
  * Detached runs use `nohup … </dev/null &` → the launch returns immediately (else we'd never poll).
  * Provisioning retries across GPU types and skips pods RunPod kills on boot (Docker-Hub pull limits).
  * The caller ALWAYS terminates in a `finally`; termination is an authoritative API call, not a hook.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RUNPODCTL = "/Users/anshulsinghle/.local/bin/runpodctl"
SSH_KEY = "/Users/anshulsinghle/.runpod/ssh/runpodctl-ssh-key"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
API = "https://rest.runpod.io/v1"
# keepalive so a dead SSH connection errors in ~60 s rather than hanging the whole run
_SSH = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]


# ── Pure helpers (no subprocess/network) — unit-tested in tests/test_pod.py ──────────────────────────
def _parse_pod_id(text: str) -> str:
    """Extract the pod id from `runpodctl pod create` output (which isn't always clean JSON)."""
    m = re.search(r'"id"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else ""


def _parse_snapshot(text: str) -> dict[str, str]:
    """Parse the 'KEY=value' lines of a poll snapshot into a dict (split on the FIRST '=' only)."""
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _run_outcome(done_seen: bool, traceback_seen: bool, proc_missing: int) -> str | None:
    """Terminal outcome of a polled run, or None to keep polling. `done_seen`: log has DONE_VERDICTS
    or a Traceback; `traceback_seen`: it was a Traceback; `proc_missing`: consecutive polls with the
    experiment process absent (≥3 ⇒ it died silently, e.g. OOM)."""
    if done_seen:
        return "crashed" if traceback_seen else "done"
    if proc_missing >= 3:
        return "died"
    return None


def _detached_command(command: str, logfile: str) -> str:
    """Build the remote command that runs `command` detached, logging to `logfile`. Wrapped in
    `bash -c` so a compound `cd … && python …` runs under a shell — `nohup` can't exec a bare `cd`
    (the bug that left a pod idle). Stdin from /dev/null so the SSH launch returns immediately."""
    return f"nohup bash -c {shlex.quote(command)} > {logfile} 2>&1 </dev/null &"


def _parse_step_data_args(command: str) -> dict:
    """Extract the data-sizing flags from a phase3_track1 runner command so the orchestrator can
    check feasibility BEFORE provisioning a pod. Defaults mirror experiments/phase3_track1.py argparse
    (n_extract 40, n_score 32, advbench/harmless defaults). Pure + unit-tested in tests/test_pod.py."""
    def _int(flag: str, default: int) -> int:
        m = re.search(rf"{flag}\s+(\d+)", command)
        return int(m.group(1)) if m else default

    def _path(flag: str, default):
        m = re.search(rf"{flag}\s+(\S+)", command)
        return m.group(1) if m else default

    return {
        "n_extract": _int("--n-extract", 40),
        "n_score": _int("--n-score", 32),
        "n_probe": _int("--n-probe", 0),        # >0 ⇒ a phase3_probe_ablation step (different data need)
        "n_harmless": _int("--n-harmless", 60),
        "advbench": _path("--advbench", "data/advbench_harmful_behaviors.csv"),
        "harmless": _path("--harmless", "data/alpaca_harmless.jsonl"),
        "matched": _path("--matched", None),
    }


def _runpodctl(*args: str) -> dict:
    """Call `runpodctl … --output json` and parse it. `pod create` prints the id amid other text, so we
    fall back to a regex (`_parse_pod_id`) for that one case. Returns {} if nothing parseable came back."""
    out = subprocess.run([RUNPODCTL, *args, "--output", "json"], capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pod_id = _parse_pod_id(out)
        return {"id": pod_id} if pod_id else {}


@dataclass
class Pod:
    """A booted GPU pod we can SSH into. Make one with `provision`; always end with `terminate()`."""
    id: str
    ip: str
    port: int
    token: str

    def ssh(self, command: str, timeout: int = 180) -> subprocess.CompletedProcess:
        """Run one command on the pod and return it (stdout/stderr captured)."""
        return subprocess.run(["ssh", *_SSH, "-p", str(self.port), f"root@{self.ip}", command],
                              capture_output=True, text=True, timeout=timeout)

    def start_background(self, command: str, logfile: str) -> None:
        """Start a long job and return immediately (see `_detached_command`: `bash -c` so a `cd && …`
        compound runs under a shell, and `</dev/null` so the SSH launch doesn't park here)."""
        self.ssh(_detached_command(command, logfile), timeout=60)

    def terminate(self) -> int:
        """Delete the pod (stops billing) via the REST API — authoritative, unlike a cleanup hook that
        can be skipped or hang. Returns the HTTP status (204 = deleted, 404 = already gone)."""
        req = urllib.request.Request(f"{API}/pods/{self.id}", method="DELETE",
                                     headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code


def provision(name: str, gpu_specs: list[tuple[str, str]], disk_gb: int, token: str,
              term_hours: int = 5) -> Pod:
    """Create a pod and wait until it accepts SSH. Tries each (gpu, cloud) in `gpu_specs` in order,
    retrying the whole list every 8 min; skips pods RunPod kills on boot. Raises if none boot in 6 h."""
    deadline = time.time() + 6 * 3600
    term_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + term_hours * 3600))
    while time.time() < deadline:
        pod_id = ""
        for gpu, cloud in gpu_specs:
            created = _runpodctl("pod", "create", "--name", name, "--gpu-id", gpu, "--cloud-type", cloud,
                                 "--image", IMAGE, "--container-disk-in-gb", str(disk_gb),
                                 "--ports", "8888/http,22/tcp", "--terminate-after", term_at)
            pod_id = created.get("id", "")
            if pod_id:
                print(f"[pod] created {gpu}/{cloud} pod={pod_id}; waiting for boot")
                break
        if not pod_id:
            print("[pod] no GPU stock; retrying in 8 min")
            time.sleep(480)
            continue
        pod = _wait_for_ssh(pod_id, token)
        if pod:
            print(f"[pod] ready root@{pod.ip}:{pod.port}")
            return pod
        print("[pod] did not boot (dud / image-pull limit) — terminating and retrying")
        Pod(pod_id, "", 0, token).terminate()
        time.sleep(120)
    raise RuntimeError(f"no healthy pod within 6 h for {name!r}")


def _wait_for_ssh(pod_id: str, token: str, tries: int = 16) -> Pod | None:
    """Poll for SSH readiness (~4 min). Returns a Pod, or None if RunPod marks the pod EXITED (a dud)."""
    for _ in range(tries):
        info = _runpodctl("ssh", "info", pod_id)
        if info.get("ip") and info.get("port"):
            return Pod(pod_id, info["ip"], int(info["port"]), token)
        if _runpodctl("pod", "get", pod_id).get("desiredStatus") in ("EXITED", "TERMINATED"):
            return None
        time.sleep(15)
    return None


def push_and_setup(pod: Pod, repo: Path, anthropic_key: str) -> None:
    """Push the repo (never .env/.venv/.git), write the pod's .env, pip-install pinned deps (retry x3)."""
    excludes = ["mech-security/.venv", "mech-security/.git", "mech-security/.env", "mech-security/artifacts",
                "mech-security/logs", "mech-security/wandb", "mech-security/graphify-out",
                "*/__pycache__", "*.egg-info"]
    tar_cmd = ["tar", "czf", "-", "-C", str(repo.parent)]
    for e in excludes:
        tar_cmd += ["--exclude", e]
    tar_cmd.append(repo.name)
    src = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    subprocess.run(["ssh", *_SSH, "-p", str(pod.port), f"root@{pod.ip}",
                    "tar xzf - --no-same-owner -C /workspace"], stdin=src.stdout)
    src.wait()
    pod.ssh(f"printf 'ANTHROPIC_API_KEY=%s\\n' {shlex.quote(anthropic_key)} > /workspace/mech-security/.env")

    deps = ("echo 'torch==2.9.1+cu128' > /tmp/c.txt && cd /workspace/mech-security && "
            "pip install --break-system-packages -c /tmp/c.txt numpy==1.26.4 transformers==5.9.0 "
            "transformer-lens==3.2.1 scikit-learn==1.6.1 anthropic==0.45.2 python-dotenv && "
            "pip install --break-system-packages -e . --no-deps && "
            "python -c 'import torch,anthropic,mech_security.redteam' && echo DEPS_OK")
    for attempt in range(1, 4):
        if "DEPS_OK" in pod.ssh(deps, timeout=900).stdout:
            return
        print(f"[pod] bootstrap attempt {attempt} failed; retrying in 20 s")
        time.sleep(20)
    raise RuntimeError("pod bootstrap (pip) failed after 3 tries")


def wait_for_run(pod: Pod, logfile: str, heartbeat: Path, label: str, max_minutes: int = 150) -> str:
    """Poll a running job until it finishes, crashes, dies, or times out. Each ~45 s cycle writes a
    heartbeat file (GPU util/mem + last log line) so `scripts/pod_status.sh` shows live progress.
    Returns 'done' | 'crashed' | 'died' | 'timeout'."""
    deadline = time.time() + max_minutes * 60
    proc_missing = 0
    while time.time() < deadline:
        snap = pod.ssh(
            f"echo DONE=$(grep -cE 'DONE_VERDICTS:|PROBE_RESULT:|CONVERGE_RESULT:|Traceback .most' {logfile} 2>/dev/null); "
            f"echo PROC=$(pgrep -fc '[e]xperiments/phase3' 2>/dev/null || echo 0); "  # bracket avoids self-match
            f"echo GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1); "
            f"echo LAST=$(tail -1 {logfile} 2>/dev/null)", timeout=90).stdout
        f = _parse_snapshot(snap)
        heartbeat.write_text(f"{time.strftime('%H:%M:%S')}  {label}  gpu={f.get('GPU', '?').strip()}  "
                             f"proc={f.get('PROC', '?')}\nlast: {f.get('LAST', '')}\n")
        done_seen = f.get("DONE", "0") not in ("0", "")
        proc_missing = proc_missing + 1 if f.get("PROC") == "0" else 0
        traceback_seen = done_seen and pod.ssh(f"grep -q 'Traceback (most' {logfile}", timeout=60).returncode == 0
        outcome = _run_outcome(done_seen, traceback_seen, proc_missing)
        if outcome:
            return outcome
        time.sleep(45)
    return "timeout"


def pull_results(pod: Pod, repo: Path) -> None:
    """Copy all results/phase3_*.{md,json} from the pod back to local."""
    src = subprocess.Popen(["ssh", *_SSH, "-p", str(pod.port), f"root@{pod.ip}",
                            "cd /workspace/mech-security/results 2>/dev/null && "
                            "tar czf - phase3_*.json phase3_*.md 2>/dev/null"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "xzf", "-", "-C", str(repo / "results")], stdin=src.stdout)
    src.wait()
