"""
Phase 2 prep — build the code-refusal contrastive set.

Curates 150 code-harmful + 150 code-harmless prompts from public sources
for Phase 2 (Qwen2.5-3B-Instruct port). The code-refusal angle is chosen
to keep some of the Qwen-Coder spirit from the master spec without needing
the Coder fine-tune variant (which TransformerLens 3.2.1 doesn't ship).

Sources (all public, no novel harmful content generated):
  Harmful side:
    - HarmBench `cybercrime_intrusion` (40 prompts, already in data/)
    - AdvBench harmful_behaviors.csv (520 prompts), filtered for code keywords
  Harmless side:
    - CodeAlpaca-20k instruction prompts, length-matched to harmful side

Per CLAUDE.md: no novel harmful content. Every harmful prompt here is a
verbatim string from a published benchmark.

Output: data/code_contrastive.jsonl (one JSON per line, schema mirrors
data/contrastive.jsonl: {text, label, source, stem}).

Run:
    cd ~/safe_ai/mech-security
    python -m experiments.build_code_contrastive
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_code_contrastive")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SEED = 20260529
TARGET_PER_SIDE = 150
SOURCE_HARM_HB = DATA / "harmbench_behaviors_text_all.csv"
ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ADVBENCH_LOCAL = DATA / "advbench_harmful_behaviors.csv"
OUT = DATA / "code_contrastive.jsonl"

# Words/phrases that mark a prompt as code/cyber-themed. Imperfect but
# good enough for a first-pass filter; the result is hand-spot-checked.
CODE_KEYWORDS = re.compile(
    r"\b(code|coding|script|program|software|"
    r"malware|virus|worm|trojan|ransomware|spyware|keylogger|rootkit|botnet|"
    r"exploit|payload|shellcode|backdoor|vulnerability|"
    r"hack|hacking|cyberattack|cybercrime|intrusion|phish|phishing|spoof|"
    r"sql\s*injection|xss|csrf|ddos|denial[\s-]of[\s-]service|brute[\s-]force|"
    r"password|credential|token|hash|crack|crypt|decrypt|"
    r"network|firewall|router|server|database|api|endpoint|port\s*scan|"
    r"linux|windows|kernel|driver|registry|"
    r"github|git|repo|pull request|"
    r"python|javascript|bash|shell|powershell|c\+\+|java|ruby|perl|"
    r"function|library|module|package|"
    r"deepfake|surveillance|wiretap|monitoring software)\b",
    re.IGNORECASE,
)


def _download_advbench() -> Path:
    """Fetch the public AdvBench harmful_behaviors.csv from GitHub if not
    already on disk. The file is permanently public at the llm-attacks repo."""
    if ADVBENCH_LOCAL.exists():
        log.info("AdvBench already at %s (%d bytes)", ADVBENCH_LOCAL, ADVBENCH_LOCAL.stat().st_size)
        return ADVBENCH_LOCAL
    log.info("Downloading AdvBench from %s ...", ADVBENCH_URL)
    urllib.request.urlretrieve(ADVBENCH_URL, ADVBENCH_LOCAL)
    log.info("Saved %d bytes to %s", ADVBENCH_LOCAL.stat().st_size, ADVBENCH_LOCAL)
    return ADVBENCH_LOCAL


def _load_harmbench_cybercrime() -> list[str]:
    """Parse HarmBench CSV; return prompts in cybercrime_intrusion category."""
    out = []
    with SOURCE_HARM_HB.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["FunctionalCategory"] == "standard"
                    and row["SemanticCategory"] == "cybercrime_intrusion"):
                out.append(row["Behavior"].strip())
    log.info("HarmBench cybercrime_intrusion: %d prompts", len(out))
    return out


def _load_advbench_code_filtered() -> list[str]:
    """Load full AdvBench, keep only prompts that match CODE_KEYWORDS."""
    path = _download_advbench()
    out = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            # AdvBench schema: "goal", "target"
            prompt = row.get("goal", "").strip()
            if not prompt: continue
            if CODE_KEYWORDS.search(prompt):
                out.append(prompt)
    log.info("AdvBench code-keyword filter: %d / 520 prompts kept", len(out))
    return out


def _load_codealpaca(length_target: tuple[int, int]) -> list[str]:
    """Load CodeAlpaca-20k via HuggingFace, filter by char length to match
    the harmful side's distribution. Use the `instruction` field (the user-
    visible prompt) and ignore `input` (which is often code context)."""
    from datasets import load_dataset
    log.info("Loading CodeAlpaca-20k from HuggingFace (sahil2801/CodeAlpaca-20k)...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    log.info("Loaded %d CodeAlpaca rows", len(ds))
    lo, hi = length_target
    candidates = []
    for row in ds:
        prompt = row.get("instruction", "").strip()
        # Skip if the prompt depends on the `input` field (code context) —
        # those don't work standalone.
        ctx = row.get("input", "").strip()
        if ctx:
            continue
        L = len(prompt)
        if lo <= L <= hi:
            candidates.append(prompt)
    log.info("CodeAlpaca length-matched (%d–%d chars, no input ctx): %d", lo, hi, len(candidates))
    return candidates


def _stem(s: str) -> str:
    """First word, lowercased — used as a coarse imperative-stem proxy
    (matches existing data/contrastive.jsonl schema)."""
    return re.split(r"\W+", s.strip())[0].lower() if s else ""


def _content_hash(items: list[dict]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(it, sort_keys=True).encode())
    return h.hexdigest()[:12]


def main() -> int:
    random.seed(SEED)

    # 1. Harmful side
    hb_cybercrime = _load_harmbench_cybercrime()
    advbench_code = _load_advbench_code_filtered()
    # Dedup across sources by lowercased prompt
    seen = set()
    harmful = []
    for src_name, prompts in [
        ("harmbench_cybercrime", hb_cybercrime),
        ("advbench_code", advbench_code),
    ]:
        for p in prompts:
            key = p.lower().strip()
            if key in seen: continue
            seen.add(key)
            harmful.append({"text": p, "label": "harmful", "source": src_name, "stem": _stem(p)})
    log.info("Combined harmful (deduped): %d", len(harmful))

    # If we have more than TARGET_PER_SIDE, downsample with fixed seed
    if len(harmful) > TARGET_PER_SIDE:
        # Keep all HarmBench (40) + sample from AdvBench to fill the rest
        hb_records = [r for r in harmful if r["source"] == "harmbench_cybercrime"]
        adv_records = [r for r in harmful if r["source"] == "advbench_code"]
        n_adv = TARGET_PER_SIDE - len(hb_records)
        if n_adv > len(adv_records):
            log.warning("Only %d AdvBench prompts; can't reach %d total", len(adv_records), TARGET_PER_SIDE)
            n_adv = len(adv_records)
        random.shuffle(adv_records)
        harmful = hb_records + adv_records[:n_adv]
        log.info("Downsampled harmful: %d HarmBench + %d AdvBench = %d", len(hb_records), n_adv, len(harmful))
    elif len(harmful) < TARGET_PER_SIDE:
        log.warning("Only %d harmful prompts (target %d) — keeping what we have", len(harmful), TARGET_PER_SIDE)

    # 2. Harmless side — length-matched
    harm_lens = sorted(len(r["text"]) for r in harmful)
    median = harm_lens[len(harm_lens) // 2]
    p10, p90 = harm_lens[len(harm_lens) // 10], harm_lens[(9 * len(harm_lens)) // 10]
    log.info("Harmful char-length distribution: median=%d, 10%%=%d, 90%%=%d", median, p10, p90)
    # Use [p10, p90] as the harmless filter window
    candidates = _load_codealpaca(length_target=(p10, p90))
    random.shuffle(candidates)
    harmless = [
        {"text": p, "label": "harmless", "source": "codealpaca", "stem": _stem(p)}
        for p in candidates[:TARGET_PER_SIDE]
    ]
    log.info("Sampled harmless: %d", len(harmless))

    # 3. Write the file
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in harmful + harmless:
            f.write(json.dumps(r) + "\n")
    h = _content_hash(harmful + harmless)
    log.info("Wrote %d records to %s | content-hash=%s", len(harmful) + len(harmless), OUT, h)

    # 4. Quick stats
    harmless_lens = sorted(len(r["text"]) for r in harmless)
    print("\n=== code_contrastive.jsonl ===")
    print(f"  harmful:  {len(harmful)} | char len median={harm_lens[len(harm_lens)//2]}, "
          f"range=[{harm_lens[0]}, {harm_lens[-1]}]")
    print(f"  harmless: {len(harmless)} | char len median={harmless_lens[len(harmless_lens)//2]}, "
          f"range=[{harmless_lens[0]}, {harmless_lens[-1]}]")
    print(f"  hash: {h}")
    print(f"  → {OUT}")

    # 5. Spot-sample a few
    print("\n  Harmful samples:")
    for r in random.sample(harmful, min(3, len(harmful))):
        print(f"    [{r['source']}] {r['text'][:110]}")
    print("  Harmless samples:")
    for r in random.sample(harmless, min(3, len(harmless))):
        print(f"    [{r['source']}] {r['text'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
