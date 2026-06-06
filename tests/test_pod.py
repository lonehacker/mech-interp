"""Unit tests for the pure orchestration logic in mech_security.pod (no pod, no network).

The provision/ssh/bootstrap paths are validated by a live run; these cover the decision logic that
turns runpodctl output + poll snapshots into ids and run outcomes — the parts that silently mis-decide.
"""
from mech_security.pod import (
    _detached_command,
    _parse_pod_id,
    _parse_snapshot,
    _parse_step_data_args,
    _run_outcome,
)


class TestDetachedCommand:
    """The bug that idled a pod: `nohup cd … &&` can't run (cd isn't an executable). The launch must
    wrap the compound in `bash -c`, redirect to the log, detach stdin, and background it."""

    def test_wraps_compound_in_bash_c_and_detaches(self):
        cmd = _detached_command("cd /workspace/x && python run.py --flag", "/workspace/r.log")
        assert cmd.startswith("nohup bash -c ")     # a bare `cd && …` runs under a shell
        assert "</dev/null" in cmd                   # frees the SSH channel → launch returns
        assert "> /workspace/r.log 2>&1" in cmd
        assert cmd.rstrip().endswith("&")            # detached / backgrounded


class TestParseStepDataArgs:
    """Pre-flight needs to read each step's data-sizing flags out of the runner command string so an
    infeasible run (the n200 crash) is caught locally, before a paid pod boots."""

    def test_extracts_flags(self):
        cmd = ("experiments/phase3_track1.py --base X --n-extract 200 --n-score 24 "
               "--harmless data/alpaca_harmless.jsonl --advbench data/advbench_harmful_behaviors.csv")
        a = _parse_step_data_args(cmd)
        assert a["n_extract"] == 200 and a["n_score"] == 24
        assert a["harmless"] == "data/alpaca_harmless.jsonl" and a["matched"] is None

    def test_defaults_when_absent(self):
        a = _parse_step_data_args("experiments/phase3_track1.py --base X --layers 18")
        assert a["n_extract"] == 40 and a["n_score"] == 32          # argparse defaults
        assert a["harmless"] == "data/alpaca_harmless.jsonl" and a["advbench"].endswith(".csv")
        assert a["n_probe"] == 0  # not a probe step

    def test_probe_step_fields(self):
        a = _parse_step_data_args("experiments/phase3_probe_ablation.py --layer 18 --k 3 "
                                  "--n-extract 60 --n-probe 120 --n-harmless 60")
        assert a["n_probe"] == 120 and a["n_harmless"] == 60 and a["n_extract"] == 60

    def test_matched_path_detected(self):
        a = _parse_step_data_args("experiments/phase3_track1.py --matched data/code_contrastive_matched.jsonl")
        assert a["matched"] == "data/code_contrastive_matched.jsonl"


class TestParsePodId:
    def test_extracts_from_messy_output(self):
        assert _parse_pod_id('Creating pod...\n  "id": "abc123def",\n ok') == "abc123def"

    def test_empty_when_absent(self):
        assert _parse_pod_id("no id in here") == ""


class TestParseSnapshot:
    def test_parses_key_values_split_on_first_equals(self):
        d = _parse_snapshot("DONE=0\nPROC=1\nGPU=99 %, 40 MiB\nLAST=a = b")
        assert d["DONE"] == "0" and d["PROC"] == "1" and d["GPU"] == "99 %, 40 MiB"
        assert d["LAST"] == "a = b"  # only the first '=' splits

    def test_ignores_lines_without_equals(self):
        assert _parse_snapshot("DONE=1\ngarbage line\nPROC=0") == {"DONE": "1", "PROC": "0"}


class TestRunOutcome:
    def test_keep_polling_returns_none(self):
        assert _run_outcome(done_seen=False, traceback_seen=False, proc_missing=0) is None

    def test_done(self):
        assert _run_outcome(True, False, 0) == "done"

    def test_crashed_on_traceback(self):
        assert _run_outcome(True, True, 0) == "crashed"

    def test_died_only_after_three_missing(self):
        assert _run_outcome(False, False, 2) is None
        assert _run_outcome(False, False, 3) == "died"

    def test_done_beats_proc_missing(self):
        # finished + process already exited (proc gone) → done, NOT died
        assert _run_outcome(True, False, 5) == "done"
