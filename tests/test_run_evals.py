"""Unit tests for the --fail-under CI gate, --json verdict artifact,
--no-history flag, --top-k validation, and eval-history CSV in evals/run_evals.py."""
import csv
import json

import pytest

from evals.run_evals import (
    gate_exit,
    main,
    parse_args,
    record_eval_history,
    verdict_dict,
    write_json,
)


def test_no_gate_all_pass():
    assert gate_exit(30, 30, None) == 0


def test_no_gate_single_failure():
    assert gate_exit(29, 30, None) == 1


def test_gate_met(capsys):
    assert gate_exit(30, 30, 100) == 0
    assert "meets --fail-under" in capsys.readouterr().out


def test_gate_tolerates_misses_above_threshold(capsys):
    assert gate_exit(96, 100, 95) == 0
    assert "meets --fail-under" in capsys.readouterr().out


def test_gate_fails_below_threshold(capsys):
    assert gate_exit(94, 100, 95) == 1
    assert "below --fail-under" in capsys.readouterr().out


def test_gate_stricter_than_perfect_fails(capsys):
    assert gate_exit(100, 100, 100.5) == 1
    assert "below --fail-under" in capsys.readouterr().out


def test_verdict_dict_no_gate_all_pass():
    assert verdict_dict(30, 30, None) == {"accuracy": 100.0, "fail_under": None, "ok": True}


def test_verdict_dict_no_gate_single_failure():
    assert verdict_dict(29, 30, None) == {"accuracy": 96.7, "fail_under": None, "ok": False}


def test_verdict_dict_gate_met():
    assert verdict_dict(96, 100, 95) == {"accuracy": 96.0, "fail_under": 95, "ok": True}


def test_verdict_dict_gate_miss():
    v = verdict_dict(94, 100, 95)
    assert v == {"accuracy": 94.0, "fail_under": 95, "ok": False}


def test_gate_prints_json_verdict(capsys):
    gate_exit(96, 100, 95)
    line = capsys.readouterr().out.splitlines()[0]
    assert json.loads(line) == verdict_dict(96, 100, 95)


def test_write_json_roundtrip(tmp_path):
    payload = {"accuracy": 100.0, "fail_under": None, "ok": True,
               "results": [{"id": "q1", "ok": True, "detail": "2 rows"}]}
    path = str(tmp_path / "nested" / "verdict.json")
    write_json(path, payload)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == payload


def test_record_eval_history_creates_file_with_header(tmp_path):
    path = str(tmp_path / "daily" / "eval_history.csv")
    record_eval_history("2026-10-19", 30, 30, 100.0, llm="mock", path=path)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["date", "passed", "total", "accuracy_pct", "llm"]
    assert rows[1] == ["2026-10-19", "30", "30", "100.0", "mock"]


def test_record_eval_history_appends_without_duplicate_header(tmp_path):
    path = str(tmp_path / "eval_history.csv")
    record_eval_history("2026-10-18", 28, 30, 93.333, llm="mock", path=path)
    record_eval_history("2026-10-19", 30, 30, 100.0, llm="mock", path=path)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["date", "passed", "total", "accuracy_pct", "llm"]
    assert len(rows) == 3
    assert rows[2] == ["2026-10-19", "30", "30", "100.0", "mock"]


def test_record_eval_history_rounds_accuracy_and_blanks_llm(tmp_path):
    path = str(tmp_path / "eval_history.csv")
    record_eval_history("2026-10-19", 29, 30, 96.666, path=path)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["2026-10-19", "29", "30", "96.7", ""]


def test_no_history_defaults_off():
    assert parse_args([]).no_history is False


def test_no_history_flag_parses():
    args = parse_args(["--mock", "--no-history"])
    assert args.no_history is True
    assert args.mock is True


def test_no_history_skips_record_eval_history(monkeypatch):
    """--no-history: record_eval_history is never called (real CSV untouched)."""
    calls = []
    monkeypatch.setattr("evals.run_evals.record_eval_history",
                        lambda *a, **k: calls.append(1))
    assert main(["--mock", "--no-history"]) == 0
    assert calls == []


def test_history_recorded_by_default(monkeypatch):
    """Without the flag, a --mock run still records one history row."""
    calls = []
    monkeypatch.setattr("evals.run_evals.record_eval_history",
                        lambda *a, **k: calls.append(1))
    assert main(["--mock"]) == 0
    assert len(calls) == 1


def test_fail_under_rejects_above_100(capsys):
    """>100 can never be met — a threshold that guarantees a red gate is a bug."""
    with pytest.raises(SystemExit) as exc:
        parse_args(["--fail-under", "120"])
    assert exc.value.code == 2
    assert "--fail-under must be between 0 and 100" in capsys.readouterr().err


def test_fail_under_rejects_negative(capsys):
    """A negative threshold always passes, hiding real regressions."""
    with pytest.raises(SystemExit) as exc:
        parse_args(["--fail-under", "-5"])
    assert exc.value.code == 2
    assert "--fail-under must be between 0 and 100" in capsys.readouterr().err


def test_fail_under_accepts_boundaries():
    """0 and 100 are meaningful thresholds and must still parse."""
    assert parse_args(["--fail-under", "0"]).fail_under == 0
    assert parse_args(["--fail-under", "100"]).fail_under == 100
    assert parse_args(["--mock"]).fail_under is None


def test_top_k_rejects_zero(capsys):
    """top-k of 0 retrieves nothing — a confusing silent failure."""
    with pytest.raises(SystemExit) as exc:
        parse_args(["--top-k", "0"])
    assert exc.value.code == 2
    assert "--top-k must be a positive integer" in capsys.readouterr().err


def test_top_k_rejects_negative(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--top-k", "-3"])
    assert exc.value.code == 2
    assert "--top-k must be a positive integer" in capsys.readouterr().err


def test_top_k_accepts_positive():
    """1 is the smallest meaningful retrieval depth; the default is untouched."""
    assert parse_args(["--top-k", "1"]).top_k == 1
    assert parse_args(["--mock"]).top_k == 5
