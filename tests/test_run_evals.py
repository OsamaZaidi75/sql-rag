"""Unit tests for the --fail-under CI gate in evals/run_evals.py."""
from evals.run_evals import gate_exit


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
