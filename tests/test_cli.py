"""CLI tests for numpy-poisson-variance-guard."""
from __future__ import annotations

import json

import pytest

from numpy_poisson_variance_guard.cli import main


def test_diagnose_json_has_required_fields(capsys):
    rc = main(["diagnose", "--json", "--lam", "1e10", "--samples", "20000"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    for field in (
        "numpy_version",
        "lam",
        "n_samples",
        "numpy_var_over_lam",
        "safe_var_over_lam",
        "numpy_mean",
        "safe_mean",
        "affected",
        "detail",
    ):
        assert field in payload
    assert rc in (0, 1)
    assert rc == (1 if payload["affected"] else 0)


def test_diagnose_text_mode_prints_status_headline(capsys):
    rc = main(["diagnose", "--no-color", "--lam", "1e10", "--samples", "20000"])
    out = capsys.readouterr().out
    assert "numpy Poisson large-lam variance probe" in out
    assert "numpy version" in out
    assert rc in (0, 1)


def test_sample_json_outputs_requested_size():
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["sample", "--lam", "1e16", "--size", "500", "--seed", "1", "--json"])
    payload = json.loads(buf.getvalue())
    assert payload["lam"] == 1e16
    assert payload["size"] == 500
    assert len(payload["samples"]) == 500
    assert rc == 0


def test_sample_text_mode_prints_summary(capsys):
    rc = main(["sample", "--lam", "1e16", "--size", "500", "--seed", "1", "--no-color"])
    out = capsys.readouterr().out
    assert "safe_poisson sample summary" in out
    assert "empirical var/lam" in out
    assert rc == 0


def test_sample_is_reproducible_with_seed():
    import io
    import contextlib

    def run():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["sample", "--lam", "1e15", "--size", "200", "--seed", "99", "--json"])
        return json.loads(buf.getvalue())["samples"]

    assert run() == run()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "numpy-poisson-variance-guard" in out


def test_no_subcommand_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
