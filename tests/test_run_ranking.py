"""Tests for the ranking analysis script.

Focus is the stratification guarantee: short (n<200) and long (n>=200)
series must be ranked separately and never pooled. The two strata have
opposite-signed effects at the small end (tiny is 0.972 short vs 1.092
long), so a pooled ranking would average away the structure being
measured.

The printed report is covered as carefully as the matrices are. A
swapped stratum label or an inverted significance mark leaves the JSON
artifacts perfectly correct, so the error is invisible in the data and
survives into the prose someone writes from the terminal output - which
is the only place these numbers are read in narrative form.
"""
import json
import sys
from datetime import datetime, timedelta

import pytest

import experiments.scripts.run_ranking as rr
import experiments.scripts.run_size_scaling as rss
from shortseq.evaluation.ranking import nemenyi_critical_difference


def _write_scaling(tmp_path, dataset, size, n, rmse, arima_rmse=100.0, failure=None):
    """Write one synthetic scaling result in the schema run_one() emits.

    `rmse=None` writes the shape a FAILED run leaves behind: an empty
    `metrics` dict plus the reason under `failed_models`. That shape has
    to be reachable from here - a helper that can only write successes
    cannot exercise the per-size failure accounting at all.
    """
    key = f"Chronos-{size}"
    payload = {
        "dataset": dataset,
        "size": size,
        "n": n,
        "stratum": "short" if n < 200 else "long",
        "n_train": int(n * 0.8),
        "n_test": n - int(n * 0.8),
        "arima_rmse": arima_rmse,
        "metrics": {} if rmse is None else {key: {"rmse": rmse}},
        "failed_models": {} if failure is None else {key: failure},
    }
    with open(tmp_path / f"{dataset}_chronos_{size}.json", "w") as f:
        json.dump(payload, f)


def _write_grid(tmp_path, datasets, sizes, n=100):
    """A complete, monotonic grid: later sizes score better on every series."""
    for i, dataset in enumerate(datasets):
        for j, size in enumerate(sizes):
            _write_scaling(tmp_path, dataset, size, n=n, rmse=90.0 - 10.0 * j + i)


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_ranking.py", *argv])
    rr.main()


def test_build_matrices_separates_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size, rmse in [("tiny", 90.0), ("small", 80.0), ("large", 70.0)]:
        _write_scaling(tmp_path, "s1", size, n=100, rmse=rmse)
        _write_scaling(tmp_path, "s2", size, n=150, rmse=rmse)
        _write_scaling(tmp_path, "L1", size, n=900, rmse=rmse)

    matrices = rr.build_matrices()

    assert set(matrices) == {"short", "long"}
    assert list(matrices["short"].index) == ["s1", "s2"]
    assert list(matrices["long"].index) == ["L1"]
    assert "pooled" not in matrices


def test_build_matrices_columns_are_sizes_in_scaling_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size in ["large", "tiny", "small"]:  # deliberately unsorted on disk
        _write_scaling(tmp_path, "s1", size, n=100, rmse=50.0)
    matrices = rr.build_matrices()
    # Columns must follow parameter-count order, not filesystem order,
    # so the report reads tiny -> large.
    assert list(matrices["short"].columns) == ["tiny", "small", "large"]


def test_build_matrices_drops_incomplete_series(tmp_path, monkeypatch):
    # A series missing one size would make the grid incomplete, which the
    # Nemenyi CD formula does not tolerate. It must be dropped loudly,
    # not silently NaN-filled.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size in ["tiny", "small", "large"]:
        _write_scaling(tmp_path, "complete", size, n=100, rmse=50.0)
    _write_scaling(tmp_path, "partial", "tiny", n=100, rmse=50.0)

    matrices = rr.build_matrices()

    assert list(matrices["short"].index) == ["complete"]
    assert not matrices["short"].isna().any().any()


def test_analyse_stratum_returns_ranking_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for i in range(15):
        _write_scaling(tmp_path, f"s{i}", "tiny", n=100, rmse=90.0 + i)
        _write_scaling(tmp_path, f"s{i}", "small", n=100, rmse=80.0 + i)
        _write_scaling(tmp_path, f"s{i}", "large", n=100, rmse=70.0 + i)

    matrices = rr.build_matrices()
    payload = rr.analyse(matrices["short"])

    assert payload["n_series"] == 15
    assert payload["n_models"] == 3
    assert payload["mean_ranks"]["large"] < payload["mean_ranks"]["tiny"]
    # Key presence would pass with the CD hardcoded to zero (making every
    # pair "significant") or inflated by the sqrt(2) the module docstring
    # warns about. Pin the value: k=3, N=15, alpha=0.05.
    assert payload["critical_difference"] == pytest.approx(0.85579845, abs=1e-8)
    for info in payload["pairwise"].values():
        assert info["significant"] == (
            info["rank_difference"] >= payload["critical_difference"]
        )


def test_size_order_is_the_campaign_size_list():
    # Not a copy that has to be kept in sync by hand: a size added to the
    # campaign but missing here is silently dropped by the column filter
    # in build_matrices(), and the paper ranks a subset without saying so.
    assert rr.SIZE_ORDER == rss.SIZES


def test_build_matrices_hard_fails_when_a_size_is_wholly_unusable(tmp_path, monkeypatch):
    # The dangerous case: 'small' failed everywhere, so its column never
    # reaches the pivot. Without this guard the script would emit a
    # confident k=4 ranking, with a CD computed for k=4, and say nothing.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for i in range(3):
        _write_scaling(tmp_path, f"s{i}", "tiny", n=100, rmse=90.0 + i)
        _write_scaling(tmp_path, f"s{i}", "large", n=100, rmse=70.0 + i)
        _write_scaling(
            tmp_path, f"s{i}", "small", n=100, rmse=None,
            failure="RuntimeError: CUDA out of memory",
        )

    with pytest.raises(SystemExit) as exc:
        rr.build_matrices()

    assert "small" in str(exc.value)
    assert "0/3" in str(exc.value)


def test_build_matrices_reports_partial_failures_and_drops_the_series(
    tmp_path, monkeypatch, capsys
):
    # 'small' failed on one series only. That is a partial failure, not a
    # broken campaign: rank the rest, but say how many were lost and why.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for i in range(3):
        _write_scaling(tmp_path, f"s{i}", "tiny", n=100, rmse=90.0 + i)
        _write_scaling(tmp_path, f"s{i}", "large", n=100, rmse=70.0 + i)
    _write_scaling(tmp_path, "s0", "small", n=100, rmse=80.0)
    _write_scaling(tmp_path, "s1", "small", n=100, rmse=81.0)
    _write_scaling(
        tmp_path, "s2", "small", n=100, rmse=None, failure="RuntimeError: boom",
    )

    matrices = rr.build_matrices()
    out = capsys.readouterr().out

    assert list(matrices["short"].index) == ["s0", "s1"]
    # The reason lives in failed_models; discarding it turns a diagnosable
    # failure into an unexplained sample-size change.
    assert "[skip] s2 @ small: RuntimeError: boom" in out
    assert "[small] 1/3 runs unusable" in out
    assert "ranking sizes: ['tiny', 'small', 'large']" in out


def test_build_matrices_names_the_sizes_it_ranked(tmp_path, monkeypatch, capsys):
    # Unconditional, because a deliberately narrowed sweep is legitimate
    # and must still be visible in the report.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    _write_grid(tmp_path, ["s0", "s1"], ["tiny", "base", "large"])
    rr.build_matrices()
    assert "ranking sizes: ['tiny', 'base', 'large']" in capsys.readouterr().out


def test_build_matrices_rejects_unknown_sizes(tmp_path, monkeypatch):
    # A complete, valid column for a size the ranking has never heard of
    # would be dropped by the column filter without a word.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    _write_grid(tmp_path, ["s0", "s1"], ["tiny", "small", "large", "xlarge"])

    with pytest.raises(SystemExit, match="xlarge"):
        rr.build_matrices()


def test_build_matrices_fails_loudly_on_an_empty_directory(tmp_path, monkeypatch):
    # Previously a bare KeyError: 'stratum' from an empty DataFrame.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path / "does_not_exist")
    with pytest.raises(SystemExit, match="no usable scaling results"):
        rr.build_matrices()


def test_build_matrices_names_the_file_in_a_malformed_result(tmp_path, monkeypatch):
    # A bare KeyError: 'size' against 1,840 candidate files is not a
    # diagnosis.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    _write_grid(tmp_path, ["s0"], ["tiny", "small", "large"])
    with open(tmp_path / "broken_chronos_tiny.json", "w") as f:
        json.dump({"dataset": "broken", "metrics": {}}, f)

    with pytest.raises(SystemExit, match="broken_chronos_tiny.json"):
        rr.build_matrices()


def test_report_labels_each_stratum_correctly(tmp_path, monkeypatch, capsys):
    # Swapping the two labels leaves both JSON artifacts correct, so this
    # printed report is the only place the mistake can be caught. With
    # opposite-signed effects at the small end, a swap is a real
    # scientific error, not a cosmetic one.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    sizes = ["tiny", "small", "large"]
    _write_grid(tmp_path, [f"s{i}" for i in range(7)], sizes, n=100)
    _write_grid(tmp_path, [f"L{i}" for i in range(11)], sizes, n=900)

    matrices = rr.build_matrices()
    for stratum in ("short", "long"):
        rr.print_report(stratum, rr.analyse(matrices[stratum]))
    lines = capsys.readouterr().out.splitlines()

    short_line = next(line for line in lines if "SHORT (n<200)" in line)
    long_line = next(line for line in lines if "LONG (n>=200)" in line)
    # The series counts are what tie each label to its stratum.
    assert "7 series" in short_line
    assert "11 series" in long_line


def test_report_marks_pairwise_significance(capsys):
    payload = {
        "n_series": 20,
        "n_models": 3,
        "alpha": 0.05,
        "mean_ranks": {"tiny": 2.6, "small": 2.0, "large": 1.4},
        "friedman_statistic": 14.4,
        "friedman_p_value": 0.00075,
        "friedman_significant": True,
        "critical_difference": 0.7,
        "pairwise": {
            "tiny_vs_small": {"rank_difference": 0.6, "significant": False},
            "tiny_vs_large": {"rank_difference": 1.2, "significant": True},
            "small_vs_large": {"rank_difference": 0.6, "significant": False},
        },
    }

    rr.print_report("short", payload)
    out = capsys.readouterr().out

    marks = {
        line.split()[0]: line.rsplit("sig=", 1)[1].strip()
        for line in out.splitlines()
        if "diff=" in line
    }
    # An inverted mark reports a null result as a finding, or hides one.
    assert marks == {
        "tiny_vs_small": "no",
        "tiny_vs_large": "YES",
        "small_vs_large": "no",
    }
    assert "p = 0.0008" in out


def test_report_p_value_is_readable_against_alpha():
    # .3e forces a mental conversion every time it is compared to 0.05;
    # fixed point does not. e-notation is kept only where .4f would print
    # a meaningless 0.0000 - which is where the real data sits.
    assert rr._format_p(0.0518) == "0.0518"
    assert rr._format_p(2.6863261956430322e-21) == "2.686e-21"


# 8 series x 5 sizes, deliberately discordant: Friedman p = 0.0518 (NOT
# significant at 0.05) while small_vs_large's mean-rank gap of 2.25 clears
# the CD of 2.157. Friedman and Nemenyi are different statistics with no
# strict nesting - for k=5 the Nemenyi threshold is ~6.10/sqrt(N) against
# Friedman's ~6.89/sqrt(N) - so this window is real, not a rounding
# artifact. Rows are RMSE in SIZE_ORDER: tiny, mini, small, base, large.
_DISCORDANT = [
    [30.0, 50.0, 40.0, 10.0, 20.0],
    [30.0, 10.0, 50.0, 40.0, 20.0],
    [50.0, 40.0, 30.0, 20.0, 10.0],
    [50.0, 10.0, 40.0, 20.0, 30.0],
    [40.0, 10.0, 50.0, 30.0, 20.0],
    [30.0, 40.0, 50.0, 20.0, 10.0],
    [20.0, 40.0, 30.0, 50.0, 10.0],
    [20.0, 50.0, 40.0, 10.0, 30.0],
]


def test_report_suppresses_verdicts_when_omnibus_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for i, row in enumerate(_DISCORDANT):
        for size, rmse in zip(rr.SIZE_ORDER, row):
            _write_scaling(tmp_path, f"s{i}", size, n=100, rmse=rmse)

    payload = rr.analyse(rr.build_matrices()["short"])
    capsys.readouterr()  # discard build_matrices' own output

    # Guard the fixture itself: if this grid ever stops being discordant
    # the test below would pass vacuously.
    assert payload["friedman_significant"] is False
    assert any(info["significant"] for info in payload["pairwise"].values())

    rr.print_report("short", payload)
    out = capsys.readouterr().out

    assert "suppressed" in out
    # A warning printed above an intact table of YES verdicts invites
    # exactly the finding the warning forbids.
    assert "sig=YES" not in out
    assert "sig=no" not in out
    assert out.count("sig=n/a") == len(payload["pairwise"])


def test_main_writes_strict_json(tmp_path, monkeypatch):
    # Every size identical on every series makes the Friedman statistic
    # and p-value nan. json.dump would emit a bare `NaN` literal, which is
    # not valid JSON, so the artifact would fail in whatever reads it
    # rather than here, where the actual problem is.
    scaling, out_dir = tmp_path / "scaling", tmp_path / "ranking"
    scaling.mkdir()
    monkeypatch.setattr(rr, "SCALING_DIR", scaling)
    monkeypatch.setattr(rr, "OUT_DIR", out_dir)
    for i in range(6):
        for size in ["tiny", "small", "large"]:
            _write_scaling(scaling, f"s{i}", size, n=100, rmse=50.0)

    with pytest.raises(ValueError, match="Out of range float"):
        _run_main(monkeypatch)


def test_main_writes_provenance(tmp_path, monkeypatch):
    # These files ARE the scientific claim. Without provenance, a re-run
    # that produces only one stratum leaves the other stratum's file
    # looking every bit as current as the one just written.
    scaling, out_dir = tmp_path / "scaling", tmp_path / "ranking"
    scaling.mkdir()
    monkeypatch.setattr(rr, "SCALING_DIR", scaling)
    monkeypatch.setattr(rr, "OUT_DIR", out_dir)
    _write_grid(scaling, [f"s{i}" for i in range(6)], ["tiny", "small", "large"])

    _run_main(monkeypatch)

    with open(out_dir / "chronos_sizes_short.json") as f:
        payload = json.load(f)
    assert payload["n_input_files"] == 18
    stamp = datetime.fromisoformat(payload["generated_at"])
    assert stamp.utcoffset() == timedelta(0)


def test_main_alpha_flag_reaches_the_analysis(tmp_path, monkeypatch):
    # analyse()'s alpha was previously unreachable: no caller passed it.
    scaling, out_dir = tmp_path / "scaling", tmp_path / "ranking"
    scaling.mkdir()
    monkeypatch.setattr(rr, "SCALING_DIR", scaling)
    monkeypatch.setattr(rr, "OUT_DIR", out_dir)
    _write_grid(scaling, [f"s{i}" for i in range(6)], ["tiny", "small", "large"])

    _run_main(monkeypatch, "--alpha", "0.01")

    with open(out_dir / "chronos_sizes_short.json") as f:
        payload = json.load(f)
    assert payload["alpha"] == 0.01
    # Echoing alpha into the payload is not the same as using it.
    assert payload["critical_difference"] == pytest.approx(
        nemenyi_critical_difference(3, 6, 0.01)
    )
    assert payload["critical_difference"] > nemenyi_critical_difference(3, 6, 0.05)


def test_main_does_not_claim_success_when_nothing_was_written(
    tmp_path, monkeypatch, capsys
):
    scaling, out_dir = tmp_path / "scaling", tmp_path / "ranking"
    scaling.mkdir()
    monkeypatch.setattr(rr, "SCALING_DIR", scaling)
    monkeypatch.setattr(rr, "OUT_DIR", out_dir)
    monkeypatch.setattr(rr, "build_matrices", dict)

    _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert "Wrote" not in out
    assert "stale" in out
    assert not list(out_dir.glob("*.json"))
