"""Fold integrity (the leak gate), determinism, and an end-to-end smoke of the gate battery."""

import os

import numpy as np
import pytest

from ntl_v8.data import classes_of, load_corpus
from ntl_v8.oof import make_folds, oof_predict, run_gates

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "passages_clean.csv.gz")


@pytest.fixture(scope="module")
def df():
    return load_corpus(DATA)


def test_full_folds_partition_prompt_groups_exactly_once(df):
    folds, chosen = make_folds(df, n_splits=2, seed=0, frac=1.0)
    groups = set(df["group"].unique())
    assert set(chosen) == groups
    seen_val = []
    for train_idx, val_idx in folds:
        tr, va = set(df["group"].iloc[train_idx]), set(df["group"].iloc[val_idx])
        assert not (tr & va)  # no prompt block may straddle a fold
        seen_val += list(va)
    assert sorted(seen_val) == sorted(groups)
    covered = np.concatenate([v for _, v in folds])
    assert len(covered) == len(np.unique(covered)) == len(df)


def test_folds_are_deterministic_and_seeded_runs_differ(df):
    a, ga = make_folds(df, 2, seed=0)
    b, gb = make_folds(df, 2, seed=0)
    assert ga == gb
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    c, gc = make_folds(df, 2, seed=1)
    assert gc != gb


def test_subsampled_folds_stay_whole_and_smaller(df):
    full, gfull = make_folds(df, 2, seed=5, frac=1.0)
    sub, gsub = make_folds(df, 2, seed=5, frac=0.8)
    assert len(gsub) < len(gfull)
    assert set(gsub) <= set(gfull)
    for train_idx, val_idx in sub:
        tr, va = set(df["group"].iloc[train_idx]), set(df["group"].iloc[val_idx])
        assert not (tr & va)
        assert tr | va == set(gsub)
    assert sum(len(v) for _, v in sub) < sum(len(v) for _, v in full)


def test_oof_predict_covers_every_row_once(df):
    small_groups = sorted(df["group"].unique())[:10]
    small = df[df["group"].isin(small_groups)].reset_index(drop=True)
    folds, _ = make_folds(small, 2, seed=0)
    res = oof_predict(small, folds, "V7-control", classes_of(small))["V7-control"]
    assert res["rows"].all()
    assert (res["truth"] >= 0).all()
    assert len(res["per_fold"]) == 2
    both = oof_predict(small, folds, ("V7-control", "V8-TRACE"), classes_of(small))
    assert set(both) == {"V7-control", "V8-TRACE"}
    assert both["V8-TRACE"]["proba"].shape[1] == both["V7-control"]["proba"].shape[1]


def test_run_gates_smoke_on_a_slice(df):
    small_groups = sorted(df["group"].unique())[:10]
    small = df[df["group"].isin(small_groups)].reset_index(drop=True)
    report = run_gates(small, classes_of(small), reps=1, subsample=0.8, n_boot=20,
                       variants=False, log=lambda *_: None)
    for key in ("gate0_data", "gate1_folds", "gate2_primary_k2", "gate3_subsampled",
                "gate4_bootstrap", "gate5_damped_calibration", "verdict"):
        assert key in report
    assert report["gate1_folds"]["max_groups_shared_between_train_and_val"] == 0
    assert report["gate1_folds"]["rows_covered_by_oof"] == len(small)
    delta = report["gate2_primary_k2"]["delta_balanced_accuracy"]
    assert np.isfinite(delta) and abs(delta) < 1.0
    assert set(report["verdict"]) >= {"pass", "delta_balanced_accuracy", "gate_ci_low"}
    assert report["gate3_subsampled"]["summary"]["n_runs"] == 1
