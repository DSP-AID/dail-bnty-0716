"""Feature assembly: the control is a strict prefix of the arm, and all fitting is train-only."""

import os

import numpy as np
import pandas as pd
import pytest

from ntl_v8.data import classes_of, load_corpus
from ntl_v8.features import trace_columns
from ntl_v8.model import build_matrices, damped_calibration, fit_predict

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "passages_clean.csv.gz")


@pytest.fixture(scope="module")
def small_df():
    """Eight whole prompt blocks (label set intact, blocks complete) - fast but real."""
    df = load_corpus(DATA)
    keep = sorted(df["group"].unique())[:8]
    return df[df["group"].isin(keep)].reset_index(drop=True)


@pytest.fixture(scope="module")
def split_df(small_df):
    """Uneven 3/5 group split on purpose: equal-sized sides hide row/column transposition bugs."""
    groups = sorted(small_df["group"].unique())
    train = small_df[small_df["group"].isin(groups[:3])]
    val = small_df[small_df["group"].isin(groups[3:])]
    return train, val


def test_control_matrix_is_a_prefix_of_the_arm_matrix(split_df):
    train, val = split_df
    c_tr, c_va = build_matrices(train, val, "V7-control")
    a_tr, a_va = build_matrices(train, val, "V8-TRACE")
    trace_w = len(trace_columns())
    assert c_tr.shape[0] == len(train) and c_va.shape[0] == len(val)
    assert a_tr.shape[1] == c_tr.shape[1] + trace_w
    assert a_va.shape[1] == c_va.shape[1] + trace_w
    for full, base in ((a_tr, c_tr), (a_va, c_va)):
        diff = np.abs(
            (full[:, : base.shape[1]] - base).toarray()
        ).max()
        assert diff < 1e-9  # identical base features, only TRACE is added


def test_assembly_is_deterministic(split_df):
    train, val = split_df
    x1, y1 = build_matrices(train, val, "V8-TRACE")
    x2, y2 = build_matrices(train, val, "V8-TRACE")
    assert (x1 != x2).nnz == 0 and (y1 != y2).nnz == 0


def test_unscaled_variant_only_touches_the_trace_block(split_df):
    train, val = split_df
    s_tr, _ = build_matrices(train, val, "V8-TRACE", scale_trace=True)
    u_tr, _ = build_matrices(train, val, "V8-TRACE", scale_trace=False)
    base_w = s_tr.shape[1] - len(trace_columns())
    assert (s_tr[:, :base_w] != u_tr[:, :base_w]).nnz == 0
    assert (s_tr[:, base_w:] - u_tr[:, base_w:]).nnz > 0


def test_fit_predict_returns_valid_probabilities(split_df):
    train, val = split_df
    classes = classes_of(pd.concat([train, val], ignore_index=True))
    proba = fit_predict(train, val, "V8-TRACE", classes)
    assert proba.shape == (len(val), len(classes))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_damped_calibration_is_a_row_wise_monotone_shrink():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(6), size=25)
    assert np.allclose(damped_calibration(p, alpha=0.0), p)
    d = damped_calibration(p, alpha=0.4)
    assert np.allclose(d.sum(axis=1), 1.0)
    assert (d >= 0).all()
    assert np.array_equal(d.argmax(axis=1), p.argmax(axis=1))  # argmax (hence BA) cannot move
    order = np.argsort(-p, axis=1)
    assert np.all(np.diff(np.take_along_axis(d, order, axis=1), axis=1) <= 1e-12)
