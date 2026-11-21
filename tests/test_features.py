"""Label-freeness and determinism of the TRACE rank-transition features."""

import numpy as np
import pandas as pd

from ntl_v8.features import (
    STATS_COLUMNS,
    VIEWS,
    style_block,
    text_stats,
    trace_columns,
    trace_matrix,
    within_group_ranks,
)


def _toy_frame():
    texts = [
        "Alpha beta gamma. Delta epsilon!" * 3,
        "Zeta eta, theta iota; kappa lambda mu nu." * 2,
        "xi omicron pi rho-sigma tau phi chi psi omega" * 4,
        "1 22 333 four FIVE **bold** - dash; quote “x”",
        "Same words again. Same words again. Same words again.",
        "short one here",
    ]
    groups = np.array(["a"] * 3 + ["b"] * 3)
    return pd.DataFrame({"text": texts, "group": groups})


def test_text_stats_is_finite_and_complete():
    for text in ["", "   ", "!!!", "a b c", "1234567890", "\u2014\u2014\u2014", None, 42]:
        stats = text_stats(text)
        assert list(stats) == VIEWS
        assert all(np.isfinite(v) for v in stats.values()), text


def test_style_block_columns_follow_views():
    block = style_block(_toy_frame()["text"])
    assert list(block.columns) == STATS_COLUMNS == VIEWS
    assert np.isfinite(block.to_numpy()).all()


def test_within_group_ranks_bounds_and_ties():
    values = np.array([1.0, 3.0, 2.0, 5.0, 5.0, 5.0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    ranks = within_group_ranks(values, groups)
    assert (ranks >= 0).all() and (ranks <= 1).all()
    assert ranks[0] == 0.0 and ranks[1] == 1.0 and ranks[2] == 0.5
    assert np.allclose(ranks[3:], 0.5)  # constant group -> midpoint
    assert within_group_ranks(np.array([7.0]), np.array(["only"]))[0] == 0.5


def test_within_group_ranks_are_monotone_within_a_group():
    rng = np.random.default_rng(0)
    values = rng.normal(size=60)
    groups = np.repeat(np.arange(10), 6)
    ranks = within_group_ranks(values, groups)
    for g in np.unique(groups):
        sel = groups == g
        order = np.argsort(values[sel], kind="mergesort")
        assert np.all(np.diff(ranks[sel][order]) >= -1e-12)


def test_ranks_ignore_rows_outside_their_group():
    """Fold-safety: a row's rank depends only on its own group, so computing it on the
    train block or on the validation block gives identical numbers."""
    df = _toy_frame()
    full = within_group_ranks(df["text"].str.len().to_numpy(), df["group"].to_numpy())
    sub = df[df["group"] == "a"]
    part = within_group_ranks(sub["text"].str.len().to_numpy(), sub["group"].to_numpy())
    assert np.allclose(full[:3], part)


def test_trace_is_permutation_equivariant_and_label_free():
    df = _toy_frame()
    base = trace_matrix(df["text"], df["group"].to_numpy())
    perm = np.array([3, 4, 5, 0, 1, 2])  # group shuffle
    shuffled = trace_matrix(df["text"].iloc[perm], df["group"].to_numpy()[perm])
    assert np.allclose(base[perm], shuffled)
    within = np.array([1, 0, 2, 4, 3, 5])  # shuffle inside groups only
    rel = trace_matrix(df["text"].iloc[within], df["group"].to_numpy()[within])
    assert np.allclose(base[within], rel)


def test_trace_width_matches_names_and_has_no_nan():
    df = _toy_frame()
    m = trace_matrix(df["text"], df["group"].to_numpy())
    names = trace_columns()
    assert m.shape == (len(df), len(names))
    assert len(set(names)) == len(names) == 2 * len(VIEWS) - 1 + 3
    assert np.isfinite(m).all()


def test_trace_ranks_are_group_local_and_normalized():
    """Ranks are computed inside each prompt block: spread views span [0, 1], a view that is
    constant inside the block carries no information and ranks at the midpoint."""
    df = _toy_frame()
    block = style_block(df["text"])
    ranks = trace_matrix(df["text"], df["group"].to_numpy())[:, : len(VIEWS)]
    for g in ["a", "b"]:
        sel = df["group"].to_numpy() == g
        for j, col in enumerate(VIEWS):
            vals, sub = block[col].to_numpy()[sel], ranks[sel][:, j]
            if np.ptp(vals) == 0:
                assert np.allclose(sub, 0.5), (g, col)
            else:
                assert np.isclose(sub.min(), 0.0) and np.isclose(sub.max(), 1.0), (g, col)
                order = np.argsort(vals, kind="mergesort")
                assert np.all(np.diff(sub[order]) >= -1e-12), (g, col)


def test_trace_matrix_handles_singletons_and_empty_rows():
    texts = pd.Series(["only row", "", None])
    groups = np.array(["x", "y", "y"])
    m = trace_matrix(texts, groups)
    assert np.isfinite(m).all()
    assert m.shape == (3, len(trace_columns()))
