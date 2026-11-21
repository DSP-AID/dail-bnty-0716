"""Fold-safe feature assembly and the two arms (V7 control, V8-TRACE)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import style_block, trace_matrix

ARMS = ("V7-control", "V8-TRACE")

WORD_KW = dict(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_features=40_000,
    strip_accents="unicode",
    dtype=np.float32,
)
CHAR_KW = dict(
    analyzer="char_wb",
    ngram_range=(2, 5),
    sublinear_tf=True,
    min_df=8,
    max_features=40_000,
    dtype=np.float32,
)


def make_model() -> LogisticRegression:
    """Identical classifier for both arms; only the feature block differs."""
    return LogisticRegression(
        C=4.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=0,
    )


def base_matrices(train_df, eval_df) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """V7 feature block: TF-IDF(word) + TF-IDF(char_wb) + standardized style stats.

    Every transform is fitted on the TRAIN rows only.
    """
    tr_text = train_df["text"].to_numpy()
    ev_text = eval_df["text"].to_numpy()

    word = TfidfVectorizer(**WORD_KW)
    char = TfidfVectorizer(**CHAR_KW)
    block_tr, block_ev = style_block(train_df["text"]), style_block(eval_df["text"])
    scaler = StandardScaler().fit(block_tr)
    parts_tr = [word.fit_transform(tr_text), char.fit_transform(tr_text),
                sp.csr_matrix(scaler.transform(block_tr))]
    parts_ev = [word.transform(ev_text), char.transform(ev_text),
                sp.csr_matrix(scaler.transform(block_ev))]
    return sp.hstack(parts_tr).tocsr(), sp.hstack(parts_ev).tocsr()


def trace_blocks(train_df, eval_df) -> dict:
    """TRACE block for one fold side-pair, in scaled and unscaled form.

    Ranks are computed separately on the train block and on the eval block: each prompt group
    lives entirely on one side of the split, so a row's sibling set is complete either way.
    """
    trace_tr = trace_matrix(train_df["text"], train_df["group"].to_numpy())
    trace_ev = trace_matrix(eval_df["text"], eval_df["group"].to_numpy())
    scaler = StandardScaler().fit(trace_tr)  # fitted on the train side only
    scaled_tr, scaled_ev = scaler.transform(trace_tr), scaler.transform(trace_ev)
    return {
        "scaled": (sp.csr_matrix(np.asarray(scaled_tr, dtype=np.float32)),
                   sp.csr_matrix(np.asarray(scaled_ev, dtype=np.float32))),
        "raw": (sp.csr_matrix(np.asarray(trace_tr, dtype=np.float32)),
                sp.csr_matrix(np.asarray(trace_ev, dtype=np.float32))),
    }


def assemble(base, trace, arm: str, scale_trace: bool = True):
    """Stack the shared V7 base with the TRACE block (arm == 'V8-TRACE')."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    x_tr, x_ev = base
    if arm == "V8-TRACE":
        tr, ev = trace["scaled" if scale_trace else "raw"]
        x_tr, x_ev = sp.hstack([x_tr, tr]), sp.hstack([x_ev, ev])
    return x_tr.tocsr(), x_ev.tocsr()


def build_matrices(train_df, eval_df, arm: str, scale_trace: bool = True):
    """Single-fold convenience wrapper (the OOF harness shares pieces across arms instead)."""
    return assemble(
        base_matrices(train_df, eval_df), trace_blocks(train_df, eval_df), arm, scale_trace
    )


def fit_predict(train_df, eval_df, arm: str, classes: list[str], scale_trace: bool = True):
    """Train the arm on `train_df`, return class probabilities for `eval_df` in `classes` order."""
    x_tr, x_ev = build_matrices(train_df, eval_df, arm, scale_trace=scale_trace)
    y = np.searchsorted(np.asarray(classes), train_df["label"].to_numpy())
    model = make_model().fit(x_tr, y)
    proba = model.predict_proba(x_ev)
    if proba.shape[1] < len(classes):  # a class unseen in this fold's training rows
        pad = np.zeros((proba.shape[0], len(classes)), dtype=proba.dtype)
        pad[:, : proba.shape[1]] = proba
        proba = pad
    return proba


def damped_calibration(proba, alpha: float = 0.15):
    """Damped calibration arm: shrink probabilities toward the uniform mixture.

    p' = (1 - alpha) * p + alpha / K, renormalised (probability-space damping, a stand-in for
    temperature > 1). It is a strictly increasing map per row, so the argmax - and therefore
    balanced accuracy - cannot move; the gate it can move is log-loss.
    """
    proba = np.asarray(proba, dtype=float)
    out = (1.0 - alpha) * proba + alpha / proba.shape[1]
    return out / out.sum(axis=1, keepdims=True)
