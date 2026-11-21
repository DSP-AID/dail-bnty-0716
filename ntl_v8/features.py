"""Text style statistics (V7 base numeric block) and TRACE rank-transition features (V8 arm).

Everything in this module is label-free: a row's numbers depend only on the text of that
row and, for TRACE, on the texts of its prompt-group siblings. Prompt groups never straddle
a fold (see oof.make_folds), so sibling text is legitimately available at predict time.
"""

from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd
from scipy.stats import rankdata

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SENT_SPLIT_RE = re.compile(r"[.!?]+(?=\s|$)")
MD_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)]|#{1,6})\s")
MD_INLINE_RE = re.compile(r"\*\*|__|`|~~")
DASH_SEMI_RE = re.compile(r"[\u2013\u2014;]")
QUOTE_RE = re.compile(r"[\u201c\u201d\"()\u2026]")
DIGIT_TOKEN_RE = re.compile(r"\d")

_PUNCT_CHARS = set(string.punctuation) | set("\u2013\u2014\u2018\u2019\u201c\u201d\u2026")

# Fixed, hand-written function-word list: no corpus fitting, so no train/val leakage.
FUNCTION_WORDS = frozenset(
    """a an the of to and in is are was were be been being it its this that these those for as
    with on at by from or nor but if then than so such can will would could should may might must
    have has had do does did not no all any some more most other another each every both few many
    much own same too very just only own about above after before between during through under over
    again further here there when where why how who whom which what i me my we our us you your he
    him his she her they them their us""".split()
)

# Style views, in canonical order. TRACE transitions are differences between consecutive
# columns of this list, so the order is part of the arm's definition and must not change.
VIEWS = [
    "n_words",
    "mean_word_len",
    "function_share",
    "type_token_ratio",
    "repeat_trigram_share",
    "punct_per_100w",
    "comma_per_100w",
    "sent_per_100w",
    "avg_sentence_words",
    "digit_token_share",
    "markdown_per_100w",
    "dash_semicolon_per_100w",
    "caps_word_share",
    "quote_paren_per_100w",
]

STATS_COLUMNS = VIEWS  # the raw style block doubles as the TRACE view list


def text_stats(text: str) -> dict[str, float]:
    """Label-free stylometry for one passage. Never raises, never NaN."""
    text = "" if text is None else str(text)
    words = WORD_RE.findall(text)
    n_words = len(words)
    lower = [w.lower() for w in words]

    def per_100w(count: float) -> float:
        return 100.0 * count / n_words if n_words else 0.0

    letters = sum(len(w) for w in words)
    mean_word_len = letters / n_words if n_words else 0.0
    function_share = (sum(1 for w in lower if w in FUNCTION_WORDS) / n_words) if n_words else 0.0
    distinct = len(set(lower))
    ttr = distinct / n_words if n_words else 0.0

    trigrams = [tuple(lower[i : i + 3]) for i in range(max(0, n_words - 2))]
    repeat_trigram_share = (
        (len(trigrams) - len(set(trigrams))) / len(trigrams) if trigrams else 0.0
    )

    punct_total = sum(1 for ch in text if ch in _PUNCT_CHARS)
    commas = text.count(",")
    sentences = [s for s in SENT_SPLIT_RE.split(text) if WORD_RE.search(s or "")]
    n_sents = len(sentences)
    avg_sentence_words = (n_words / n_sents) if n_sents else float(n_words > 0) * n_words
    digits = sum(1 for w in words if DIGIT_TOKEN_RE.search(w))
    md = len(MD_LINE_RE.findall(text)) + len(MD_INLINE_RE.findall(text))
    dash_semi = len(DASH_SEMI_RE.findall(text))
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    quotes = len(QUOTE_RE.findall(text))

    return {
        "n_words": float(n_words),
        "mean_word_len": mean_word_len,
        "function_share": function_share,
        "type_token_ratio": ttr,
        "repeat_trigram_share": repeat_trigram_share,
        "punct_per_100w": per_100w(punct_total),
        "comma_per_100w": per_100w(commas),
        "sent_per_100w": per_100w(n_sents),
        "avg_sentence_words": avg_sentence_words,
        "digit_token_share": (digits / n_words) if n_words else 0.0,
        "markdown_per_100w": per_100w(md),
        "dash_semicolon_per_100w": per_100w(dash_semi),
        "caps_word_share": (caps / n_words) if n_words else 0.0,
        "quote_paren_per_100w": per_100w(quotes),
    }


def style_block(texts: pd.Series) -> pd.DataFrame:
    """(n_rows, len(VIEWS)) raw style statistics, column order == VIEWS."""
    block = pd.DataFrame([text_stats(t) for t in texts])
    return block.reindex(columns=STATS_COLUMNS).astype(float).fillna(0.0)


def within_group_ranks(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank of each value inside its own group, scaled to [0, 1].

    Ties are averaged (scipy convention), a single-member group ranks 0.5, and a constant
    group ranks at the midpoint. Pure function of (values, groups).
    """
    out = np.full(len(values), 0.5, dtype=float)
    for _, idx in pd.Series(range(len(groups))).groupby(groups, sort=False):
        pos = np.asarray(idx, dtype=int)
        vals = np.asarray(values[pos], dtype=float)
        if not np.isfinite(vals).all():
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        m = len(pos)
        if m == 1:
            out[pos] = 0.5
            continue
        ranks = rankdata(vals, method="average")
        spread = ranks.max() - ranks.min()
        out[pos] = 0.5 if spread == 0 else (ranks - ranks.min()) / spread
    return out


def trace_matrix(texts: pd.Series, groups: np.ndarray) -> np.ndarray:
    """TRACE rank-transition features: within-group view ranks + step transitions.

    Columns: len(VIEWS) ranks, then len(VIEWS)-1 transitions (rank of view i+1 minus rank of
    view i), then 3 summaries of the transition sequence (max |step|, mean |step|, sign flips).
    """
    block = style_block(texts)
    ranks = np.column_stack(
        [within_group_ranks(block[c].to_numpy(), groups) for c in VIEWS]
    )
    steps = np.diff(ranks, axis=1)
    summaries = np.column_stack(
        [
            np.abs(steps).max(axis=1),
            np.abs(steps).mean(axis=1),
            (np.diff(np.signbit(steps), axis=1) != 0).sum(axis=1).astype(float),
        ]
    )
    return np.hstack([ranks, steps, summaries])


def trace_columns() -> list[str]:
    """Feature names matching trace_matrix's column order."""
    ranks = [f"rk_{v}" for v in VIEWS]
    steps = [f"tr_{a}__to__{b}" for a, b in zip(VIEWS[:-1], VIEWS[1:])]
    return ranks + steps + ["tr_max_abs", "tr_mean_abs", "tr_sign_flips"]
