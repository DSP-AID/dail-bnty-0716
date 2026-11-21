"""Corpus loading for the standalone V8-arm probe."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["split", "prompt_id", "model", "text"]
MIN_GROUP = 6


def load_corpus(path: str, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Read the shipped passage table (.csv or .csv.gz) into a fold-ready frame.

    One row per passage; `group` is the prompt block (split + prompt_id), which is the unit
    that must never straddle a CV fold. Rows with no text or no label are dropped, and blocks
    smaller than `min_group` passages are dropped (the shipped table holds 18- and 12-passage
    blocks, so this only guards against a hand-trimmed sample).
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.dropna(subset=["model", "text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["group"] = df["split"].astype(str) + "::" + df["prompt_id"].astype(str)
    sizes = df.groupby("group")["text"].transform("size")
    df = df[sizes >= min_group]

    df = df.sort_values(["group", "model"], kind="mergesort").reset_index(drop=True)
    df["label"] = df["model"].astype(str)
    return df[["id", "split", "prompt_id", "group", "model", "label", "text"]]


def classes_of(df: pd.DataFrame) -> list[str]:
    """Deterministic class list (alphabetical model names)."""
    return sorted(df["label"].unique().tolist())
