"""k=2 prompt-grouped OOF harness, gates and bootstrap for the V8-arm probe."""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from .features import trace_columns
from .model import ARMS, assemble, base_matrices, damped_calibration, make_model, trace_blocks


def make_folds(df, n_splits: int = 2, seed: int = 0, frac: float = 1.0):
    """Prompt-group folds: whole prompt blocks, round-robin over a seeded group order.

    `frac` subsamples the group list (never the rows), so every run still holds its validation
    fold to complete prompt blocks - random row splits would put the same prompt in train and
    validation and badly overstate accuracy.
    """
    groups = sorted(df["group"].unique())
    rng = np.random.default_rng(seed)
    take = max(n_splits, math.ceil(frac * len(groups)))
    chosen = [groups[i] for i in rng.permutation(len(groups))[:take]]
    fold_of = {g: i % n_splits for i, g in enumerate(chosen)}
    assign = df["group"].map(fold_of)

    folds = []
    for f in range(n_splits):
        in_fold = assign.notna().to_numpy()
        val = np.where(in_fold & (assign.to_numpy() == f))
        train = np.where(in_fold & (assign.to_numpy() != f))
        folds.append((train[0], val[0]))
    return folds, chosen


def oof_predict(df, folds, arms, classes: list[str], seed: int = 0, scale_trace: bool = True):
    """OOF pass for one or more arms over the same folds; feature fitting is shared.

    Returns {arm: {"rows", "truth", "proba", "per_fold", "seconds", + pooled metrics}}.
    """
    if isinstance(arms, str):
        arms = (arms,)
    acc = {
        arm: {
            "rows": np.zeros(len(df), dtype=bool),
            "truth": np.full(len(df), -1, dtype=int),
            "proba": np.zeros((len(df), len(classes)), dtype=float),
            "per_fold": [],
        }
        for arm in arms
    }
    t0 = time.time()
    for train_idx, val_idx in folds:
        train_df, eval_df = df.iloc[train_idx], df.iloc[val_idx]
        base = base_matrices(train_df, eval_df)  # fitted once per fold, reused by every arm
        trace = trace_blocks(train_df, eval_df) if any(a == "V8-TRACE" for a in arms) else None
        y_val = np.searchsorted(np.asarray(classes), df["label"].to_numpy()[val_idx])
        for arm in arms:
            x_tr, x_ev = assemble(base, trace, arm, scale_trace=scale_trace)
            y_tr = np.searchsorted(np.asarray(classes), train_df["label"].to_numpy())
            model = make_model().fit(x_tr, y_tr)
            p = model.predict_proba(x_ev)
            if p.shape[1] < len(classes):
                pad = np.zeros((p.shape[0], len(classes)), dtype=p.dtype)
                pad[:, : p.shape[1]] = p
                p = pad
            out = acc[arm]
            out["proba"][val_idx] = p
            out["truth"][val_idx] = y_val
            out["rows"][val_idx] = True
            out["per_fold"].append(
                {
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_val, p.argmax(axis=1))),
                }
            )
    for arm, out in acc.items():
        rows = out["rows"]
        out["seconds"] = round(time.time() - t0, 2)
        out.update(_metrics(out["truth"][rows], out["proba"][rows], classes))
    return acc


def _metrics(truth, proba, classes):
    pred = proba.argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", labels=list(range(len(classes))))),
        "log_loss": float(log_loss(truth, proba, labels=list(range(len(classes))))),
    }


def bootstrap_group_delta(
    df, rows, arm_a: str, arm_b: str, n_boot: int = 400, seed: int = 7
):
    """Prompt-group bootstrap 95% CI of the OOF balanced-accuracy delta (A - B)."""
    covered = df.loc[rows]
    y_all = np.searchsorted(np.asarray(sorted(covered["label"].unique())), covered["label"].to_numpy())
    pred_a = arm_a.argmax(axis=1)
    pred_b = arm_b.argmax(axis=1)
    by_group = [np.where((covered["group"] == g).to_numpy())[0] for g in sorted(covered["group"].unique())]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = np.concatenate([by_group[i] for i in rng.integers(0, len(by_group), len(by_group))])
        yb = y_all[pick]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(
            balanced_accuracy_score(yb, pred_a[pick]) - balanced_accuracy_score(yb, pred_b[pick])
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": int(len(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "share_positive": float(np.mean(np.asarray(deltas) > 0)),
    }


def run_gates(
    df,
    classes: list[str],
    reps: int = 5,
    subsample: float = 0.8,
    n_splits: int = 2,
    seed: int = 0,
    variants: bool = True,
    alpha: float = 0.15,
    n_boot: int = 400,
    log=print,
) -> dict:
    """Full gate battery. Numbers are honest deltas, not a leaderboard chase."""
    report: dict = {"config": {"reps": reps, "subsample": subsample, "n_splits": n_splits,
                               "seed": seed, "damping_alpha": alpha, "bootstrap_draws": n_boot}}

    sizes = df.groupby("group").size()
    counts = df["label"].value_counts()
    report["gate0_data"] = {
        "rows": int(len(df)),
        "prompt_groups": int(sizes.shape[0]),
        "classes": len(classes),
        "group_size_min": int(sizes.min()),
        "group_size_max": int(sizes.max()),
        "group_size_mode": sorted(sizes.value_counts().to_dict().items()),
        "per_class_min": int(counts.min()),
        "per_class_max": int(counts.max()),
        "trace_feature_columns": len(trace_columns()),
    }

    folds, chosen = make_folds(df, n_splits=n_splits, seed=seed, frac=1.0)
    leak = max(
        len(set(df["group"].iloc[a]) & set(df["group"].iloc[b])) for a, b in folds
    )
    covered_rows = sum(len(v) for _, v in folds)
    report["gate1_folds"] = {
        "n_splits": n_splits,
        "groups_in_folds": len(chosen),
        "max_groups_shared_between_train_and_val": leak,
        "rows_covered_by_oof": int(covered_rows),
        "rows_total": int(len(df)),
    }

    oof = oof_predict(df, folds, ARMS, classes, seed=seed)
    control, trace = oof["V7-control"], oof["V8-TRACE"]
    report["gate2_primary_k2"] = {
        "n_splits": n_splits,
        "subsample": 1.0,
        "control": {k: control[k] for k in ("balanced_accuracy", "macro_f1", "log_loss", "seconds")},
        "arm": {k: trace[k] for k in ("balanced_accuracy", "macro_f1", "log_loss", "seconds")},
        "per_fold_control_ba": [f["balanced_accuracy"] for f in control["per_fold"]],
        "per_fold_arm_ba": [f["balanced_accuracy"] for f in trace["per_fold"]],
        "delta_balanced_accuracy": trace["balanced_accuracy"] - control["balanced_accuracy"],
        "delta_macro_f1": trace["macro_f1"] - control["macro_f1"],
        "delta_log_loss": trace["log_loss"] - control["log_loss"],
    }

    report["gate3_subsampled"] = {"n_splits": n_splits, "subsample": subsample, "runs": []}
    for r in range(reps):
        sub_folds, sub_chosen = make_folds(df, n_splits=n_splits, seed=seed + 1 + r, frac=subsample)
        entry = {"seed": seed + 1 + r, "groups_used": len(sub_chosen)}
        res = oof_predict(df, sub_folds, ARMS, classes, seed=seed)
        entry["control_ba"] = res["V7-control"]["balanced_accuracy"]
        entry["arm_ba"] = res["V8-TRACE"]["balanced_accuracy"]
        entry["delta_balanced_accuracy"] = (
            res["V8-TRACE"]["balanced_accuracy"] - res["V7-control"]["balanced_accuracy"]
        )
        report["gate3_subsampled"]["runs"].append(entry)
    d = [e["delta_balanced_accuracy"] for e in report["gate3_subsampled"]["runs"]]
    report["gate3_subsampled"]["summary"] = {
        "n_runs": len(d),
        "mean_delta": float(np.mean(d)),
        "min_delta": float(np.min(d)),
        "max_delta": float(np.max(d)),
        "std_delta": float(np.std(d, ddof=1)) if len(d) > 1 else None,
        "positive_runs": int(sum(x > 0 for x in d)),
    }

    report["gate4_bootstrap"] = bootstrap_group_delta(
        df, control["rows"], trace["proba"], control["proba"], n_boot=n_boot
    )

    damped = damped_calibration(control["proba"][control["rows"]], alpha=alpha)
    base = _metrics(control["truth"][control["rows"]], control["proba"][control["rows"]], classes)
    aft = _metrics(control["truth"][control["rows"]], damped, classes)
    report["gate5_damped_calibration"] = {
        "alpha": alpha,
        "log_loss_before": base["log_loss"],
        "log_loss_after": aft["log_loss"],
        "delta_log_loss": aft["log_loss"] - base["log_loss"],
        "delta_balanced_accuracy": aft["balanced_accuracy"] - base["balanced_accuracy"],
    }

    if variants:
        unscaled = oof_predict(df, folds, "V8-TRACE", classes, seed=seed, scale_trace=False)["V8-TRACE"]
        report["gate6_trace_unscaled"] = {
            "balanced_accuracy": unscaled["balanced_accuracy"],
            "log_loss": unscaled["log_loss"],
            "delta_balanced_vs_control": unscaled["balanced_accuracy"] - control["balanced_accuracy"],
            "per_fold_ba": [f["balanced_accuracy"] for f in unscaled["per_fold"]],
            "seconds": unscaled["seconds"],
        }

    verdict(report, log=log)
    return report


def verdict(report: dict, log=print) -> str:
    """Plain-English pass/fail against the stated gate, printed and returned."""
    g2 = report["gate2_primary_k2"]
    g3 = report["gate3_subsampled"]["summary"]
    g4 = report["gate4_bootstrap"]
    delta = g2["delta_balanced_accuracy"]
    positive_share = g3["positive_runs"] / g3["n_runs"] if g3["n_runs"] else 0.0
    passed = bool(delta > 0 and g4["ci95_low"] > 0 and positive_share >= 0.8)
    lines = [
        f"GATE k=2 prompt-grouped OOF delta (V8-TRACE - V7-control): {delta:+.4f} balanced accuracy",
        f"  per-fold control {['%.4f' % v for v in g2['per_fold_control_ba']]}",
        f"  per-fold arm     {['%.4f' % v for v in g2['per_fold_arm_ba']]}",
        f"GATE subsampled stability ({g3['n_runs']} runs @ {report['gate3_subsampled']['subsample']:.0%} of prompt groups): "
        f"mean {g3['mean_delta']:+.4f}, min {g3['min_delta']:+.4f}, max {g3['max_delta']:+.4f}, "
        f"positive {g3['positive_runs']}/{g3['n_runs']}",
        f"GATE prompt-group bootstrap 95% CI of delta: [{g4['ci95_low']:+.4f}, {g4['ci95_high']:+.4f}] "
        f"(mean {g4['mean_delta']:+.4f}, positive in {g4['share_positive']:.1%} of draws)",
        f"GATE log-loss delta: {g2['delta_log_loss']:+.4f}",
        f"VERDICT: {'PASS' if passed else 'FAIL'} - the arm must beat the control on the k=2 OOF "
        "delta, with a bootstrap CI above zero and positive in >=80% of subsamples",
    ]
    report["verdict"] = {"pass": passed, "delta_balanced_accuracy": delta,
                         "gate_ci_low": g4["ci95_low"], "positive_share": positive_share}
    log("\n".join(lines))
    return "\n".join(lines)
