# V8 arm isolation — **TRACE rank-transition features** (DAiL bounty `bnty_0716`)

One V8 arm, shipped on its own and measured against the V7 control with a **k=2 prompt-grouped OOF**
check on public data. Everything needed to re-run the check is in this repo: no API keys, no downloads.

## Headline (shipped evidence: [`evidence/oof_report.json`](evidence/oof_report.json))

| gate | number |
|---|---|
| **2 — k=2 prompt-grouped OOF balanced accuracy** | V7 control **0.2786**, V8-TRACE **0.3013**, **delta +0.02268** |
| 2 — per-fold balanced accuracy (k=2) | control `[0.28725, 0.26996]`, arm `[0.30324, 0.29928]` |
| 2 — delta macro-F1 / delta log-loss | **+0.02600** / **−0.10377** (log-loss down = better) |
| 3 — 5 subsampled k=2 runs @ 80 % of prompt blocks | deltas `+0.0111, +0.0140, +0.0096, +0.0173, +0.0145` → mean **+0.01329** ± 0.00303, **positive 5/5** |
| 4 — 400-draw bootstrap over prompt blocks | mean **+0.02196**, 95 % CI **[+0.01201, +0.03259]**, positive in 100 % of draws |
| 5 — damped calibration arm (`p' = 0.85 p + 0.15/K`, post-hoc) | balanced-accuracy delta **exactly 0**, log-loss **−0.01347** |
| 6 — TRACE without standardising the rank block | balanced accuracy **0.30722**, delta **+0.02862** |
| 1 — leak gate | 0 prompt blocks shared between train and validation; all 5,070 rows scored exactly once |
| 0 — corpus | 5,070 passages / 296 prompt blocks (253×18 models + 43×12) / 18 labels, 253–296 per label |

**Verdict: PASS.** The arm must beat the control on the k=2 OOF delta, with a block-bootstrap CI above
zero and positive deltas in ≥ 80 % of subsamples — it does, on all three.

## What the two arms are

* **V7 control** (documented re-implementation of the house baseline): TF-IDF word (1–2 g, cap 40 k) +
  TF-IDF `char_wb` (2–5 g, cap 40 k) + a 14-column stylometry block, every transform fitted on the
  training rows only → one multinomial logistic regression (`C=4`, `class_weight="balanced"`, lbfgs, seed 0).
* **V8-TRACE arm**: exactly that control, plus 30 rank-transition columns built inside each prompt block:

  ```
  ranks  r_v(d) = rank of style view v among d's block siblings, scaled to [0,1]   14 columns
  steps  t_v(d) = r_{v+1}(d) − r_v(d)                                              13 columns
  sums   max|t|, mean|t|, number of sign flips in t                                 3 columns
  ```

  The level of a style view is mostly topic and length protocol; its **position inside the block** is what
  the author moves, and a model's register shows up as a *pattern of steps* across views — which no
  single-view feature can express. The classifier, hyperparameters, folds and seeds are identical for both
  arms: the only difference is the feature block, so the delta belongs to the arm alone.

**Leakage rules (enforced by tests):** TRACE uses no labels; a row's features depend only on its own
prompt group, so computing them on the train side or the validation side yields identical numbers; all
vectorizers/scalers are fitted on training rows; whole prompt blocks move between folds — a random row
split would put the same prompt in train and validation and badly overstate accuracy.

## Data (public, ships in the repo)

[`data/passages_clean.csv.gz`](data/DATASET_CARD.md) — all 5,070 clean rows of
[`prvsiyan/extended-name-that-llm-dataset`](https://www.kaggle.com/datasets/prvsiyan/extended-name-that-llm-dataset)
(CC BY 4.0; `train_regen` prompt texts CC BY-SA 4.0, attribution pjmathematician), cut to nine columns and
gzipped. Upstream file sha256 `502653b8…9eaf9`, shipped gzip sha256 `3b3097a6…7d9ae`. Every passage is one
answer to one prompt, and prompts were issued as blocks — same prompt, every model once — which is exactly
what a prompt-grouped check needs.

## Reproduce

```bash
pip install -r requirements.txt          # pinned; exact versions listed below
python run_oof.py --reps 5 --boot 400 --out evidence/oof_report.json   # ~17-27 min, one core
python -m pytest tests -q                # 19 checks, ~30 s
# same battery from the notebook path (the shipped .ipynb already carries these outputs):
python - <<'PY'
import nbclient, nbformat
nb = nbformat.read("v8_trace_arm.ipynb", as_version=4)
nbclient.NotebookClient(nb, timeout=2400, kernel_name="python3",
                        resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "v8_trace_arm.ipynb")
PY
```

Measured on Python 3.12.8 / numpy 2.5.4 / pandas 3.1.1 / scipy 1.18.1 / scikit-learn 1.9.1 with BLAS pinned
to one thread (`OMP_NUM_THREADS=1` in `run_oof.py`). Seeds fixed everywhere. Two independent code paths —
`run_oof.py` (per-arm fitting) and the notebook (shared fitting) — produced identical metrics digit for
digit; only the wall-clock seconds differ. The same repo re-run unmodified on Ubuntu 24.04 (Python 3.12.3,
same pinned libraries) reproduces the primary gate exactly — [`evidence/linux_rerun.md`](evidence/linux_rerun.md).

## Honest limits

* The delta is measured on **this public corpus**, not on the Kaggle leaderboard: the host's own prompts are
  unpublished, so `train_regen` rows use community-reconstructed prompt texts. Absolute balanced accuracy
  here (0.28–0.31 over 18 classes, chance 0.0556) says how hard sibling attribution is when topic is held
  fixed; it is a signal-isolation harness, not a submission attempt.
* The primary k=2 delta (**+0.0227**) is bigger than the 80 %-subsample average (**+0.0133**). Both are
  positive and the block bootstrap excludes zero, but gate 3 is the more conservative read: expect roughly
  +0.013 on a fresh fold draw, not +0.023. The gate-4 CI shares gate 2's single-run optimism.
* The shipped arm **standardises** the rank block (consistent with how the control treats numerics) and
  scores +0.0227; the unscaled variant scores **higher** (+0.0286), so the headline number is the
  conservative choice, not the cherry-picked one.
* Folds mix dataset splits (`new`, `proxy_v1`, `train_regen`, `new2`), so a split-specific generation-protocol
  cue can help inside a training fold — never across folds, since blocks are whole and every split appears on
  both sides.
* Only two of the five V8 arms are measured here: TRACE (the only pure new-signal arm needing no external
  data or live API) and damped calibration, which rides the same OOF pass for free and is reported at gate 5
  (it cannot move balanced accuracy by construction — an increasing row-wise map — so it only moves log-loss).
* Provider API terms restrict using AI outputs to train competing commercial models; this is a verification
  harness on an explicitly CC BY 4.0 research dataset. Check the dataset card before commercial reuse.
* Cross-platform drift is real but tiny: on Ubuntu the primary gate reproduces exactly, while one of five
  subsampled stability runs moved by 0.00023 (last-bit lbfgs reductions) and log-loss differs at 1e-8. The
  verdict is `PASS` on both systems - [`evidence/linux_rerun.md`](evidence/linux_rerun.md).

## Repo map

```
v8_trace_arm.ipynb        standalone notebook: loads data, defines arms, runs all gates (executed, outputs inline)
run_oof.py                same battery from the shell, writes evidence/oof_report.json
ntl_v8/features.py        14 style views; within-group ranks + transitions (label-free)
ntl_v8/model.py           fold-safe feature assembly, both arms, damping post-processor
ntl_v8/oof.py             k=2 group folds, OOF harness, gates 0-6, block bootstrap, verdict
ntl_v8/data.py            corpus loading + fold-ready grouping
tests/                    19 checks: rank locality/monotonicity/ties, label-freeness, permutation
                          equivariance, control-is-a-prefix, train-only fitting, fold partition,
                          determinism, damping argmax-preservation, end-to-end smoke
data/                     passages_clean.csv.gz + DATASET_CARD.md + column dictionary subset
evidence/oof_report.json           every gate number above, plus environment and data hash (shell path)
evidence/oof_report_notebook.json  same battery from the notebook path; agrees metric-for-metric
evidence/oof_report_linux.json     Ubuntu re-run (gate 6 skipped there)
evidence/linux_rerun.md            field-by-field Windows vs Ubuntu comparison, with drift quantified
```

Code MIT (see `LICENSE`); data CC BY 4.0 / CC BY-SA 4.0 as noted above and in `data/DATASET_CARD.md`.
