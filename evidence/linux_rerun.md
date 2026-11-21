# Cross-platform re-run — Ubuntu 24.04 (host `linserv`, 10.0.0.40)

The same repo, unmodified, was copied to a Linux box and the OOF check re-run there, because the house
re-runs this check itself and probably not on Windows Server.

| | Windows run (canonical) | Ubuntu re-run |
|---|---|---|
| platform | Windows-2022Server-10.0.26100 | Linux-6.8.0-149-generic x86_64, glibc 2.39 |
| python / numpy / pandas / scipy / scikit-learn | 3.12.8 / 2.5.4 / 3.1.1 / 1.18.1 / 1.9.1 | 3.12.3 / 2.5.4 / 3.1.1 / 1.18.1 / 1.9.1 |
| BLAS threads | 1 (`OMP_NUM_THREADS=1`) | 1 |
| data sha256 | `3b3097a6…7d9ae` | identical |
| `pytest tests -q` | 19 passed in 34.7 s | **19 passed in 20.9 s** |
| command | `run_oof.py --reps 5 --boot 400` (gate 6 included) | `run_oof.py --reps 5 --boot 400 --no-variants` |
| wall time | 1,609.7 s (shared box) | **1,013.7 s** |

Machine-readable copy of the Linux run: [`oof_report_linux.json`](oof_report_linux.json).

## Numbers, compared field by field

| quantity | Windows | Ubuntu | difference |
|---|---|---|---|
| primary k=2 balanced accuracy — control | 0.27859603674821065 | same | **exact** |
| primary k=2 balanced accuracy — arm | 0.3012721219242958 | same | **exact** |
| per-fold balanced accuracy (both arms) | `[0.287246, 0.269957]` / `[0.303244, 0.299284]` | same | **exact** |
| delta balanced accuracy | +0.02267608517608516 | same | **exact** |
| delta macro-F1 | +0.026004799800151746 | same | **exact** |
| delta log-loss | −0.10376887696565129 | −0.10376890678900175 | 3.0e-08 (float reduction order) |
| bootstrap CI of delta (400 draws over blocks) | `[+0.012006, +0.032589]` | same | **exact** |
| gate-3 subsampled runs (seeds 1, 2, 4, 5) | as shipped | same | **exact** |
| gate-3 run seed 3 — arm BA | 0.29223762641484163 | 0.2924720380416583 | +0.000234 (one document's argmax moved) |
| gate-3 summary mean / min delta | +0.013289 / +0.009554 | +0.013336 / +0.009788 | 4.7e-05 / 2.3e-04 |

**Read-out, honestly:** the headline gate reproduces exactly across operating systems with the pinned
libraries — same folds, same per-fold accuracies, same delta, same bootstrap interval. The only drift is in
the *subsampled stability* battery, where one of five runs flips a single document's argmax because lbfgs
gradient reductions differ in the last bits between platforms; that moves the mean delta by 0.00005 and the
worst run by 0.00023. The verdict is `PASS` on both systems, and the sign of every delta stays positive in
10/10 subsampled runs across the two platforms.

If a future library bump changes more than the last digits, re-run `python run_oof.py --quick` (a few
minutes) and compare against this file before trusting any leaderboard claim built on the arm.
