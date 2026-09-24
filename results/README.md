# Training results

Saved, reproducible training runs (seeded, `torch.manual_seed(0)`),
produced by the config-driven `src/train.py` CLI (see the main README's
"Reproducing experiments" section). Each run directory under
`results/runs/` contains everything needed to reproduce or reload it:

```
results/runs/<name>/
  config.yaml           # the fully-resolved Hydra config this run used
  checkpoint.pt          # final trained model weights -- reloadable via src/infer.py
  history.json           # per-epoch train/test NSE, basin ID lists, timing
  test_predictions.json  # final-epoch per-basin sim streamflow + NSE on the test set
  checkpoints/
    epoch_%04d.pt         # model + optimizer state, every train.checkpoint_every
                          # epochs and once more on the final epoch -- for
                          # resuming, not needed to just reload/score a run
                          # (not committed -- see .gitignore)
```

## `runs/model_9yrs_spatial/` -- primary result

> **⚠️ Stale w.r.t. current code — retrain pending.** This run was trained
> under the previous *reverse-mode* cross-container coupling. The gradient
> path has since been rewritten to *forward-mode* Tesseract composition
> (see `src/coupling.py` and notes/logs.md, 2026-09-22). The two are
> gradient-equivalent up to finite-difference noise — the test suite checks
> the coupled gradient against an independent brute-force FD through both
> real Fortran models — and the forward pass is unchanged, so the committed
> `checkpoint.pt` still reloads and scores as recorded via `src/infer.py`.
> But re-running training now follows a different optimization trajectory
> and will not reproduce the exact numbers below. A fresh run under the new
> code has not yet been committed; the table and `test_predictions.json`
> reflect the prior run. Treat the numbers as indicative, not as the
> current code's reproducible output, until this run is regenerated.

Snow17 + SAC-SMA + `ParamNet` (LSTM climatology encoder + static
attributes -> 27 bounded physical parameters), trained end-to-end
through both Tesseracts via `src/coupling.py`. 35 train / 10 heldout
basins (`split=spatial`), WY1991-1999 (9-year) window, 150 epochs,
~13 minutes total. This is `configs/`'s current default; regenerate with:

```bash
.venv/bin/python src/train.py output_dir=results/runs/model_9yrs_spatial
```

| epoch | median train NSE | median test (heldout) NSE |
|---|---|---|
| 1 | +0.38 | +0.28 |
| 150 (final) | **+0.84** | **+0.70** |

Train/test gap at epoch 150: **~0.14** -- held-out basins track training
basins closely throughout, no overfitting observed at this scale.
Per-basin simulated streamflow + NSE for all 10 test basins at this
final epoch: `runs/model_9yrs_spatial/test_predictions.json`.

## External reference: a properly-engineered LSTM

[`results/external/neuralhydrology_lstm_pub/`](external/neuralhydrology_lstm_pub/README.md)
-- a [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
LSTM trained and tested on **the exact same 35/10 basin split** as the
run above:

| model | median test NSE |
|---|---|
| NeuralHydrology LSTM | **0.795** |
| our hybrid model (`runs/model_9yrs_spatial/`) | 0.70 |

See that directory's README for details and reproduction steps.

## Comparing runs

`results/compare_runs.py` turns a comparison like the one above into a
repeatable script instead of a hand-copied table -- run it whenever a
new run needs checking against the canonical one (e.g. after a
coupling/gradient change, or a different seed):

```bash
.venv/bin/python src/train.py seed=1 output_dir=results/runs/model_9yrs_spatial_seed1
.venv/bin/python results/compare_runs.py \
    results/runs/model_9yrs_spatial \
    results/runs/model_9yrs_spatial_seed1
```

Takes any number of run directories, not just two. Prints final
train/test NSE, the train/test gap, and seconds/epoch for each run;
also saves an overlaid NSE-vs-epoch plot to `<first run dir>/comparison.png`
if matplotlib is installed (skip with `--no-plot`).

**`agg` column:** `src/train.py` reports cross-basin NSE as a
**median** (a few badly-fit basins shouldn't dominate the headline
number the way they would under a mean). The script reads whichever
aggregation a given run's `history.json` actually has and labels it in
the `agg` column rather than assuming -- and prints an explicit warning
if you pass runs that mix mean- and median-aggregated `history.json`
files, since that comparison isn't apples to apples (relevant if you
ever compare against a run from before this project's own median
switch).

## Predictions from a trained checkpoint

`src/infer.py` loads a saved `checkpoint.pt` and scores it (no
training) against a config-selected split -- useful for checking a
model against a different window or basin set without retraining:

```bash
.venv/bin/python src/infer.py \
    checkpoint=results/runs/model_9yrs_spatial/checkpoint.pt
```

Writes `<output_dir>/predictions.json`: per-basin simulated streamflow
(mm/day) + NSE for every basin in the composed split. (For just the
final-epoch test-set result of the canonical run itself, the training
run already wrote this -- see `test_predictions.json` above; `infer.py`
is for scoring against a *different* split/window without retraining.)
