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

> **Trained under the previous coupling.** This run used the earlier
> hand-written cross-container coupling. The gradient path has since been
> rewritten to *forward-mode* Tesseract composition (see `src/coupling.py`
> and notes/logs.md, 2026-09-22 and 2026-09-27). The forward pass is
> unchanged, so `checkpoint.pt` still reloads and scores as recorded via
> `src/infer.py`, but re-running training now follows a slightly different
> optimization trajectory. The current code's reproduction of this run is
> `runs/model_9yrs_spatial_fwdmode/` below.

Snow17 + SAC-SMA + `ParamNet` (LSTM climatology encoder + static
attributes -> 27 bounded physical parameters), trained end-to-end
through both Tesseracts via `src/coupling.py`. 35 train / 10 heldout
basins (`split=spatial`), WY1991-1999 (9-year) window, 150 epochs,
~13 minutes total under the previous coupling. This is `configs/`'s
current default.

| epoch | median train NSE | median test (heldout) NSE |
|---|---|---|
| 1 | +0.38 | +0.28 |
| 150 (final) | **+0.84** | **+0.70** |

Train/test gap at epoch 150: **~0.14** -- held-out basins track training
basins closely throughout, no overfitting observed at this scale.
Per-basin simulated streamflow + NSE for all 10 test basins at this
final epoch: `runs/model_9yrs_spatial/test_predictions.json`.

## `runs/model_9yrs_spatial_fwdmode/` -- reproduction under the current code

The same config and seed as the run above, retrained from scratch with
the current forward-mode coupling:

```bash
.venv/bin/python src/train.py output_dir=results/runs/model_9yrs_spatial_fwdmode
```

| epoch | median train NSE | median test (heldout) NSE |
|---|---|---|
| 1 | +0.38 | +0.28 |
| 150 (final) | **+0.84** | **+0.73** |

Epoch 1 matches the run above exactly (same initial network, same
forward pass). By epoch 150 the training score is unchanged, and the
median held-out score is 0.03 higher. That is not an improvement: basin
by basin, 5 of the 10 held-out basins get better and 5 get worse, the
largest change being `13313000` dropping from 0.33 to -0.02, and the
mean held-out NSE falls from 0.66 to 0.62. Within this run, the median
held-out score also ranges from 0.69 to 0.73 across its last five
evaluations. Read it as the same
result within run-to-run variation; this project has no multi-seed
estimate of that variation yet.

Cost: **~50 s/epoch, ~2 h total** (vs. ~5 s/epoch before) -- each
gradient now takes ~126 Fortran runs per basin instead of 63, plus
per-call Tesseract overhead (see notes/logs.md, 2026-09-27).

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
