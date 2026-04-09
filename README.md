# auto-cfd

Minimal surrogate-optimization scaffold for a baby CFD-style problem: the
brachistochrone.

The goal is simple: train a neural network surrogate that predicts travel time
for a discretized curve between two fixed endpoints, then optimize the curve
through the surrogate and check whether the recovered shape approaches the true
cycloid.

## Problem

We fix:

- `A = (0, 1)`
- `B = (1, 0)`
- `K` evenly spaced interior `y` values as design variables

The ground-truth simulator treats the full curve as a polyline and computes a
discrete travel time under gravity:

```text
T = sum_i ds_i / vbar_i
```

This simulator is cheap, but we use it as if it were an expensive CFD solver:
only for synthetic data generation and final validation.

## Files

- **`prepare.py`**: defines the brachistochrone geometry, discrete simulator,
  Latin Hypercube Sampling, train/test split, normalization, and cycloid
  reference helpers.
- **`train.py`**: trains a small MLP surrogate with MSE loss, reports `R2` and
  `MAE`, optimizes the design vector through the frozen surrogate, and saves the
  plots and summary metrics.
- **`program.md`**: lightweight instructions for an external coding agent if
  you want to run the experiment loop autonomously.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Generate synthetic brachistochrone data
uv run prepare.py

# 3. Train the surrogate and optimize the curve
uv run train.py
```

Artifacts are written to `results/brachistochrone/`.

## Outputs

The training script writes:

- `training_loss.png`
- `test_predictions.png`
- `curve_comparison.png`
- `train_summary.json`

The summary includes:

- surrogate test `R2`
- surrogate test `MAE`
- straight-line travel time
- surrogate-predicted optimal time
- true travel time of the surrogate-optimal curve
- analytical cycloid time
- percent gap between the surrogate-optimal true time and the cycloid time

## Design Notes

- The repo is intentionally flat and small.
- Data is self-generated, so there is no download step or tokenizer.
- Inputs and targets are normalized before training.
- The baseline model is a small MLP, not a transformer.
- The real downstream score is the true travel time after optimizing through the
  surrogate, not training loss alone.
