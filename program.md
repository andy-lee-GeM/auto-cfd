# auto-cfd

This is an experiment to have the LLM do its own research on the
brachistochrone surrogate problem.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date, for example `apr8-demo`.
2. Create a fresh branch:
   `git checkout -b auto-cfd/<tag>`
3. Read the in-scope files for context:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `program.md`
4. Verify that the fixed dataset already exists at:
   `results/brachistochrone/dataset.pt`
5. Do not run `prepare.py` during the experiment loop. The dataset is already
   generated and must stay fixed across all experiments.
6. Use this interpreter for all runs:
   `/home/andylee/.venvs/auto-cfd-cpu/bin/python`
7. Initialize:
   - `results.tsv` with the header row only
   - `notes.md` as an untracked running lab notebook for experiment notes
8. Run the baseline with the current `train.py`.

## Goal

The goal is to make the surrogate in `train.py` approximate the true simulator
as accurately as possible on the fixed dataset, and to keep that behavior
stable as the dataset grows.

The agent should treat `train.py` as the research surface. Everything there is
fair game:

- model architecture
- hidden size
- network depth
- optimizer
- learning rate
- batch size
- number of epochs
- loss shaping
- training loop details
- surrogate optimization settings

The primary objective is surrogate quality on held-out data:

1. lower prediction error on true travel time
2. improve `test_r2`
3. lower `test_mae`
4. keep the model simple enough to generalize when more data is added later

The downstream optimized trajectory is still useful, but it is a secondary
check. It helps verify that the surrogate behaves sensibly when used for design
optimization.

Training time is a hard practical constraint. Each experiment should stay under
5 minutes on the CPU setup for this repo. Increase model complexity
incrementally so runtime remains within budget.

## What You Can Modify

- `train.py`

## What You Should Not Modify

- `prepare.py`
- dependency files
- the fixed dataset at `results/brachistochrone/dataset.pt`

The point of this run is to compare modeling and training changes on the same
data.

## Environment

Fresh agents have had trouble running this repo when they guess the interpreter.
Do not guess.

Always run experiments with:

```bash
/home/andylee/.venvs/auto-cfd-cpu/bin/python train.py
```

If you need to inspect the dataset or rerun the baseline, use the same
interpreter.

## Evaluation Priority

Use this priority order:

1. Higher surrogate `R2`
2. Lower surrogate `MAE`
3. Lower other held-out prediction error metrics
4. Sensible downstream optimized-trajectory behavior
5. Simpler code and more stable behavior

Do not optimize only for training loss. Held-out surrogate accuracy is the main
target, because the model needs to stay reliable as more data is added later.

## Output Format

Each run of `train.py` prints a summary like:

```text
---
test_r2:                     0.123456
test_mae:                    0.123456
test_mse:                    0.123456
base_case_true_time:         0.638551
best_predicted_time:         0.600000
best_true_time:              0.590000
delta_vs_base_true:          0.048551
delta_vs_base_true_pct:      7.603
loss_plot:                   results/brachistochrone/training_loss.png
curve_plot:                  results/brachistochrone/curve_comparison.png
prediction_plot:             results/brachistochrone/test_predictions.png
summary_json:                results/brachistochrone/train_summary.json
```

Treat `test_r2`, `test_mae`, and `test_mse` as the primary metrics.
Use `best_true_time` and `curve_plot` as secondary checks to confirm that the
surrogate behaves sensibly when optimized.

## Logging Results

After each experiment, append one row to `results.tsv` using tab separation.

Header:

```text
commit	test_r2	test_mae	test_mse	true_time	predicted_time	decision	change
```

Columns:

1. short git commit hash
2. held-out `test_r2`
3. held-out `test_mae`
4. held-out `test_mse`
5. true travel time of the optimized trajectory
6. surrogate-predicted travel time of the optimized trajectory
7. `keep`, `discard`, or `crash`
8. short description of what changed in `train.py`

## Notes

After each experiment, write a short note to `notes.md`.

Each note should include:

- experiment number
- commit hash
- what was changed
- key metrics:
  - `test_r2`
  - `test_mae`
  - `test_mse`
  - `true_time`
  - `predicted_time`
- the decision: `keep`, `discard`, or `crash`
- one sentence about what to try next

Keep `notes.md` untracked by git.

## Experiment Loop

For each iteration:

1. Check the current git state.
2. Make one focused change to `train.py`.
3. Commit the change.
4. Run:
   `/home/andylee/.venvs/auto-cfd-cpu/bin/python train.py > run.log 2>&1`
5. If the run crashes, inspect and fix:
   `tail -n 50 run.log`
6. Read the final metrics from `run.log` or from:
   `results/brachistochrone/train_summary.json`
7. Append a row to `results.tsv`.
8. Append a note to `notes.md`.
9. Keep the commit only if held-out surrogate quality improved. Use `test_r2`
   first, then `test_mae` and `test_mse` as tie-breakers. Use downstream
   optimized-trajectory behavior only as a secondary sanity check. Otherwise
   revert to the previous good commit.
10. If the result improved and the commit is kept, push it to git immediately so
    the best result is preserved remotely.
11. On every kept improvement:
    - `git add train.py` only if those files were
      intentionally changed
    - create a normal commit
    - push the branch immediately
    - do not add generated data, plots, summaries, logs, or dataset files

## General Guidance

- Keep changes small and attributable.
- Prefer simple ideas before complex ones.
- Increase model complexity gradually. Do not jump straight to a large network.
- Reuse the same dataset for the whole run.
- Review both metrics and plots. A result that looks numerically good but
  produces a visibly strange optimized curve is suspicious.
- Use branch history plus `results.tsv` and `notes.md` to inspect progression
  over time. The git history should represent the sequence of kept model
  improvements; generated artifacts stay local and untracked.
- Do not stop to ask whether to continue once the loop begins unless the run is
  blocked by something external.
