"""
Build a synthetic dataset for the brachistochrone surrogate.

Usage:
    python prepare.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

# Output paths for generated data artifacts.
RESULTS_DIR = os.path.join("results", "brachistochrone")
DATASET_PATH = os.path.join(RESULTS_DIR, "dataset.pt")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "prepare_summary.json")

# Fixed brachistochrone geometry and physics constants.
X0, Y0 = 0.0, 1.0
X1, Y1 = 1.0, 0.0
K = 16          # Number of free interior y values.
Y_MIN = -0.5    # Lower bound so the curve may dip below the final point.
G = 9.81        # Gravity used in the travel-time calculation.
EPS = 1e-8      # Prevents division by zero near the zero-speed start.

# Synthetic dataset settings.
N_SAMPLES = 20000
TEST_FRAC = 0.2
BATCH_SIZE = 256  # Saved with the dataset so train.py can reuse the default.
SEED = 42


def calculate_travel_time(trajectory: np.ndarray) -> float:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx * dx + dy * dy)

    v0 = np.sqrt(np.clip(2.0 * G * (Y0 - y[:-1]), 0.0, None))
    v1 = np.sqrt(np.clip(2.0 * G * (Y0 - y[1:]), 0.0, None))
    vbar = np.maximum(0.5 * (v0 + v1), EPS)
    return float(np.sum(ds / vbar))


def straight_line_trajectory() -> np.ndarray:
    x_values = np.linspace(X0, X1, K + 2, dtype=np.float64)
    y_values = Y0 + (Y1 - Y0) * (x_values - X0) / (X1 - X0)
    return np.column_stack((x_values, y_values))


def sample_trajectory(random_generator: np.random.Generator) -> np.ndarray:
    x_values = np.linspace(X0, X1, K + 2, dtype=np.float64)
    y_values = random_generator.uniform(Y_MIN, Y0, size=K)
    y_values = np.concatenate(([Y0], y_values, [Y1])).astype(np.float64)
    return np.column_stack((x_values, y_values))


def build_dataset(path: str = DATASET_PATH) -> dict[str, torch.Tensor | dict[str, float | int]]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    random_generator = np.random.default_rng(SEED)

    interior_y_values = np.empty((N_SAMPLES, K), dtype=np.float64)
    travel_times = np.empty((N_SAMPLES, 1), dtype=np.float64)

    for i in range(N_SAMPLES):
        trajectory = sample_trajectory(random_generator)
        interior_y_values[i] = trajectory[1:-1, 1]
        travel_times[i, 0] = calculate_travel_time(trajectory)

    permutation = random_generator.permutation(N_SAMPLES)
    train_size = int((1.0 - TEST_FRAC) * N_SAMPLES)
    train_idx = permutation[:train_size]
    test_idx = permutation[train_size:]

    x_train_raw = torch.tensor(interior_y_values[train_idx], dtype=torch.float32)
    x_test_raw = torch.tensor(interior_y_values[test_idx], dtype=torch.float32)
    t_train_raw = torch.tensor(travel_times[train_idx], dtype=torch.float32)
    t_test_raw = torch.tensor(travel_times[test_idx], dtype=torch.float32)

    x_mean = x_train_raw.mean(dim=0)
    x_std = x_train_raw.std(dim=0, unbiased=False).clamp_min(1e-6)
    t_mean = t_train_raw.mean(dim=0)
    t_std = t_train_raw.std(dim=0, unbiased=False).clamp_min(1e-6)

    payload = {
        "config": {
            "x0": X0,
            "y0": Y0,
            "x1": X1,
            "y1": Y1,
            "k": K,
            "y_min": Y_MIN,
            "gravity": G,
            "num_samples": N_SAMPLES,
            "test_fraction": TEST_FRAC,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "x_train": (x_train_raw - x_mean) / x_std,
        "x_test": (x_test_raw - x_mean) / x_std,
        "t_train": (t_train_raw - t_mean) / t_std,
        "t_test": (t_test_raw - t_mean) / t_std,
        "x_mean": x_mean,
        "x_std": x_std,
        "t_mean": t_mean,
        "t_std": t_std,
    }
    torch.save(payload, path)

    summary = {
        "dataset_path": path,
        "num_samples": N_SAMPLES,
        "train_size": int(x_train_raw.size(0)),
        "test_size": int(x_test_raw.size(0)),
        "time_min": float(t_train_raw.min().item()),
        "time_max": float(t_train_raw.max().item()),
        "time_mean": float(t_train_raw.mean().item()),
        "straight_line_time": calculate_travel_time(straight_line_trajectory()),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return payload


def load_dataset(path: str = DATASET_PATH) -> dict[str, torch.Tensor | dict[str, float | int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare.py first.")
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    payload = build_dataset()
    print(f"Saved dataset to {DATASET_PATH}")
    print(f"Train samples: {payload['x_train'].size(0)}")
    print(f"Test samples: {payload['x_test'].size(0)}")
    print(f"Straight-line time: {calculate_travel_time(straight_line_trajectory()):.6f}")
