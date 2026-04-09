"""
Brachistochrone surrogate training and optimization.

Usage:
    python train.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    DATASET_PATH,
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    G,
    RESULTS_DIR,
    X0,
    X1,
    Y0,
    Y1,
    Y_MIN,
    build_dataset,
    calculate_travel_time,
    load_dataset,
    straight_line_trajectory,
)

# Keep the baseline deliberately simple.
BATCH_SIZE = DEFAULT_BATCH_SIZE  # Match the saved dataset default unless overridden.
EPOCHS = 400                     # Training passes over the surrogate dataset.
LEARNING_RATE = 1e-3             # Optimizer step size for the surrogate weights.
SEED = 42                        # Reproducible training and optimization behavior.
OPT_STEPS = 500                  # Gradient steps taken on the design vector.
OPT_LR = 2e-2                    # Step size when optimizing the curve through the surrogate.
HIDDEN_DIM = 128                 # Width of the hidden layers in the MLP surrogate.
MONOTONIC_WEIGHT = 10.0          # Penalize uphill segments in the optimized curve.
SMOOTHNESS_WEIGHT = 1.0          # Penalize large second differences in the optimized curve.

# Output files for the training run.
SUMMARY_PATH = os.path.join(RESULTS_DIR, "train_summary.json")
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "training_loss.png")
CURVE_PLOT_PATH = os.path.join(RESULTS_DIR, "curve_comparison.png")
PREDICTION_PLOT_PATH = os.path.join(RESULTS_DIR, "test_predictions.png")


def clamp_design(y_interior: torch.Tensor) -> torch.Tensor:
    return y_interior.clamp(Y_MIN, Y0)


def build_trajectory(interior_y_values: np.ndarray) -> np.ndarray:
    interior_y_values = np.asarray(interior_y_values, dtype=np.float64)
    interior_x_values = np.linspace(X0, X1, interior_y_values.shape[0] + 2, dtype=np.float64)[1:-1]
    x_values = np.concatenate(([X0], interior_x_values, [X1])).astype(np.float64)
    y_values = np.concatenate(([Y0], interior_y_values, [Y1])).astype(np.float64)
    return np.column_stack((x_values, y_values))


def augment_design_with_endpoints(y_interior: torch.Tensor) -> torch.Tensor:
    y0 = torch.full((y_interior.size(0), 1), Y0, dtype=y_interior.dtype, device=y_interior.device)
    y1 = torch.full((y_interior.size(0), 1), Y1, dtype=y_interior.dtype, device=y_interior.device)
    return torch.cat((y0, y_interior, y1), dim=1)


def normalize_design(y_interior: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (y_interior - stats["x_mean"].to(y_interior.device)) / stats["x_std"].to(y_interior.device)


def denormalize_time(pred_time_norm: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return pred_time_norm * stats["t_std"].to(pred_time_norm.device) + stats["t_mean"].to(pred_time_norm.device)


def make_dataloaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader, dict[str, torch.Tensor], dict[str, object]]:
    payload = load_dataset(DATASET_PATH)
    train_dataset = TensorDataset(payload["x_train"], payload["t_train"])
    test_dataset = TensorDataset(payload["x_test"], payload["t_test"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    stats = {
        "x_mean": payload["x_mean"],
        "x_std": payload["x_std"],
        "t_mean": payload["t_mean"],
        "t_std": payload["t_std"],
    }
    return train_loader, test_loader, stats, payload


def solve_cycloid_parameters() -> tuple[float, float]:
    dx = X1 - X0
    drop = Y0 - Y1

    def f(theta: float) -> float:
        return drop * (theta - np.sin(theta)) - dx * (1.0 - np.cos(theta))

    lo = 1e-6
    hi = 2.0 * np.pi - 1e-6
    flo = f(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-12:
            theta_end = mid
            break
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    else:
        theta_end = 0.5 * (lo + hi)

    radius = drop / (1.0 - np.cos(theta_end))
    return radius, theta_end


def cycloid_curve(num_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    radius, theta_end = solve_cycloid_parameters()
    theta = np.linspace(0.0, theta_end, num_points, dtype=np.float64)
    x = X0 + radius * (theta - np.sin(theta))
    y = Y0 - radius * (1.0 - np.cos(theta))
    return x, y


def cycloid_time() -> float:
    radius, theta_end = solve_cycloid_parameters()
    return float(theta_end * np.sqrt(radius / G))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class MLPSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    pred_batches = []
    target_batches = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            losses.append(loss.item())
            pred_batches.append(denormalize_time(pred, stats).cpu())
            target_batches.append(denormalize_time(yb, stats).cpu())

    preds = torch.cat(pred_batches).view(-1).numpy()
    targets = torch.cat(target_batches).view(-1).numpy()
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    metrics = {"mse": mse, "mae": mae, "r2": r2}
    return float(np.mean(losses)), mae, metrics, preds, targets


def train_surrogate(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[float], list[float], dict[str, float], np.ndarray, np.ndarray]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_losses: list[float] = []
    test_losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            count += xb.size(0)

        train_losses.append(running_loss / count)
        test_loss, _, test_metrics, preds, targets = evaluate_model(
            model, test_loader, stats, device
        )
        test_losses.append(test_loss)

        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch:04d} | "
                f"train_mse_norm: {train_losses[-1]:.6f} | "
                f"test_mse_norm: {test_losses[-1]:.6f} | "
                f"test_r2: {test_metrics['r2']:.6f} | "
                f"test_mae: {test_metrics['mae']:.6f}"
            )

    _, _, final_metrics, preds, targets = evaluate_model(model, test_loader, stats, device)
    return train_losses, test_losses, final_metrics, preds, targets


def optimize_design(
    model: nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, float, list[float]]:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    initial_design = torch.tensor(
        straight_line_trajectory()[1:-1, 1],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    design = nn.Parameter(initial_design.clone())
    optimizer = torch.optim.Adam([design], lr=OPT_LR)

    best_pred_time = float("inf")
    best_design = None
    pred_history: list[float] = []

    for step in range(1, OPT_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        clamped = clamp_design(design)
        pred_norm = model(normalize_design(clamped, stats))
        pred_time = denormalize_time(pred_norm, stats).mean()
        full_design = augment_design_with_endpoints(clamped)
        monotonic_penalty = F.relu(full_design[:, 1:] - full_design[:, :-1]).pow(2).mean()
        smoothness_penalty = (
            full_design[:, 2:] - 2.0 * full_design[:, 1:-1] + full_design[:, :-2]
        ).pow(2).mean()
        objective = (
            pred_time
            + MONOTONIC_WEIGHT * monotonic_penalty
            + SMOOTHNESS_WEIGHT * smoothness_penalty
        )
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            design.copy_(clamp_design(design))
            current_design = design.detach().clone()
            current_pred = float(
                denormalize_time(
                    model(normalize_design(current_design, stats)),
                    stats,
                ).item()
            )

        pred_history.append(current_pred)
        if current_pred < best_pred_time:
            best_pred_time = current_pred
            best_design = current_design.detach().cpu().numpy().reshape(-1)

        if step == 1 or step % 100 == 0 or step == OPT_STEPS:
            print(f"opt_step {step:04d} | predicted_time: {current_pred:.6f}")

    if best_design is None:
        raise RuntimeError("Optimization did not produce a valid design.")

    return best_design, best_pred_time, pred_history


def save_training_plot(train_losses: list[float], test_losses: list[float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train MSE (normalized)")
    plt.plot(test_losses, label="Test MSE (normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Surrogate Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=160)
    plt.close()


def save_curve_plot(best_design: np.ndarray) -> None:
    opt_trajectory = build_trajectory(best_design)
    x_cyc, y_cyc = cycloid_curve()
    line_trajectory = straight_line_trajectory()

    plt.figure(figsize=(7, 5))
    plt.plot(x_cyc, y_cyc, label="Cycloid", linewidth=2.5)
    plt.plot(opt_trajectory[:, 0], opt_trajectory[:, 1], "o-", label="Surrogate-optimal", linewidth=2)
    plt.plot(line_trajectory[:, 0], line_trajectory[:, 1], "--", label="Straight line", linewidth=1.5)
    plt.scatter([X0, X1], [Y0, Y1], color="black", zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Brachistochrone Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CURVE_PLOT_PATH, dpi=160)
    plt.close()


def save_prediction_plot(preds: np.ndarray, targets: np.ndarray) -> None:
    low = float(min(preds.min(), targets.min()))
    high = float(max(preds.max(), targets.max()))

    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, s=14, alpha=0.65)
    plt.plot([low, high], [low, high], "k--", linewidth=1.25)
    plt.xlabel("True travel time")
    plt.ylabel("Predicted travel time")
    plt.title("Surrogate Test Predictions")
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=160)
    plt.close()


def save_summary(summary: dict[str, object]) -> None:
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        print(f"{DATASET_PATH} missing, generating dataset first...")
        build_dataset()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, stats, payload = make_dataloaders(batch_size=BATCH_SIZE)
    stats = {key: value.to(device) for key, value in stats.items()}

    input_dim = int(payload["x_train"].shape[1])
    model = MLPSurrogate(input_dim).to(device)

    print(
        f"Training surrogate | model=mlp | input_dim={input_dim} | hidden_dim={HIDDEN_DIM} | epochs={EPOCHS}"
    )
    train_losses, test_losses, test_metrics, preds, targets = train_surrogate(
        model, train_loader, test_loader, stats, device
    )

    best_design, surrogate_pred_time, pred_history = optimize_design(model, stats, device)
    true_opt_time = calculate_travel_time(build_trajectory(best_design))
    straight_time = calculate_travel_time(straight_line_trajectory())
    true_cycloid = cycloid_time()

    save_training_plot(train_losses, test_losses)
    save_curve_plot(best_design)
    save_prediction_plot(preds, targets)

    gap_pct = 100.0 * (true_opt_time - true_cycloid) / true_cycloid
    summary = {
        "train_config": {
            "model": "mlp",
            "hidden_dim": HIDDEN_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        },
        "optimization_config": {
            "steps": OPT_STEPS,
            "learning_rate": OPT_LR,
            "monotonic_weight": MONOTONIC_WEIGHT,
            "smoothness_weight": SMOOTHNESS_WEIGHT,
        },
        "test_metrics": test_metrics,
        "times": {
            "straight_line_time": straight_time,
            "surrogate_predicted_opt_time": surrogate_pred_time,
            "surrogate_opt_true_time": true_opt_time,
            "cycloid_time": true_cycloid,
            "true_gap_percent_vs_cycloid": gap_pct,
        },
        "paths": {
            "loss_plot": LOSS_PLOT_PATH,
            "curve_plot": CURVE_PLOT_PATH,
            "prediction_plot": PREDICTION_PLOT_PATH,
        },
        "optimized_design": best_design.tolist(),
        "optimization_trace": pred_history,
    }
    save_summary(summary)

    print("---")
    print(f"test_r2:                     {test_metrics['r2']:.6f}")
    print(f"test_mae:                    {test_metrics['mae']:.6f}")
    print(f"test_mse:                    {test_metrics['mse']:.6f}")
    print(f"straight_line_time:          {straight_time:.6f}")
    print(f"surrogate_predicted_opt:     {surrogate_pred_time:.6f}")
    print(f"surrogate_opt_true_time:     {true_opt_time:.6f}")
    print(f"cycloid_time:                {true_cycloid:.6f}")
    print(f"gap_vs_cycloid_percent:      {gap_pct:.3f}")
    print(f"loss_plot:                   {LOSS_PLOT_PATH}")
    print(f"curve_plot:                  {CURVE_PLOT_PATH}")
    print(f"prediction_plot:             {PREDICTION_PLOT_PATH}")
    print(f"summary_json:                {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
