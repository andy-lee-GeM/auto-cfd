"""
Minimal brachistochrone surrogate baseline.

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
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    DATASET_PATH,
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

BATCH_SIZE = 64
EPOCHS = 320
LEARNING_RATE = 0.0017
SEED = 42
OPT_STEPS = 300
OPT_LR = 0.01
HIDDEN_DIM = 80
NUM_HIDDEN_LAYERS = 1
ACTIVATION = "gelu"
WEIGHT_DECAY = 0.0

SUMMARY_PATH = os.path.join(RESULTS_DIR, "train_summary.json")
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "training_loss.png")
CURVE_PLOT_PATH = os.path.join(RESULTS_DIR, "curve_comparison.png")
PREDICTION_PLOT_PATH = os.path.join(RESULTS_DIR, "test_predictions.png")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_trajectory(interior_y_values: np.ndarray) -> np.ndarray:
    interior_y_values = np.asarray(interior_y_values, dtype=np.float64)
    interior_x_values = np.linspace(X0, X1, interior_y_values.shape[0] + 2, dtype=np.float64)[1:-1]
    x_values = np.concatenate(([X0], interior_x_values, [X1])).astype(np.float64)
    y_values = np.concatenate(([Y0], interior_y_values, [Y1])).astype(np.float64)
    return np.column_stack((x_values, y_values))


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
    x_values = X0 + radius * (theta - np.sin(theta))
    y_values = Y0 - radius * (1.0 - np.cos(theta))
    return x_values, y_values


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


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        if ACTIVATION == "relu":
            activation_factory = nn.ReLU
        elif ACTIVATION == "gelu":
            activation_factory = nn.GELU
        else:
            activation_factory = nn.Tanh

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_factory())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    pred_batches = []
    target_batches = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            losses.append(F.mse_loss(pred, yb).item())
            pred_batches.append(denormalize_time(pred, stats).cpu())
            target_batches.append(denormalize_time(yb, stats).cpu())

    preds = torch.cat(pred_batches).view(-1).numpy()
    targets = torch.cat(target_batches).view(-1).numpy()
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(np.mean(losses)), {"mse": mse, "mae": mae, "r2": r2}, preds, targets


def train_surrogate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[float], list[float], dict[str, float], np.ndarray, np.ndarray, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_losses: list[float] = []
    test_losses: list[float] = []
    best_test_loss = float("inf")
    best_epoch = 1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

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
        test_loss, test_metrics, preds, targets = evaluate_model(model, test_loader, stats, device)
        test_losses.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch:04d} | "
                f"train_mse_norm: {train_losses[-1]:.6f} | "
                f"test_mse_norm: {test_losses[-1]:.6f} | "
                f"test_r2: {test_metrics['r2']:.6f} | "
                f"test_mae: {test_metrics['mae']:.6f}"
            )

    model.load_state_dict(best_state)
    _, final_metrics, preds, targets = evaluate_model(model, test_loader, stats, device)
    return train_losses, test_losses, final_metrics, preds, targets, best_epoch


def clamp_design(y_interior: torch.Tensor) -> torch.Tensor:
    return y_interior.clamp(Y_MIN, Y0)


def optimize_design(
    model: nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    design = torch.tensor(
        straight_line_trajectory()[1:-1, 1],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    design = nn.Parameter(design)
    optimizer = torch.optim.Adam([design], lr=OPT_LR)

    best_pred_time = float("inf")
    best_design = design.detach().cpu().numpy().reshape(-1)

    for step in range(1, OPT_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        pred_norm = model(normalize_design(design, stats))
        pred_time = denormalize_time(pred_norm, stats).mean()
        pred_time.backward()
        optimizer.step()

        with torch.no_grad():
            design.copy_(clamp_design(design))
            current_pred = float(denormalize_time(model(normalize_design(design, stats)), stats).item())

        if current_pred < best_pred_time:
            best_pred_time = current_pred
            best_design = design.detach().cpu().numpy().reshape(-1)

        if step == 1 or step % 100 == 0 or step == OPT_STEPS:
            print(f"opt_step {step:04d} | predicted_time: {current_pred:.6f}")

    return best_design, best_pred_time


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
    base_trajectory = straight_line_trajectory()
    best_trajectory = build_trajectory(best_design)
    x_cyc, y_cyc = cycloid_curve()

    plt.figure(figsize=(7, 5))
    plt.plot(x_cyc, y_cyc, color="black", linewidth=2.0, label="Cycloid reference")
    plt.plot(base_trajectory[:, 0], base_trajectory[:, 1], "--", linewidth=1.5, label="Baseline")
    plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], "o-", linewidth=2.0, label="Surrogate optimum")
    plt.scatter([X0, X1], [Y0, Y1], color="black", zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Baseline vs Surrogate Curve")
    plt.grid(alpha=0.2)
    plt.legend()
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
    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


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
    model = SmallMLP(input_dim).to(device)
    print(
        f"Training surrogate | model=mlp | input_dim={input_dim} | hidden_dim={HIDDEN_DIM} | "
        f"layers={NUM_HIDDEN_LAYERS} | activation={ACTIVATION} | epochs={EPOCHS}"
    )

    train_losses, test_losses, test_metrics, preds, targets, best_epoch = train_surrogate(
        model, train_loader, test_loader, stats, device
    )
    best_design, best_predicted_time = optimize_design(model, stats, device)

    base_true_time = calculate_travel_time(straight_line_trajectory())
    best_true_time = calculate_travel_time(build_trajectory(best_design))
    delta_vs_base_true = base_true_time - best_true_time
    delta_vs_base_true_percent = 100.0 * delta_vs_base_true / base_true_time

    save_training_plot(train_losses, test_losses)
    save_curve_plot(best_design)
    save_prediction_plot(preds, targets)

    summary = {
        "train_config": {
            "model": "mlp",
            "hidden_dim": HIDDEN_DIM,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "activation": ACTIVATION,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "optimization_config": {
            "steps": OPT_STEPS,
            "learning_rate": OPT_LR,
        },
        "test_metrics": test_metrics,
        "times": {
            "base_case_true_time": base_true_time,
            "best_predicted_time": best_predicted_time,
            "best_true_time": best_true_time,
            "delta_vs_base_true": delta_vs_base_true,
            "delta_vs_base_true_percent": delta_vs_base_true_percent,
        },
        "paths": {
            "loss_plot": LOSS_PLOT_PATH,
            "curve_plot": CURVE_PLOT_PATH,
            "prediction_plot": PREDICTION_PLOT_PATH,
        },
        "optimized_design": best_design.tolist(),
    }
    save_summary(summary)

    print("---")
    print(f"test_r2:                     {test_metrics['r2']:.6f}")
    print(f"test_mae:                    {test_metrics['mae']:.6f}")
    print(f"test_mse:                    {test_metrics['mse']:.6f}")
    print(f"base_case_true_time:         {base_true_time:.6f}")
    print(f"best_predicted_time:         {best_predicted_time:.6f}")
    print(f"best_true_time:              {best_true_time:.6f}")
    print(f"delta_vs_base_true:          {delta_vs_base_true:.6f}")
    print(f"delta_vs_base_true_pct:      {delta_vs_base_true_percent:.3f}")
    print(f"loss_plot:                   {LOSS_PLOT_PATH}")
    print(f"curve_plot:                  {CURVE_PLOT_PATH}")
    print(f"prediction_plot:             {PREDICTION_PLOT_PATH}")
    print(f"summary_json:                {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
