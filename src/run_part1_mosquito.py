#!/usr/bin/env python3
"""Part 1: Climate-driven mosquito density formula discovery.

Pipeline:
  1. Merge BI + weather data for 8 cities (306 samples)
  2. Train NN: (T, H, R) -> M_hat (normalized BI)
  3. Knowledge distillation: NN generates 8000 grid points
  4. PySR symbolic regression on distilled data
  5. Leave-One-City-Out CV for NN generalization
  6. Evaluation & plotting
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "results" / "part1_mosquito"

# ── Seed ──────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Data loading & merging ────────────────────────────────────────────────

def load_and_merge() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Merge BI and weather data, return (df, X_raw_6d, y_raw).

    X_raw: (N, 6) — T, H, R, T_mean, H_mean, R_mean  (raw values)
    y_raw: (N,)   — log1p(raw BI)  — preserves cross-city absolute signal
    """
    weather = pd.read_csv(DATA / "cases_weather_monthly_utf8.csv")
    bi = pd.read_csv(DATA / "bi_guangdong_monthly_proxy.csv")

    bi_sub = bi[["city_en", "year", "month", "index_value"]].copy()

    merged = weather.merge(bi_sub, on=["city_en", "year", "month"], how="inner")
    merged = merged.dropna(subset=["tem", "rhu", "pre", "index_value"])
    merged = merged.sort_values(["city_en", "year", "month"]).reset_index(drop=True)

    # City climate means as proxy for city fixed effect
    city_means = merged.groupby("city_en")[["tem", "rhu", "pre"]].transform("mean")
    merged["T_mean"] = city_means["tem"]
    merged["H_mean"] = city_means["rhu"]
    merged["R_mean"] = city_means["pre"]

    X_raw = merged[["tem", "rhu", "pre", "T_mean", "H_mean", "R_mean"]].values.astype(np.float64)
    y_raw = np.log1p(merged["index_value"].values.astype(np.float64))

    print(f"Merged dataset: {len(merged)} samples, {merged['city_en'].nunique()} cities")
    for city in sorted(merged["city_en"].unique()):
        sub = merged[merged["city_en"] == city]
        print(f"  {city:12s}: {len(sub):3d} samples, "
              f"T_mean={sub['T_mean'].iloc[0]:.1f}, BI_mean={sub['index_value'].mean():.1f}")

    return merged, X_raw, y_raw


def normalize_features(X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-Max normalize to [0, 1]. Returns (X_norm, x_min, x_max)."""
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    rng = x_max - x_min
    rng[rng == 0] = 1.0
    X_norm = (X_raw - x_min) / rng
    return X_norm, x_min, x_max

# ── Neural Network ────────────────────────────────────────────────────────

class MosquitoNN(nn.Module):
    """6 -> 64 -> 64 -> 32 -> 1, Softplus + Dropout."""
    def __init__(self, n_input=6, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Softplus(), nn.Dropout(0.1),
            nn.Linear(n_hidden, n_hidden), nn.Softplus(), nn.Dropout(0.1),
            nn.Linear(n_hidden, 32), nn.Softplus(),
            nn.Linear(32, 1), nn.Softplus(),  # output >= 0 (log1p(BI) >= 0)
        )

    def forward(self, x):
        return self.net(x)


def train_nn(X_norm: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
             n_epochs: int = 5000, lr: float = 3e-3) -> Tuple[MosquitoNN, list]:
    """Train NN to predict log1p(BI) from normalized weather + city climate means.

    Loss = Huber(pred, obs) + 0.3 * (1 - pearson_r)
    """
    model = MosquitoNN(n_input=X_norm.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    x_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    mask = torch.tensor(train_mask, dtype=torch.bool)

    huber = nn.HuberLoss(delta=1.0)
    best_loss, best_state, losses = float("inf"), None, []

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()

        pred = model(x_t).squeeze(-1)
        p_tr, o_tr = pred[mask], y_t[mask]

        loss_main = huber(p_tr, o_tr)

        # Pearson correlation loss
        loss_corr = torch.tensor(0.0)
        if p_tr.std() > 1e-6 and o_tr.std() > 1e-6:
            p_z = (p_tr - p_tr.mean()) / (p_tr.std() + 1e-8)
            o_z = (o_tr - o_tr.mean()) / (o_tr.std() + 1e-8)
            r = torch.mean(p_z * o_z)
            loss_corr = 1.0 - r

        loss = loss_main + 0.3 * loss_corr
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 500 == 0:
            print(f"  NN Epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    return model, losses


def nn_predict(model: MosquitoNN, X_norm: np.ndarray) -> np.ndarray:
    """Predict M_hat from normalized weather."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_norm, dtype=torch.float32)).squeeze(-1).numpy()
    return np.maximum(pred, 0.0)

# ── Knowledge distillation ────────────────────────────────────────────────

def distill(model: MosquitoNN, x_min: np.ndarray, x_max: np.ndarray,
            n_grid: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate grid points in 6D raw weather space using quantile-based sampling.

    For T/H/R: uniform grid.  For T_mean/H_mean/R_mean: sample from observed city means.
    Returns (X_raw_grid, X_norm_grid, y_grid).
    """
    # T/H/R: 20-point uniform grid
    n_thr = 20
    T_vals = np.linspace(x_min[0], x_max[0], n_thr)
    H_vals = np.linspace(x_min[1], x_max[1], n_thr)
    R_vals = np.linspace(x_min[2], x_max[2], n_thr)

    # T_mean/H_mean/R_mean: 5 quantile points each (from observed city means)
    n_mean = 5
    Tm_vals = np.linspace(x_min[3], x_max[3], n_mean)
    Hm_vals = np.linspace(x_min[4], x_max[4], n_mean)
    Rm_vals = np.linspace(x_min[5], x_max[5], n_mean)

    # Full grid would be 20^3 * 5^3 = 1M — too large. Use Latin hypercube sampling.
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n_grid):
        t = rng.choice(T_vals)
        h = rng.choice(H_vals)
        r = rng.choice(R_vals)
        tm = rng.choice(Tm_vals)
        hm = rng.choice(Hm_vals)
        rm = rng.choice(Rm_vals)
        rows.append([t, h, r, tm, hm, rm])
    X_raw_grid = np.array(rows)

    # Normalize using same min/max
    rng_scale = x_max - x_min
    rng_scale[rng_scale == 0] = 1.0
    X_norm_grid = (X_raw_grid - x_min) / rng_scale

    y_grid = nn_predict(model, X_norm_grid)

    print(f"Distillation: {len(X_raw_grid)} grid points, "
          f"M_hat range=[{y_grid.min():.4f}, {y_grid.max():.4f}]")
    return X_raw_grid, X_norm_grid, y_grid

# ── PySR symbolic regression (via Julia CLI) ─────────────────────────────

def run_pysr(X_raw: np.ndarray, y: np.ndarray, out_dir: Path):
    """Run SymbolicRegression.jl directly via Julia CLI (bypasses juliacall).
    X_raw is 6D: T, H, R, T_mean, H_mean, R_mean.
    Returns (None, pareto_df, best_pred_on_distill)."""
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "pysr_tmp"
    tmp.mkdir(exist_ok=True)

    # Save data for Julia
    np.savetxt(tmp / "X.csv", X_raw, delimiter=",")
    np.savetxt(tmp / "y.csv", y, delimiter=",")

    julia_script = f'''
using SymbolicRegression, DelimitedFiles

X_raw = readdlm("{tmp}/X.csv", ',', Float64)   # [rows, 6]
y     = vec(readdlm("{tmp}/y.csv", ',', Float64))
X     = permutedims(X_raw)                       # [6, rows]

options = SymbolicRegression.Options(
    binary_operators=[+, -, *, /],
    unary_operators=[exp, cos, safe_sqrt],
    maxsize=25,
    populations=30,
    timeout_in_seconds=300,
)

hall = equation_search(X, y;
    options=options,
    niterations=300,
    variable_names=["T", "H", "R", "Tm", "Hm", "Rm"],
)

# Save hall of fame
dominating = calculate_pareto_frontier(hall)
open("{tmp}/hall_of_fame.csv", "w") do io
    println(io, "complexity,loss,equation")
    for member in dominating
        c = compute_complexity(member, options)
        l = member.loss
        eq = string_tree(member.tree, options)
        println(io, "$c,$l,\\"$eq\\"")
    end
end

# Save best (last) equation text
best = dominating[end]
best_eq = string_tree(best.tree, options)
open("{tmp}/best_formula.txt", "w") do io
    println(io, best_eq)
end

# Evaluate ALL Pareto members on X for later selection
open("{tmp}/pareto_preds.csv", "w") do io
    for (idx, member) in enumerate(dominating)
        preds, ok = eval_tree_array(member.tree, X, options)
        if !ok
            preds = fill(NaN, size(X, 2))
        end
        println(io, join(preds, ","))
    end
end

println("DONE: best = ", best_eq)
'''
    script_path = tmp / "run_sr.jl"
    script_path.write_text(julia_script)

    print("  Running Julia SymbolicRegression (this may take a few minutes) ...")
    env = {**dict(os.environ), "OMP_NUM_THREADS": "4", "JULIA_NUM_THREADS": "4"}
    result = subprocess.run(
        ["julia", "--threads=4", str(script_path)],
        capture_output=True, text=True, timeout=600, env=env,
    )
    if result.returncode != 0:
        print(f"  Julia stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Julia SR failed (exit {result.returncode})")
    print(f"  Julia stdout: {result.stdout.strip()[-200:]}")

    # Parse results
    pareto_df = pd.read_csv(tmp / "hall_of_fame.csv")
    pareto_df.to_csv(out_dir / "formula_candidates.csv", index=False)

    # Load all Pareto predictions on distillation data
    pareto_preds = np.loadtxt(tmp / "pareto_preds.csv", delimiter=",")
    if pareto_preds.ndim == 1:
        pareto_preds = pareto_preds.reshape(1, -1)

    best_eq_str = (tmp / "best_formula.txt").read_text().strip()
    best_pred = pareto_preds[-1]  # last = most complex

    with open(out_dir / "best_formula.txt", "w") as f:
        f.write(f"Equation: {best_eq_str}\n")
        f.write(f"\nPareto front:\n")
        f.write(pareto_df.to_string(index=False))
        f.write("\n")

    print(f"  PySR best equation: {best_eq_str}")
    return None, pareto_df, pareto_preds


# ── Evaluate formula string on data ──────────────────────────────────────────
def _eval_formula(eq_str: str, X_raw: np.ndarray) -> np.ndarray:
    """Evaluate a PySR formula string on raw 6D data (T, H, R, T_mean, H_mean, R_mean).
    Handles safe_sqrt -> np.sqrt, exp -> np.exp, cos -> np.cos."""
    T = X_raw[:, 0]
    H = X_raw[:, 1]
    R = X_raw[:, 2]
    T_mean = X_raw[:, 3]
    H_mean = X_raw[:, 4]
    R_mean = X_raw[:, 5]
    # Also create short aliases matching PySR variable names
    Tm = T_mean
    Hm = H_mean
    Rm = R_mean
    # Replace Julia function names with numpy equivalents
    expr = eq_str.replace("safe_sqrt", "np.sqrt").replace("sqrt", "np.sqrt")
    expr = expr.replace("exp(", "np.exp(").replace("cos(", "np.cos(")
    expr = expr.replace("^", "**")
    try:
        result = eval(expr, {"np": np, "T": T, "H": H, "R": R,
                             "Tm": Tm, "Hm": Hm, "Rm": Rm,
                             "T_mean": T_mean, "H_mean": H_mean, "R_mean": R_mean})
        return np.asarray(result, dtype=float)
    except Exception as e:
        print(f"  WARNING: formula eval failed: {e}")
        print(f"  Formula: {expr}")
        return np.full(len(X_raw), np.nan)


# ── Leave-One-City-Out CV ─────────────────────────────────────────────────

def loco_cv(df_merged: pd.DataFrame, X_norm: np.ndarray, y: np.ndarray,
            cities: list) -> pd.DataFrame:
    """8-fold LOCO CV at NN level. Returns DataFrame with per-city metrics."""
    rows = []
    for test_city in cities:
        mask_train = df_merged["city_en"].values != test_city
        mask_test = ~mask_train

        model_cv, _ = train_nn(X_norm, y, mask_train, n_epochs=2000, lr=5e-3)
        pred_cv = nn_predict(model_cv, X_norm)

        obs_test = y[mask_test]
        pred_test = pred_cv[mask_test]

        met = evaluate(obs_test, pred_test)
        met["test_city"] = test_city
        met["n_test"] = int(mask_test.sum())
        rows.append(met)
        print(f"  LOCO {test_city:12s}: r={met['pearson_r']:.3f}, "
              f"R2={met['r2']:.3f}, RMSE={met['rmse']:.4f} (n={met['n_test']})")

    cv_df = pd.DataFrame(rows)
    print(f"  LOCO mean: r={cv_df['pearson_r'].mean():.3f} +/- {cv_df['pearson_r'].std():.3f}")
    return cv_df

# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(obs: np.ndarray, pred: np.ndarray) -> Dict:
    """Compute regression metrics between observed and predicted."""
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    valid = np.isfinite(o) & np.isfinite(p)
    o, p = o[valid], p[valid]
    if len(o) < 3 or np.std(o) == 0 or np.std(p) == 0:
        return {k: np.nan for k in ["pearson_r", "spearman_rho", "r2", "r2_log",
                                     "mae", "rmse", "rmsle"]}
    lo, lp = np.log1p(o), np.log1p(p)
    return {
        "pearson_r": float(pearsonr(o, p)[0]),
        "spearman_rho": float(spearmanr(o, p)[0]),
        "r2": float(r2_score(o, p)),
        "r2_log": float(1 - np.sum((lo - lp)**2) / (np.sum((lo - lo.mean())**2) + 1e-12)),
        "mae": float(np.mean(np.abs(o - p))),
        "rmse": float(np.sqrt(np.mean((o - p)**2))),
        "rmsle": float(np.sqrt(np.mean((lo - lp)**2))),
    }


def compute_aic_bic(n: int, k: int, ss_res: float) -> Tuple[float, float]:
    """AIC and BIC from residual sum of squares."""
    aic = n * np.log(ss_res / n + 1e-30) + 2 * k
    bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
    return float(aic), float(bic)

# ── Plotting ──────────────────────────────────────────────────────────────

def plot_nn_vs_obs(df_merged: pd.DataFrame, obs: np.ndarray, pred: np.ndarray,
                   metrics: Dict, path: Path):
    """Scatter plot: NN predicted M_hat vs observed BI, colored by city."""
    cities = df_merged["city_en"].values
    unique_cities = sorted(set(cities))
    cmap = plt.cm.get_cmap("tab10", len(unique_cities))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax = axes[0]
    for i, city in enumerate(unique_cities):
        mask = cities == city
        ax.scatter(obs[mask], pred[mask], s=15, alpha=0.7, color=cmap(i), label=city)
    mx = max(obs.max(), pred.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.set_xlabel("Observed BI (normalized)")
    ax.set_ylabel("NN predicted M_hat")
    ax.set_title(f"(a) r={metrics['pearson_r']:.3f}, R2={metrics['r2']:.3f}")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.2)

    # log1p scatter
    ax = axes[1]
    lo, lp = np.log1p(obs), np.log1p(pred)
    for i, city in enumerate(unique_cities):
        mask = cities == city
        ax.scatter(lo[mask], lp[mask], s=15, alpha=0.7, color=cmap(i))
    mx = max(lo.max(), lp.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.set_xlabel("log(1 + observed)")
    ax.set_ylabel("log(1 + predicted)")
    ax.set_title(f"(b) R2_log={metrics['r2_log']:.3f}, RMSLE={metrics['rmsle']:.3f}")
    ax.grid(alpha=0.2)

    fig.suptitle("Part 1: NN mosquito density prediction (8 cities)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_response_surface(model: MosquitoNN, x_min: np.ndarray, x_max: np.ndarray,
                          path: Path):
    """3 panels: M_hat vs T (fix H,R), vs H (fix T,R), vs R (fix T,H)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["Temperature (C)", "Humidity (%)", "Precipitation (mm)"]
    medians = (x_min + x_max) / 2.0

    for dim in range(3):
        ax = axes[dim]
        vals = np.linspace(x_min[dim], x_max[dim], 200)
        X_test = np.tile(medians, (200, 1))
        X_test[:, dim] = vals

        # Normalize
        rng = x_max - x_min
        rng[rng == 0] = 1.0
        X_norm = (X_test - x_min) / rng

        pred = nn_predict(model, X_norm)
        ax.plot(vals, pred, "b-", lw=1.5)
        ax.set_xlabel(labels[dim])
        ax.set_ylabel("M_hat")
        fix_info = ", ".join(f"{labels[j]}={medians[j]:.1f}" for j in range(3) if j != dim)
        ax.set_title(f"Response: {labels[dim]}\n(fix: {fix_info})", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("NN response curves (univariate slices)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_formula_vs_nn(nn_pred: np.ndarray, formula_pred: np.ndarray,
                       obs: np.ndarray, path: Path):
    """Compare formula prediction vs NN prediction and observed values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Formula vs NN
    ax = axes[0]
    ax.scatter(nn_pred, formula_pred, s=12, alpha=0.5, color="steelblue")
    mx = max(nn_pred.max(), formula_pred.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    r_fn = pearsonr(nn_pred, formula_pred)[0] if np.std(formula_pred) > 0 else 0
    ax.set_xlabel("NN prediction")
    ax.set_ylabel("Formula prediction")
    ax.set_title(f"(a) Formula vs NN (r={r_fn:.3f})")
    ax.grid(alpha=0.2)

    # Formula vs observed
    ax = axes[1]
    ax.scatter(obs, formula_pred, s=12, alpha=0.5, color="coral")
    mx = max(obs.max(), formula_pred.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    r_fo = pearsonr(obs, formula_pred)[0] if np.std(formula_pred) > 0 else 0
    ax.set_xlabel("Observed BI (normalized)")
    ax.set_ylabel("Formula prediction")
    ax.set_title(f"(b) Formula vs Observed (r={r_fo:.3f})")
    ax.grid(alpha=0.2)

    fig.suptitle("Part 1: Symbolic regression formula evaluation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loco_cv(cv_df: pd.DataFrame, path: Path):
    """Bar chart of LOCO CV metrics per city."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cities = cv_df["test_city"].values
    x = np.arange(len(cities))

    col_labels = {
        "pearson_r": "Pearson r",
        "rmse": "RMSE (log1p space)",
    }

    for i, col in enumerate(["pearson_r", "rmse"]):
        ax = axes[i]
        vals = cv_df[col].values
        if col == "pearson_r":
            colors = ["#4CAF50" if v > 0.5 else "#FFB74D" if v > 0.3 else "#E57373" for v in vals]
        else:
            colors = "steelblue"
        bars = ax.bar(x, vals, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=35, fontsize=7)
        ax.set_title(col_labels[col], fontsize=10)
        ax.grid(alpha=0.2)
        mean_val = np.nanmean(vals)
        ax.axhline(mean_val, color="red", ls="--", lw=0.8,
                   label=f"mean={mean_val:.3f}")
        ax.legend(fontsize=7)
        for bar, v in zip(bars if hasattr(bars, '__iter__') else [bars], vals):
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, v,
                    f"{v:.2f}", ha="center", va=va, fontsize=6)

    fig.suptitle("Leave-One-City-Out Cross-Validation (NN)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data preparation ──────────────────────────────────────
    print("=" * 60)
    print("Part 1: Climate-driven mosquito density formula discovery")
    print("=" * 60)

    print("\nStep 1: Loading and merging data ...")
    df_merged, X_raw, y_raw = load_and_merge()
    X_norm, x_min, x_max = normalize_features(X_raw)
    cities = sorted(df_merged["city_en"].unique().tolist())

    # ── Step 2: Train NN on all data ──────────────────────────────────
    print("\nStep 2: Training NN on all 8 cities ...")
    train_mask_all = np.ones(len(y_raw), dtype=bool)
    nn_model, losses = train_nn(X_norm, y_raw, train_mask_all, n_epochs=5000, lr=3e-3)
    nn_pred_all = nn_predict(nn_model, X_norm)

    # Evaluate NN on real data
    nn_metrics = evaluate(y_raw, nn_pred_all)
    print(f"  NN on real data: r={nn_metrics['pearson_r']:.3f}, "
          f"R2={nn_metrics['r2']:.3f}, RMSE={nn_metrics['rmse']:.4f}")
    pd.DataFrame([nn_metrics]).to_csv(OUT / "nn_mosquito_metrics.csv", index=False)

    # Save NN training loss
    pd.DataFrame({"epoch": range(len(losses)), "loss": losses}).to_csv(
        OUT / "nn_training_loss.csv", index=False)

    # ── Step 3: Knowledge distillation + PySR ─────────────────────────
    print("\nStep 3a: Knowledge distillation ...")
    X_dist_raw, X_dist_norm, y_dist = distill(nn_model, x_min, x_max, n_grid=10000)

    print("\nStep 3b: PySR symbolic regression on distilled data ...")
    _, pareto_df, pareto_preds = run_pysr(X_dist_raw, y_dist, OUT)

    # ── Step 3c: Select best formula using real-data R² ───────────────
    print("\nStep 3c: Selecting best formula by real-data R² ...")
    best_idx, best_r2_real, best_eq_str = -1, -np.inf, ""
    formula_results = []
    for i, row in pareto_df.iterrows():
        eq_str = row["equation"].strip().strip('"')
        try:
            pred_real = _eval_formula(eq_str, X_raw)
            if np.any(np.isnan(pred_real)) or np.any(np.isinf(pred_real)):
                continue
            met = evaluate(y_raw, pred_real)
            formula_results.append({
                "idx": i, "complexity": row["complexity"],
                "loss_distill": row["loss"], "equation": eq_str,
                "r2_real": met["r2"], "r_real": met["pearson_r"],
                "rmse_real": met["rmse"],
            })
            if met["r2"] > best_r2_real:
                best_r2_real = met["r2"]
                best_idx = i
                best_eq_str = eq_str
        except Exception as e:
            continue

    formula_sel_df = pd.DataFrame(formula_results)
    formula_sel_df.to_csv(OUT / "formula_selection.csv", index=False)

    if best_idx >= 0:
        formula_pred_real = _eval_formula(best_eq_str, X_raw)
        formula_metrics = evaluate(y_raw, formula_pred_real)
        k_formula = int(pareto_df.loc[best_idx, "complexity"])
    else:
        # Fallback: use most complex
        best_eq_str = pareto_df.iloc[-1]["equation"].strip().strip('"')
        formula_pred_real = _eval_formula(best_eq_str, X_raw)
        formula_metrics = evaluate(y_raw, formula_pred_real)
        k_formula = int(pareto_df.iloc[-1]["complexity"])

    n_real = len(y_raw)
    ss_res = float(np.sum((y_raw - formula_pred_real) ** 2))
    aic, bic = compute_aic_bic(n_real, k_formula, ss_res)
    formula_metrics["aic"] = aic
    formula_metrics["bic"] = bic
    formula_metrics["n_real"] = n_real
    formula_metrics["k"] = k_formula

    # Overwrite best_formula.txt with the selected formula
    with open(OUT / "best_formula.txt", "w") as f:
        f.write(f"Equation: {best_eq_str}\n")
        f.write(f"Selected by: max real-data R² = {best_r2_real:.4f}\n")
        f.write(f"Complexity: {k_formula}\n")
        f.write(f"\nAll candidates evaluated on real data:\n")
        f.write(formula_sel_df.to_string(index=False))
        f.write("\n")

    print(f"  Selected formula (complexity={k_formula}): {best_eq_str}")
    print(f"  Formula on real data: r={formula_metrics['pearson_r']:.3f}, "
          f"R2={formula_metrics['r2']:.3f}, RMSE={formula_metrics['rmse']:.4f}, "
          f"AIC={aic:.1f}, BIC={bic:.1f}")
    pd.DataFrame([formula_metrics]).to_csv(OUT / "formula_real_metrics.csv", index=False)

    # ── Step 4: LOCO CV ───────────────────────────────────────────────
    print("\nStep 4: Leave-One-City-Out cross-validation (NN) ...")
    cv_df = loco_cv(df_merged, X_norm, y_raw, cities)
    cv_df.to_csv(OUT / "cv_leave_one_city.csv", index=False)

    # ── Step 5: Plots ─────────────────────────────────────────────────
    print("\nStep 5: Generating plots ...")
    plot_nn_vs_obs(df_merged, y_raw, nn_pred_all, nn_metrics,
                   OUT / "fig_nn_vs_obs.png")
    plot_response_surface(nn_model, x_min, x_max,
                          OUT / "fig_response_surface.png")
    plot_formula_vs_nn(nn_pred_all, formula_pred_real, y_raw,
                       OUT / "fig_formula_vs_nn.png")
    plot_loco_cv(cv_df, OUT / "fig_loco_cv.png")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 1 complete!")
    print(f"  Output directory: {OUT}")
    print(f"  NN metrics:      {OUT / 'nn_mosquito_metrics.csv'}")
    print(f"  Best formula:    {OUT / 'best_formula.txt'}")
    print(f"  Formula metrics: {OUT / 'formula_real_metrics.csv'}")
    print(f"  LOCO CV:         {OUT / 'cv_leave_one_city.csv'}")
    print(f"  Pareto front:    {OUT / 'formula_candidates.csv'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
