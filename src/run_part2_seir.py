#!/usr/bin/env python3
"""Part 2: SEIR dengue prediction using climate-driven mosquito formula.

Pipeline:
  1. Load M_hat formula from Part 1 (PySR best equation)
  2. SEIR model with Briere beta'(T) and formula-based M_hat
  3. Differential evolution to optimize (c, T_min, T_max, eta) on Guangzhou
  4. Transfer to 16 cities
  5. Evaluation: BI-group vs non-BI-group comparison
  6. Auxiliary: formula M_hat vs observed BI (where available)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
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
PART1_OUT = ROOT / "results" / "part1_mosquito"
OUT = ROOT / "results" / "part2_seir"

# ── SEIR constants ────────────────────────────────────────────────────────
SIGMA_H = 1.0 / 5.9       # latent -> infectious (day^-1)
GAMMA   = 1.0 / 14.0      # recovery rate (day^-1)
DAYS_PER_MONTH = 30

# ── PLACEHOLDER sections ─────────────────────────────────────────────────

# ── Load formula from Part 1 ──────────────────────────────────────────────

def load_formula():
    """Load PySR model from Part 1 output. Returns a callable f(T, H, R) -> M_hat.

    Falls back to a simple placeholder if PySR pickle not found.
    """
    pkl_candidates = list(PART1_OUT.glob("*.pkl"))
    # PySR saves model as hall_of_fame*.pkl or similar
    sr_pkl = None
    for p in pkl_candidates:
        if "hall_of_fame" in p.name or "sr" in p.name.lower():
            sr_pkl = p
            break
    if sr_pkl is None and pkl_candidates:
        sr_pkl = pkl_candidates[0]

    if sr_pkl is not None:
        import pickle
        with open(sr_pkl, "rb") as f:
            sr_model = pickle.load(f)
        print(f"Loaded PySR model from {sr_pkl}")
        return sr_model

    # Fallback: try to load sympy expression from best_formula.txt
    formula_file = PART1_OUT / "best_formula.txt"
    if formula_file.exists():
        import sympy
        text = formula_file.read_text()
        # First line: "Sympy:  <expr>"
        for line in text.splitlines():
            if line.startswith("Sympy:"):
                expr_str = line.split(":", 1)[1].strip()
                T, H, R = sympy.symbols("T H R")
                expr = sympy.sympify(expr_str)
                f_numpy = sympy.lambdify((T, H, R), expr, modules="numpy")
                print(f"Loaded formula from text: {expr_str}")

                class FormulaWrapper:
                    def __init__(self, fn, expr_s):
                        self._fn = fn
                        self._expr = expr_s
                    def predict(self, X):
                        return np.maximum(self._fn(X[:, 0], X[:, 1], X[:, 2]), 0.0)
                    def __repr__(self):
                        return f"Formula({self._expr})"

                return FormulaWrapper(f_numpy, expr_str)

    raise FileNotFoundError(
        f"No PySR model found in {PART1_OUT}. Run Part 1 first."
    )

# ── Briere temperature-dependent transmission rate ────────────────────────

def briere(T: np.ndarray, c: float, T_min: float, T_max: float) -> np.ndarray:
    """Briere function: beta'(T) = c * T * (T - T_min) * sqrt(T_max - T).

    Returns 0 outside [T_min, T_max].
    """
    T = np.asarray(T, dtype=np.float64)
    result = np.zeros_like(T)
    valid = (T > T_min) & (T < T_max)
    Tv = T[valid]
    result[valid] = c * Tv * (Tv - T_min) * np.sqrt(T_max - Tv)
    return np.maximum(result, 0.0)

# ── SEIR forward simulation ───────────────────────────────────────────────

def seir_simulate(tem: np.ndarray, m_hat: np.ndarray,
                  c: float, T_min: float, T_max: float, eta: float,
                  N_h: float, days_per_step: int = DAYS_PER_MONTH) -> np.ndarray:
    """Forward SEIR simulation with Briere beta'(T) and formula M_hat.

    Returns predicted monthly new cases array.
    """
    n_steps = len(tem)
    beta_prime = briere(tem, c, T_min, T_max)

    # Initial state (proportions)
    s = 1.0 - 1.0 / N_h
    e = 0.0
    i_state = 1.0 / N_h
    r = 0.0

    cases_pred = np.zeros(n_steps)

    for t in range(n_steps):
        monthly_cases = 0.0
        bp = float(beta_prime[t])
        mt = float(m_hat[t])

        for _ in range(days_per_step):
            # Force of infection
            lam = bp * mt * i_state + eta / N_h

            new_exposed = lam * s
            ds = -new_exposed
            de = new_exposed - SIGMA_H * e
            di = SIGMA_H * max(e, 0) - GAMMA * i_state
            dr = GAMMA * max(i_state, 0)

            # Accumulate new infectious (E -> I transitions)
            monthly_cases += SIGMA_H * max(e, 0) * N_h

            s = max(s + ds, 0.0)
            e = max(e + de, 0.0)
            i_state = max(i_state + di, 0.0)
            r = r + dr

        cases_pred[t] = monthly_cases

    return cases_pred

# ── Optimization objective ────────────────────────────────────────────────

def objective(params: np.ndarray, tem: np.ndarray, cases_obs: np.ndarray,
              m_hat: np.ndarray, N_h: float, train_mask: np.ndarray) -> float:
    """Objective for differential evolution: MSE(log1p(pred), log1p(obs)) on train set."""
    c, T_min, T_max, eta = params

    # Sanity: T_min must be < T_max
    if T_min >= T_max - 1.0:
        return 1e12

    try:
        pred = seir_simulate(tem, m_hat, c, T_min, T_max, eta, N_h)
    except Exception:
        return 1e12

    obs_tr = cases_obs[train_mask]
    pred_tr = pred[train_mask]

    # log1p MSE
    lo = np.log1p(obs_tr)
    lp = np.log1p(np.maximum(pred_tr, 0.0))
    mse = float(np.mean((lo - lp) ** 2))

    # Penalize if all predictions are zero
    if np.sum(pred_tr) < 1.0:
        mse += 100.0

    return mse

# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_cases(obs: np.ndarray, pred: np.ndarray) -> Dict:
    """Compute case prediction metrics."""
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    valid = np.isfinite(o) & np.isfinite(p)
    o, p = o[valid], p[valid]
    if len(o) < 3 or np.std(o) == 0 or np.std(p) == 0:
        return {k: np.nan for k in ["pearson_r", "spearman_rho", "r2_log",
                                     "mae", "rmse", "wape", "rmsle"]}
    lo, lp = np.log1p(o), np.log1p(p)
    return {
        "pearson_r": float(pearsonr(o, p)[0]),
        "spearman_rho": float(spearmanr(o, p)[0]),
        "r2_log": float(1 - np.sum((lo - lp)**2) / (np.sum((lo - lo.mean())**2) + 1e-12)),
        "mae": float(np.mean(np.abs(o - p))),
        "rmse": float(np.sqrt(np.mean((o - p)**2))),
        "wape": float(np.sum(np.abs(o - p)) / (np.sum(np.abs(o)) + 1e-12)),
        "rmsle": float(np.sqrt(np.mean((lo - lp)**2))),
    }

# ── Plotting ──────────────────────────────────────────────────────────────

def plot_guangzhou_fit(obs: np.ndarray, pred: np.ndarray,
                       years: np.ndarray, months: np.ndarray,
                       metrics: Dict, path: Path):
    """Guangzhou case fitting: time series + scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    t = np.arange(len(obs))

    ax = axes[0]
    ax.plot(t, obs, "k-", lw=1.2, label="Observed")
    ax.plot(t, pred, "r-", lw=0.9, alpha=0.8, label="Predicted")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Cases (symlog)")
    ax.set_title(f"(a) Guangzhou  r={metrics['pearson_r']:.3f}  "
                 f"R2_log={metrics['r2_log']:.3f}")
    # Mark 2014
    m14 = years == 2014
    if m14.any():
        ax.axvspan(t[m14][0], t[m14][-1], alpha=0.15, color="red", label="2014")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    lo, lp = np.log1p(obs), np.log1p(pred)
    ax.scatter(lo, lp, s=12, alpha=0.6)
    mx = max(lo.max(), lp.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.set_xlabel("log(1 + observed)")
    ax.set_ylabel("log(1 + predicted)")
    ax.set_title(f"(b) RMSLE={metrics['rmsle']:.3f}")
    ax.grid(alpha=0.2)

    fig.suptitle("Part 2: SEIR fit (Guangzhou, Briere beta'(T) + formula M_hat)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_briere_curve(c: float, T_min: float, T_max: float, path: Path):
    """Plot beta'(T) Briere curve."""
    T_range = np.linspace(0, 45, 500)
    bp = briere(T_range, c, T_min, T_max)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T_range, bp, "b-", lw=2)
    ax.axvline(T_min, color="gray", ls="--", lw=0.8, label=f"T_min={T_min:.1f}")
    ax.axvline(T_max, color="gray", ls="--", lw=0.8, label=f"T_max={T_max:.1f}")
    T_opt = T_range[np.argmax(bp)]
    ax.axvline(T_opt, color="red", ls=":", lw=0.8, label=f"T_opt={T_opt:.1f}")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("beta'(T)")
    ax.set_title(f"Briere function: c={c:.2e}, T_min={T_min:.1f}, T_max={T_max:.1f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_transfer_scatter(annual: pd.DataFrame, has_bi_set: set, path: Path):
    """16-city 2014 annual scatter: obs vs pred, colored by BI group."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    obs = annual["cases_annual"].values
    pred = annual["pred_annual"].values

    # Scatter
    ax = axes[0]
    for _, row in annual.iterrows():
        color = "steelblue" if row["city_en"] in has_bi_set else "coral"
        ax.scatter(np.log1p(row["cases_annual"]), np.log1p(row["pred_annual"]),
                   s=30, alpha=0.7, color=color)
        ax.annotate(row["city_en"][:4],
                    (np.log1p(row["cases_annual"]), np.log1p(row["pred_annual"])),
                    fontsize=6)
    mx = max(np.log1p(obs).max(), np.log1p(pred).max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    rho = spearmanr(obs, pred)[0]
    ax.set_xlabel("log(1 + observed)")
    ax.set_ylabel("log(1 + predicted)")
    ax.set_title(f"(a) 2014 annual  rho={rho:.3f}")
    # Legend
    ax.scatter([], [], s=30, color="steelblue", label="Has BI (8)")
    ax.scatter([], [], s=30, color="coral", label="No BI (8)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Bar chart
    ax = axes[1]
    idx = np.argsort(obs)[::-1]
    cities = annual["city_en"].values[idx]
    colors = ["steelblue" if c in has_bi_set else "coral" for c in cities]
    y_pos = np.arange(len(cities))
    ax.barh(y_pos, np.log1p(obs[idx]), alpha=0.4, color=colors, label="Obs")
    ax.barh(y_pos, np.log1p(pred[idx]), alpha=0.4, color="gray", label="Pred")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities, fontsize=7)
    ax.set_xlabel("log(1 + cases)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle("Part 2: 16-city transfer (2014)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_all_cities_grid(tm: pd.DataFrame, path: Path):
    """4x4 grid of all 16 cities: observed vs predicted time series."""
    cities = sorted(tm["city_en"].unique())
    ncols, nrows = 4, (len(cities) + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes = axes.ravel()

    for i, city in enumerate(cities):
        ax = axes[i]
        cdf = tm[tm["city_en"] == city].sort_values(["year", "month"])
        t = np.arange(len(cdf))
        ax.plot(t, cdf["cases"].values, "k-", lw=1.2, label="Obs")
        ax.plot(t, cdf["pred_cases"].values, "r-", lw=0.9, alpha=0.8, label="Pred")
        ax.set_title(city, fontsize=9)
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("16 cities: observed vs predicted (SEIR + Briere + formula M_hat)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_comparison(met_df: pd.DataFrame, path: Path):
    """Bar chart comparing BI-group vs non-BI-group metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metrics_cols = ["pearson_r", "spearman_rho", "r2_log", "wape"]

    for i, col in enumerate(metrics_cols):
        ax = axes[i]
        grp = met_df.groupby("has_bi")[col].mean()
        labels = ["No BI (8)", "Has BI (8)"]
        vals = [grp.get(False, 0), grp.get(True, 0)]
        bars = ax.bar([0, 1], vals, alpha=0.7, color=["coral", "steelblue"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(col)
        ax.grid(alpha=0.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("BI-group vs non-BI-group: case prediction metrics", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mhat_vs_bi(df_bi_compare: pd.DataFrame, path: Path):
    """Scatter: formula M_hat vs observed BI for cities with BI data."""
    cities = sorted(df_bi_compare["city_en"].unique())
    n_cities = len(cities)
    ncols = min(4, n_cities)
    nrows = (n_cities + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).ravel()

    for i, city in enumerate(cities):
        ax = axes[i]
        cdf = df_bi_compare[df_bi_compare["city_en"] == city]
        ax.scatter(cdf["bi_obs"].values, cdf["m_hat"].values, s=15, alpha=0.7)
        mx = max(cdf["bi_obs"].max(), cdf["m_hat"].max()) * 1.1
        ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        r_val = pearsonr(cdf["bi_obs"].values, cdf["m_hat"].values)[0] \
            if len(cdf) >= 3 and np.std(cdf["bi_obs"].values) > 0 else np.nan
        ax.set_title(f"{city} (n={len(cdf)}, r={r_val:.2f})", fontsize=9)
        ax.set_xlabel("Observed BI")
        ax.set_ylabel("Formula M_hat")
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Auxiliary: formula M_hat vs observed BI (where available)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Part 2: SEIR dengue prediction with Briere beta'(T) + formula M_hat")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading data ...")
    weather = pd.read_csv(DATA / "cases_weather_monthly_utf8.csv")
    bi = pd.read_csv(DATA / "bi_guangdong_monthly_proxy.csv")
    bi_cities = set(bi["city_en"].unique())
    all_cities = sorted(weather["city_en"].unique())
    print(f"  {len(all_cities)} cities, BI available for: {sorted(bi_cities)}")

    # ── Load formula ──────────────────────────────────────────────────
    print("\nLoading M_hat formula from Part 1 ...")
    formula_model = load_formula()
    print(f"  Formula: {formula_model}")

    # ── Guangzhou: compute M_hat and optimize SEIR params ─────────────
    print("\nOptimizing SEIR parameters on Guangzhou ...")
    gz = weather[weather["city_en"] == "Guangzhou"].sort_values(["year", "month"]).copy()
    gz = gz.reset_index(drop=True)

    # Compute M_hat from formula
    X_gz = gz[["tem", "rhu", "pre"]].values.astype(np.float64)
    m_hat_gz = formula_model.predict(X_gz)
    gz["m_hat"] = m_hat_gz

    tem_gz = gz["tem"].values.astype(np.float64)
    cases_gz = gz["cases"].values.astype(np.float64)

    # N_h: Guangzhou population (can be updated per year if data available)
    N_H_GZ = 1.426e7

    # Train mask: exclude 2014
    train_mask = gz["year"].values != 2014

    # Differential evolution bounds: [c, T_min, T_max, eta]
    bounds = [
        (1e-7, 1e-3),    # c
        (8.0, 18.0),     # T_min
        (33.0, 42.0),    # T_max
        (0.001, 2.0),    # eta
    ]

    print("  Running differential evolution (4D) ...")
    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(tem_gz, cases_gz, m_hat_gz, N_H_GZ, train_mask),
        maxiter=500,
        seed=42,
        tol=1e-8,
        polish=True,
        disp=True,
    )

    c_opt, T_min_opt, T_max_opt, eta_opt = result.x
    print(f"  Optimal: c={c_opt:.2e}, T_min={T_min_opt:.1f}, "
          f"T_max={T_max_opt:.1f}, eta={eta_opt:.4f}")
    print(f"  DE loss={result.fun:.6f}")

    # Guangzhou prediction with optimal params
    pred_gz = seir_simulate(tem_gz, m_hat_gz, c_opt, T_min_opt, T_max_opt, eta_opt, N_H_GZ)
    gz_metrics = evaluate_cases(cases_gz, pred_gz)
    gz_metrics_train = evaluate_cases(cases_gz[train_mask], pred_gz[train_mask])
    gz_metrics_2014 = evaluate_cases(
        cases_gz[~train_mask], pred_gz[~train_mask]
    )

    print(f"  Guangzhou train: r={gz_metrics_train['pearson_r']:.3f}, "
          f"R2_log={gz_metrics_train['r2_log']:.3f}")
    print(f"  Guangzhou 2014:  r={gz_metrics_2014['pearson_r']:.3f}, "
          f"R2_log={gz_metrics_2014['r2_log']:.3f}")
    print(f"  Guangzhou all:   r={gz_metrics['pearson_r']:.3f}, "
          f"R2_log={gz_metrics['r2_log']:.3f}")

    # Save params
    params_dict = {
        "c": c_opt, "T_min": T_min_opt, "T_max": T_max_opt, "eta": eta_opt,
        "sigma_h": SIGMA_H, "gamma": GAMMA, "N_h_gz": N_H_GZ,
        "de_loss": result.fun,
    }
    params_dict.update({f"gz_{k}": v for k, v in gz_metrics.items()})
    pd.DataFrame([params_dict]).to_csv(OUT / "seir_params.csv", index=False)

    # Save Guangzhou fit
    gz["pred_cases"] = pred_gz
    gz["beta_prime"] = briere(tem_gz, c_opt, T_min_opt, T_max_opt)
    gz[["city_en", "year", "month", "cases", "pred_cases", "m_hat", "beta_prime",
        "tem", "rhu", "pre"]].to_csv(OUT / "guangzhou_fit.csv", index=False)

    # Save Briere curve data
    T_curve = np.linspace(0, 45, 500)
    bp_curve = briere(T_curve, c_opt, T_min_opt, T_max_opt)
    pd.DataFrame({"temperature": T_curve, "beta_prime": bp_curve}).to_csv(
        OUT / "beta_prime_curve.csv", index=False)

    # ── Transfer to 16 cities ─────────────────────────────────────────
    print("\nTransferring to 16 cities ...")
    city_tables = []
    city_metrics = []

    for city in all_cities:
        cdf = weather[weather["city_en"] == city].sort_values(["year", "month"]).copy()
        cdf = cdf.reset_index(drop=True)

        X_city = cdf[["tem", "rhu", "pre"]].values.astype(np.float64)
        m_hat_city = formula_model.predict(X_city)
        cdf["m_hat"] = m_hat_city

        # Use same N_h for all cities (simplification; can be updated per city)
        # TODO: load per-city population if available
        N_h_city = N_H_GZ

        pred_city = seir_simulate(
            cdf["tem"].values.astype(np.float64),
            m_hat_city, c_opt, T_min_opt, T_max_opt, eta_opt, N_h_city
        )
        cdf["pred_cases"] = pred_city
        cdf["beta_prime"] = briere(cdf["tem"].values.astype(np.float64),
                                    c_opt, T_min_opt, T_max_opt)
        city_tables.append(cdf)

        met = evaluate_cases(cdf["cases"].values, pred_city)
        met["city_en"] = city
        met["has_bi"] = city in bi_cities
        city_metrics.append(met)

    transfer_monthly = pd.concat(city_tables, ignore_index=True)
    transfer_monthly.to_csv(OUT / "transfer_16cities.csv", index=False)

    met_df = pd.DataFrame(city_metrics)
    met_df.to_csv(OUT / "transfer_metrics.csv", index=False)

    print("\nPer-city metrics:")
    print(met_df[["city_en", "has_bi", "pearson_r", "spearman_rho",
                   "r2_log", "wape"]].to_string(index=False))

    # Group comparison
    grp = met_df.groupby("has_bi")[["pearson_r", "spearman_rho", "r2_log", "wape"]].mean()
    grp.index = grp.index.map({True: "Has BI (8)", False: "No BI (8)"})
    grp.to_csv(OUT / "transfer_group_analysis.csv")
    print("\nGroup comparison:")
    print(grp.to_string())

    # Annual 2014
    y14 = transfer_monthly[transfer_monthly["year"] == 2014]
    annual = y14.groupby("city_en").agg(
        cases_annual=("cases", "sum"),
        pred_annual=("pred_cases", "sum"),
    ).reset_index()
    annual.to_csv(OUT / "transfer_annual2014.csv", index=False)

    # ── Auxiliary: M_hat vs observed BI ────────────────────────────────
    print("\nAuxiliary: formula M_hat vs observed BI ...")
    bi_sub = bi[["city_en", "year", "month", "index_norm_city"]].copy()
    bi_compare = transfer_monthly.merge(
        bi_sub, on=["city_en", "year", "month"], how="inner"
    )
    bi_compare = bi_compare.rename(columns={"index_norm_city": "bi_obs"})
    bi_compare = bi_compare.dropna(subset=["bi_obs", "m_hat"])

    if len(bi_compare) > 0:
        bi_compare[["city_en", "year", "month", "bi_obs", "m_hat"]].to_csv(
            OUT / "mhat_vs_bi.csv", index=False)
        bi_met = evaluate_cases(bi_compare["bi_obs"].values, bi_compare["m_hat"].values)
        print(f"  M_hat vs BI (all): r={bi_met['pearson_r']:.3f}, "
              f"RMSE={bi_met['rmse']:.4f} (n={len(bi_compare)})")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_guangzhou_fit(cases_gz, pred_gz, gz["year"].values, gz["month"].values,
                       gz_metrics, OUT / "fig_guangzhou_cases.png")
    plot_briere_curve(c_opt, T_min_opt, T_max_opt, OUT / "fig_beta_prime_T.png")
    plot_transfer_scatter(annual, bi_cities, OUT / "fig_transfer_scatter.png")
    plot_all_cities_grid(transfer_monthly, OUT / "fig_all_cities_grid.png")
    plot_group_comparison(met_df, OUT / "fig_group_comparison.png")

    if len(bi_compare) > 0:
        plot_mhat_vs_bi(bi_compare, OUT / "fig_mhat_vs_bi.png")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2 complete!")
    print(f"  Output directory:  {OUT}")
    print(f"  SEIR params:       {OUT / 'seir_params.csv'}")
    print(f"  Guangzhou fit:     {OUT / 'guangzhou_fit.csv'}")
    print(f"  16-city transfer:  {OUT / 'transfer_metrics.csv'}")
    print(f"  Group analysis:    {OUT / 'transfer_group_analysis.csv'}")
    print(f"  Briere curve:      {OUT / 'beta_prime_curve.csv'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
