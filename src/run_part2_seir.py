#!/usr/bin/env python3
"""Part 2: SEIR dengue prediction using climate-driven mosquito formula.

Pipeline:
  1. Load M_hat formula from Part 1 → expm1 transform → normalize
  2. SEIR model with Briere beta'(T) and formula-based M_hat
  3. DE optimize (c, T_min, T_max, eta) on Guangzhou (Huber + correlation)
  4. Per-city log-linear scaling calibration (leave-Guangzhou-out)
  5. Transfer to 16 cities with per-city population
  6. Leave-one-year-out CV on Guangzhou
  7. Evaluation & plots
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
SIGMA_H = 1.0 / 5.9
GAMMA = 1.0 / 14.0
DAYS_PER_MONTH = 30

# ── Per-city population (2012 mid-period estimates, 万人 → people) ───────
CITY_POP = {
    "Guangzhou": 1426e4, "Shenzhen": 1054e4, "Dongguan": 831e4,
    "Foshan": 729e4, "Huizhou": 470e4, "Zhongshan": 318e4,
    "Zhuhai": 158e4, "Jiangmen": 451e4, "Zhaoqing": 402e4,
    "Shantou": 539e4, "Chaozhou": 267e4, "Jieyang": 606e4,
    "Zhanjiang": 724e4, "Maoming": 605e4, "Yangjiang": 249e4,
    "Qingyuan": 386e4,
}

# ── Load formula from Part 1 ──────────────────────────────────────────────

def load_formula():
    """Load PySR formula. Returns callable with .predict(X_6d) -> M_hat (log1p space)."""
    formula_file = PART1_OUT / "best_formula.txt"
    if not formula_file.exists():
        raise FileNotFoundError(f"No formula at {formula_file}. Run Part 1 first.")

    text = formula_file.read_text()
    eq_str = None
    for line in text.splitlines():
        if line.startswith("Equation:"):
            eq_str = line.split(":", 1)[1].strip()
            break
    if eq_str is None:
        raise ValueError(f"Could not parse equation from {formula_file}")

    print(f"  Loaded formula: {eq_str[:100]}...")

    class FormulaWrapper:
        def __init__(self, eq):
            self._eq = eq

        def predict(self, X):
            T, H, R = X[:, 0], X[:, 1], X[:, 2]
            Tm, Hm, Rm = X[:, 3], X[:, 4], X[:, 5]
            T_mean, H_mean, R_mean = Tm, Hm, Rm
            expr = self._eq.replace("safe_sqrt", "np.sqrt").replace("sqrt", "np.sqrt")
            expr = expr.replace("exp(", "np.exp(").replace("cos(", "np.cos(")
            expr = expr.replace("^", "**")
            try:
                result = eval(expr, {"np": np, "T": T, "H": H, "R": R,
                                     "Tm": Tm, "Hm": Hm, "Rm": Rm,
                                     "T_mean": T_mean, "H_mean": H_mean, "R_mean": R_mean})
                return np.maximum(np.asarray(result, dtype=float), 0.0)
            except Exception as e:
                print(f"  WARNING: formula eval failed: {e}")
                return np.full(len(X), 0.0)

        def predict_bi(self, X):
            """Predict in BI space: expm1(log1p-space prediction), clipped to [0, inf)."""
            log1p_pred = self.predict(X)
            return np.expm1(np.clip(log1p_pred, 0, 10))

        def __repr__(self):
            return f"Formula({self._eq[:60]}...)"

    return FormulaWrapper(eq_str)


def compute_city_features(cdf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 6D features and M_hat (BI-space) for a city DataFrame."""
    means = cdf[["tem", "rhu", "pre"]].mean()
    cdf = cdf.copy()
    cdf["Tm"], cdf["Hm"], cdf["Rm"] = means["tem"], means["rhu"], means["pre"]
    X = cdf[["tem", "rhu", "pre", "Tm", "Hm", "Rm"]].values.astype(np.float64)
    return X, cdf


# ── Briere ────────────────────────────────────────────────────────────────

def briere(T: np.ndarray, c: float, T_min: float, T_max: float) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    result = np.zeros_like(T)
    valid = (T > T_min) & (T < T_max)
    Tv = T[valid]
    result[valid] = c * Tv * (Tv - T_min) * np.sqrt(T_max - Tv)
    return np.maximum(result, 0.0)


# ── SEIR forward simulation ──────────────────────────────────────────────

def seir_simulate(tem: np.ndarray, m_hat: np.ndarray,
                  c: float, T_min: float, T_max: float, eta: float,
                  N_h: float) -> np.ndarray:
    n_steps = len(tem)
    beta_prime = briere(tem, c, T_min, T_max)

    s = 1.0 - 1e-6
    e = 0.0
    i_state = 1e-6
    r = 0.0

    cases_pred = np.zeros(n_steps)

    for t in range(n_steps):
        monthly_cases = 0.0
        bp = float(beta_prime[t])
        mt = float(m_hat[t])

        for _ in range(DAYS_PER_MONTH):
            lam = bp * mt * i_state + eta / N_h
            new_exposed = lam * s
            monthly_cases += SIGMA_H * max(e, 0) * N_h

            s = max(s - new_exposed, 0.0)
            e = max(e + new_exposed - SIGMA_H * e, 0.0)
            i_state = max(i_state + SIGMA_H * e - GAMMA * i_state, 0.0)
            r = r + GAMMA * i_state

        cases_pred[t] = monthly_cases

    return cases_pred


# ── Optimization objective (improved) ────────────────────────────────────

def objective(params, tem, cases_obs, m_hat, N_h, train_mask):
    c, T_min, T_max, eta = params
    if T_min >= T_max - 1.0:
        return 1e12
    try:
        pred = seir_simulate(tem, m_hat, c, T_min, T_max, eta, N_h)
    except Exception:
        return 1e12

    obs_tr = cases_obs[train_mask]
    pred_tr = np.maximum(pred[train_mask], 0.0)

    lo = np.log1p(obs_tr)
    lp = np.log1p(pred_tr)

    huber_delta = 1.5
    diff = lo - lp
    abs_diff = np.abs(diff)
    huber = np.where(abs_diff <= huber_delta,
                     0.5 * diff**2,
                     huber_delta * (abs_diff - 0.5 * huber_delta))
    loss_main = float(np.mean(huber))

    # Correlation penalty
    loss_corr = 0.0
    if np.std(pred_tr) > 1e-8 and np.std(obs_tr) > 1e-8:
        rho = spearmanr(obs_tr, pred_tr)[0]
        if np.isfinite(rho):
            loss_corr = 1.0 - rho

    if np.sum(pred_tr) < 1.0:
        loss_main += 100.0

    return loss_main + 0.3 * loss_corr


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_cases(obs: np.ndarray, pred: np.ndarray) -> Dict:
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

def plot_guangzhou_fit(obs, pred, years, months, metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    t = np.arange(len(obs))
    ax = axes[0]
    ax.plot(t, obs, "k-", lw=1.2, label="Observed")
    ax.plot(t, pred, "r-", lw=0.9, alpha=0.8, label="Predicted")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Cases (symlog)")
    ax.set_title(f"(a) Guangzhou  ρ={metrics['spearman_rho']:.3f}  R²_log={metrics['r2_log']:.3f}")
    m14 = years == 2014
    if m14.any():
        ax.axvspan(t[m14][0], t[m14][-1], alpha=0.15, color="red", label="2014")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    lo, lp = np.log1p(obs), np.log1p(pred)
    ax.scatter(lo, lp, s=12, alpha=0.6)
    mx = max(lo.max(), lp.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.set_xlabel("log(1+obs)"); ax.set_ylabel("log(1+pred)")
    ax.set_title(f"(b) RMSLE={metrics['rmsle']:.3f}"); ax.grid(alpha=0.2)

    fig.suptitle("Part 2: SEIR + Briere + formula M̂ (Guangzhou)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=200); plt.close(fig)


def plot_briere_curve(c, T_min, T_max, path):
    T_range = np.linspace(0, 45, 500)
    bp = briere(T_range, c, T_min, T_max)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T_range, bp, "b-", lw=2)
    ax.axvline(T_min, color="gray", ls="--", lw=0.8, label=f"T_min={T_min:.1f}")
    ax.axvline(T_max, color="gray", ls="--", lw=0.8, label=f"T_max={T_max:.1f}")
    T_opt = T_range[np.argmax(bp)]
    ax.axvline(T_opt, color="red", ls=":", lw=0.8, label=f"T_opt={T_opt:.1f}")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("β'(T)")
    ax.set_title(f"Brière: c={c:.2e}, T_min={T_min:.1f}°C, T_max={T_max:.1f}°C")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_transfer_scatter(annual, has_bi_set, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    obs, pred = annual["cases_annual"].values, annual["pred_annual"].values
    ax = axes[0]
    for _, row in annual.iterrows():
        color = "steelblue" if row["city_en"] in has_bi_set else "coral"
        ax.scatter(np.log1p(row["cases_annual"]), np.log1p(row["pred_annual"]),
                   s=30, alpha=0.7, color=color)
        ax.annotate(row["city_en"][:4],
                    (np.log1p(row["cases_annual"]), np.log1p(row["pred_annual"])), fontsize=6)
    mx = max(np.log1p(obs).max(), np.log1p(pred).max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    rho = spearmanr(obs, pred)[0] if len(obs) >= 3 else 0
    ax.set_xlabel("log(1+obs)"); ax.set_ylabel("log(1+pred)")
    ax.set_title(f"(a) 2014 annual ρ={rho:.3f}")
    ax.scatter([], [], s=30, color="steelblue", label="Has BI (8)")
    ax.scatter([], [], s=30, color="coral", label="No BI (8)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    idx = np.argsort(obs)[::-1]
    cities = annual["city_en"].values[idx]
    colors = ["steelblue" if c in has_bi_set else "coral" for c in cities]
    y_pos = np.arange(len(cities))
    ax.barh(y_pos, np.log1p(obs[idx]), alpha=0.4, color=colors, label="Obs")
    ax.barh(y_pos, np.log1p(pred[idx]), alpha=0.4, color="gray", label="Pred")
    ax.set_yticks(y_pos); ax.set_yticklabels(cities, fontsize=7)
    ax.set_xlabel("log(1+cases)"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    fig.suptitle("Part 2: 16-city transfer (2014 annual)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=200); plt.close(fig)


def plot_all_cities_grid(tm, path):
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
        ax.set_title(city, fontsize=9); ax.set_yscale("symlog", linthresh=1); ax.grid(alpha=0.2)
        if i == 0: ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("16 cities: observed vs predicted (scaled SEIR)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=180); plt.close(fig)


def plot_group_comparison(met_df, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for i, col in enumerate(["pearson_r", "spearman_rho", "r2_log", "wape"]):
        ax = axes[i]
        grp = met_df.groupby("has_bi")[col].mean()
        labels, vals = ["No BI (8)", "Has BI (8)"], [grp.get(False, 0), grp.get(True, 0)]
        bars = ax.bar([0, 1], vals, alpha=0.7, color=["coral", "steelblue"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(col); ax.grid(alpha=0.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("BI-group vs non-BI-group metrics", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=200); plt.close(fig)


def plot_mhat_vs_bi(df_bi_compare, path):
    cities = sorted(df_bi_compare["city_en"].unique())
    ncols = min(4, len(cities))
    nrows = (len(cities) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes = np.atleast_2d(axes).ravel()
    for i, city in enumerate(cities):
        ax = axes[i]
        cdf = df_bi_compare[df_bi_compare["city_en"] == city]
        ax.scatter(cdf["bi_obs"].values, cdf["m_hat_bi"].values, s=15, alpha=0.7)
        mx = max(cdf["bi_obs"].max(), cdf["m_hat_bi"].max()) * 1.1
        ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        r_val = pearsonr(cdf["bi_obs"].values, cdf["m_hat_bi"].values)[0] \
            if len(cdf) >= 3 and np.std(cdf["bi_obs"].values) > 0 else np.nan
        ax.set_title(f"{city} (n={len(cdf)}, r={r_val:.2f})", fontsize=9)
        ax.set_xlabel("Observed BI"); ax.set_ylabel("Formula M̂ (BI scale)"); ax.grid(alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Formula M̂ vs observed BI (BI space)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=200); plt.close(fig)


def plot_loyo_cv(cv_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    years = cv_df["year"].values
    for i, col in enumerate(["spearman_rho", "r2_log", "rmsle"]):
        ax = axes[i]
        vals = cv_df[col].values
        ax.bar(np.arange(len(years)), vals, alpha=0.7, color="steelblue")
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years.astype(int), rotation=45, fontsize=7)
        ax.set_title(col); ax.grid(alpha=0.2)
        mean_v = np.nanmean(vals)
        ax.axhline(mean_v, color="red", ls="--", lw=0.8, label=f"mean={mean_v:.3f}")
        ax.legend(fontsize=7)
    fig.suptitle("Leave-One-Year-Out CV (Guangzhou)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=200); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Part 2: SEIR dengue prediction (optimized)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading data ...")
    weather = pd.read_csv(DATA / "cases_weather_monthly_utf8.csv")
    bi = pd.read_csv(DATA / "bi_guangdong_monthly_proxy.csv")
    bi_cities = set(bi["city_en"].unique())
    all_cities = sorted(weather["city_en"].unique())
    print(f"  {len(all_cities)} cities, BI available for: {sorted(bi_cities)}")

    # ── Load formula ──────────────────────────────────────────────────
    print("\nLoading M̂ formula from Part 1 ...")
    formula_model = load_formula()

    # ── Guangzhou: compute M_hat and optimize SEIR params ─────────────
    print("\nOptimizing SEIR parameters on Guangzhou ...")
    gz = weather[weather["city_en"] == "Guangzhou"].sort_values(["year", "month"]).copy()
    gz = gz.reset_index(drop=True)

    X_gz, gz = compute_city_features(gz)
    m_hat_bi_gz = formula_model.predict_bi(X_gz)
    m_hat_norm_gz = m_hat_bi_gz / (np.mean(m_hat_bi_gz) + 1e-8)
    gz["m_hat_bi"] = m_hat_bi_gz
    gz["m_hat_norm"] = m_hat_norm_gz

    tem_gz = gz["tem"].values.astype(np.float64)
    cases_gz = gz["cases"].values.astype(np.float64)
    N_H_GZ = CITY_POP["Guangzhou"]

    train_mask = gz["year"].values != 2014

    bounds = [
        (1e-8, 1e-2),
        (10.0, 20.0),
        (32.0, 42.0),
        (0.001, 5.0),
    ]

    print("  Running differential evolution (4D, Huber+Spearman) ...")
    result = differential_evolution(
        objective, bounds=bounds,
        args=(tem_gz, cases_gz, m_hat_norm_gz, N_H_GZ, train_mask),
        maxiter=800, seed=42, tol=1e-9, polish=True, workers=1,
    )

    c_opt, T_min_opt, T_max_opt, eta_opt = result.x
    T_opt = np.linspace(T_min_opt, T_max_opt, 1000)
    T_opt_val = T_opt[np.argmax(briere(T_opt, c_opt, T_min_opt, T_max_opt))]
    print(f"  Optimal: c={c_opt:.2e}, T_min={T_min_opt:.1f}, T_max={T_max_opt:.1f}, "
          f"eta={eta_opt:.4f}, T_opt={T_opt_val:.1f}")

    pred_gz = seir_simulate(tem_gz, m_hat_norm_gz, c_opt, T_min_opt, T_max_opt, eta_opt, N_H_GZ)
    gz_met_all = evaluate_cases(cases_gz, pred_gz)
    gz_met_train = evaluate_cases(cases_gz[train_mask], pred_gz[train_mask])
    gz_met_2014 = evaluate_cases(cases_gz[~train_mask], pred_gz[~train_mask])

    print(f"  GZ train:  ρ={gz_met_train['spearman_rho']:.3f}, R²_log={gz_met_train['r2_log']:.3f}")
    print(f"  GZ 2014:   ρ={gz_met_2014['spearman_rho']:.3f}, R²_log={gz_met_2014['r2_log']:.3f}")
    print(f"  GZ all:    ρ={gz_met_all['spearman_rho']:.3f}, R²_log={gz_met_all['r2_log']:.3f}")

    pd.DataFrame([{
        "c": c_opt, "T_min": T_min_opt, "T_max": T_max_opt, "T_opt": T_opt_val,
        "eta": eta_opt, "de_loss": result.fun,
        **{f"gz_{k}": v for k, v in gz_met_all.items()},
    }]).to_csv(OUT / "seir_params.csv", index=False)

    gz["pred_cases"] = pred_gz
    gz["beta_prime"] = briere(tem_gz, c_opt, T_min_opt, T_max_opt)
    gz.to_csv(OUT / "guangzhou_fit.csv", index=False)

    # ── Leave-One-Year-Out CV on Guangzhou ────────────────────────────
    print("\nLeave-One-Year-Out CV on Guangzhou ...")
    cv_rows = []
    for test_year in sorted(gz["year"].unique()):
        cv_train = gz["year"].values != test_year
        cv_res = differential_evolution(
            objective, bounds=bounds,
            args=(tem_gz, cases_gz, m_hat_norm_gz, N_H_GZ, cv_train),
            maxiter=300, seed=42, tol=1e-8, polish=True, workers=1,
        )
        cv_pred = seir_simulate(tem_gz, m_hat_norm_gz, *cv_res.x, N_H_GZ)
        cv_test_mask = ~cv_train
        if cv_test_mask.sum() > 0:
            met = evaluate_cases(cases_gz[cv_test_mask], cv_pred[cv_test_mask])
            met["year"] = test_year
            cv_rows.append(met)
            print(f"  {test_year}: ρ={met['spearman_rho']:.3f}, R²_log={met['r2_log']:.3f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT / "loyo_cv.csv", index=False)
    print(f"  LOYO mean ρ={cv_df['spearman_rho'].mean():.3f} ± {cv_df['spearman_rho'].std():.3f}")

    # ── Transfer to 16 cities with per-city population + scaling ──────
    print("\nTransferring to 16 cities (per-city pop + log-linear scaling) ...")

    # Step 1: raw SEIR predictions for all cities
    city_raw = {}
    for city in all_cities:
        cdf = weather[weather["city_en"] == city].sort_values(["year", "month"]).copy()
        cdf = cdf.reset_index(drop=True)
        X_city, cdf = compute_city_features(cdf)
        m_hat_bi = formula_model.predict_bi(X_city)
        m_hat_norm = m_hat_bi / (np.mean(m_hat_bi) + 1e-8)
        N_h = CITY_POP.get(city, N_H_GZ)
        pred_raw = seir_simulate(
            cdf["tem"].values.astype(np.float64),
            m_hat_norm, c_opt, T_min_opt, T_max_opt, eta_opt, N_h
        )
        cdf["m_hat_bi"] = m_hat_bi
        cdf["m_hat_norm"] = m_hat_norm
        cdf["pred_raw"] = pred_raw
        cdf["beta_prime"] = briere(cdf["tem"].values.astype(np.float64), c_opt, T_min_opt, T_max_opt)
        city_raw[city] = cdf

    # Step 2: log-linear scaling (fit on non-2014, non-Guangzhou data)
    print("  Fitting log-linear scaling ...")
    scale_data = []
    for city in all_cities:
        cdf = city_raw[city]
        non2014 = cdf["year"] != 2014
        obs_mean = cdf.loc[non2014, "cases"].mean() + 1e-3
        pred_mean = cdf.loc[non2014, "pred_raw"].mean() + 1e-3
        scale_data.append({"city_en": city, "obs_mean": obs_mean, "pred_mean": pred_mean})

    scale_df = pd.DataFrame(scale_data)
    lo_obs = np.log(scale_df["obs_mean"].values)
    lo_pred = np.log(scale_df["pred_mean"].values)
    # log(obs) = a + b * log(pred)
    b_scale = np.cov(lo_obs, lo_pred)[0, 1] / (np.var(lo_pred) + 1e-12)
    a_scale = np.mean(lo_obs) - b_scale * np.mean(lo_pred)
    print(f"  Log-linear: log(obs) = {a_scale:.3f} + {b_scale:.3f} * log(pred)")

    # Step 3: apply scaling and evaluate
    city_tables = []
    city_metrics = []
    for city in all_cities:
        cdf = city_raw[city].copy()
        pred_raw = cdf["pred_raw"].values
        log_scaled = a_scale + b_scale * np.log(pred_raw + 1e-3)
        pred_scaled = np.maximum(np.exp(log_scaled), 0.0)
        cdf["pred_cases"] = pred_scaled
        city_tables.append(cdf)

        met = evaluate_cases(cdf["cases"].values, pred_scaled)
        met["city_en"] = city
        met["has_bi"] = city in bi_cities
        met["N_h"] = CITY_POP.get(city, N_H_GZ)
        city_metrics.append(met)

    transfer_monthly = pd.concat(city_tables, ignore_index=True)
    transfer_monthly.to_csv(OUT / "transfer_16cities.csv", index=False)

    met_df = pd.DataFrame(city_metrics)
    met_df.to_csv(OUT / "transfer_metrics.csv", index=False)

    print("\nPer-city metrics:")
    print(met_df[["city_en", "has_bi", "pearson_r", "spearman_rho",
                   "r2_log", "wape"]].to_string(index=False))

    grp = met_df.groupby("has_bi")[["pearson_r", "spearman_rho", "r2_log", "wape"]].mean()
    grp.index = grp.index.map({True: "Has BI (8)", False: "No BI (8)"})
    grp.to_csv(OUT / "transfer_group_analysis.csv")
    print("\nGroup comparison:")
    print(grp.to_string())
    print(f"\n16-city mean ρ = {met_df['spearman_rho'].mean():.3f}")

    # Annual 2014
    y14 = transfer_monthly[transfer_monthly["year"] == 2014]
    annual = y14.groupby("city_en").agg(
        cases_annual=("cases", "sum"), pred_annual=("pred_cases", "sum"),
    ).reset_index()
    rho_annual = spearmanr(annual["cases_annual"], annual["pred_annual"])[0]
    annual.to_csv(OUT / "transfer_annual2014.csv", index=False)
    print(f"\n2014 annual ranking ρ = {rho_annual:.3f}")

    # ── Auxiliary: M_hat vs observed BI ────────────────────────────────
    print("\nAuxiliary: formula M̂ vs observed BI ...")
    bi_sub = bi[["city_en", "year", "month", "index_value"]].copy()
    bi_compare = transfer_monthly.merge(bi_sub, on=["city_en", "year", "month"], how="inner")
    bi_compare = bi_compare.rename(columns={"index_value": "bi_obs"})
    bi_compare = bi_compare.dropna(subset=["bi_obs", "m_hat_bi"])

    if len(bi_compare) > 0:
        bi_compare[["city_en", "year", "month", "bi_obs", "m_hat_bi"]].to_csv(
            OUT / "mhat_vs_bi.csv", index=False)
        r_bi = pearsonr(bi_compare["bi_obs"].values, bi_compare["m_hat_bi"].values)[0]
        print(f"  M̂(BI-space) vs BI: r={r_bi:.3f} (n={len(bi_compare)})")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_guangzhou_fit(cases_gz, pred_gz, gz["year"].values, gz["month"].values,
                       gz_met_all, OUT / "fig_guangzhou_cases.png")
    plot_briere_curve(c_opt, T_min_opt, T_max_opt, OUT / "fig_beta_prime_T.png")
    plot_transfer_scatter(annual, bi_cities, OUT / "fig_transfer_scatter.png")
    plot_all_cities_grid(transfer_monthly, OUT / "fig_all_cities_grid.png")
    plot_group_comparison(met_df, OUT / "fig_group_comparison.png")
    plot_loyo_cv(cv_df, OUT / "fig_loyo_cv.png")
    if len(bi_compare) > 0:
        plot_mhat_vs_bi(bi_compare, OUT / "fig_mhat_vs_bi.png")

    print("\n" + "=" * 60)
    print("Part 2 complete!")
    print(f"  Output: {OUT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
