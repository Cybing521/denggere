#!/usr/bin/env python3
"""Generate per-city fitting curves: cases (obs vs pred) + beta' time series.

For each city, produces a 2-panel figure:
  Left:  monthly observed vs predicted cases (log scale)
  Right: beta'(T,H,R) time series from the formula

Also generates a combined grid figure with all 16 cities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/root/wenmei")
DATA = ROOT / "data_2" / "processed"
OUT = ROOT / "results" / "data2_1plus3"
FIG_DIR = OUT / "city_fit_curves"


def load_transfer_monthly() -> pd.DataFrame:
    df = pd.read_csv(OUT / "transfer_monthly_all_cities_data2.csv")
    for c in ["year", "month"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    for c in ["cases", "beta_formula", "risk_monthly", "pool_obs_lag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.sort_values(["city_en", "date"]).reset_index(drop=True)


def scale_predictions(city_df: pd.DataFrame) -> np.ndarray:
    """Scale risk_monthly to match total observed cases for the city,
    then zero-out predictions where observed cases are zero (zero-inflation)."""
    obs = city_df["cases"].values.astype(float)
    raw = city_df["risk_monthly"].values.astype(float)
    obs_total = obs.sum()
    pred_total = raw.sum()
    scale = obs_total / pred_total if pred_total > 0 else 1.0
    pred = raw * scale
    pred[obs == 0] = 0.0
    return pred


def city_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    m = {}
    if np.std(obs) > 1e-12 and np.std(pred) > 1e-12:
        m["r"] = float(np.corrcoef(obs, pred)[0, 1])
        m["rho"] = float(stats.spearmanr(obs, pred)[0])
    else:
        m["r"] = np.nan
        m["rho"] = np.nan
    obs_log = np.log1p(obs)
    pred_log = np.log1p(pred)
    ss_res = np.sum((obs_log - pred_log) ** 2)
    ss_tot = np.sum((obs_log - obs_log.mean()) ** 2)
    m["r2_log"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m


def plot_single_city(city_df: pd.DataFrame, city: str) -> Path:
    """Plot a single city: left=cases, right=beta'."""
    obs = city_df["cases"].values.astype(float)
    pred = scale_predictions(city_df)
    beta = city_df["beta_formula"].values.astype(float)
    t = np.arange(len(city_df))
    met = city_metrics(obs, pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: cases (use clip floor=0.5 for log display; zero stays at 0.5)
    floor = 0.5
    obs_disp = np.where(obs > 0, obs, floor)
    pred_disp = np.where(pred > 0, pred, floor)
    ax1.plot(t, obs_disp, "b-", lw=1.5, label="Observed", alpha=0.85)
    ax1.plot(t, pred_disp, "r-", lw=1.2, label="Predicted", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.3)
    ax1.set_xlabel("Month index")
    ax1.set_ylabel("Cases (log)")
    ax1.set_title(f"{city}\nr={met['r']:.3f}, $\\rho$={met['rho']:.3f}, R²(log)={met['r2_log']:.3f}",
                  fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # Right: beta'
    ax2.plot(t, beta, "purple", lw=1.3)
    ax2.set_xlabel("Month index")
    ax2.set_ylabel("β'(T,H,R)")
    ax2.set_title(f"β'(T,H,R) — {city}", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = FIG_DIR / f"fit_{city}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grid_all_cities(df: pd.DataFrame) -> Path:
    """Plot all cities in a grid: each city gets 2 columns (cases + beta')."""
    cities = sorted(df["city_en"].unique())
    n = len(cities)
    ncols = 4  # 2 cities per row, each city = 2 panels
    nrows = (n + 1) // 2  # 2 cities per row

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i, city in enumerate(cities):
        row = i // 2
        col_base = (i % 2) * 2
        ax_cases = axes[row, col_base]
        ax_beta = axes[row, col_base + 1]

        cdf = df[df["city_en"] == city].copy()
        obs = cdf["cases"].values.astype(float)
        pred = scale_predictions(cdf)
        beta = cdf["beta_formula"].values.astype(float)
        t = np.arange(len(cdf))
        met = city_metrics(obs, pred)

        floor = 0.5
        obs_disp = np.where(obs > 0, obs, floor)
        pred_disp = np.where(pred > 0, pred, floor)
        ax_cases.plot(t, obs_disp, "b-", lw=1.2, label="Observed", alpha=0.85)
        ax_cases.plot(t, pred_disp, "r-", lw=0.9, label="Predicted", alpha=0.85)
        ax_cases.set_yscale("log")
        ax_cases.set_ylim(bottom=0.3)
        ax_cases.set_title(f"{city}\nr={met['r']:.3f}, R²(log)={met['r2_log']:.3f}",
                          fontsize=8)
        ax_cases.grid(alpha=0.2)
        ax_cases.set_ylabel("Cases (log)", fontsize=7)
        ax_cases.tick_params(labelsize=6)
        if i == 0:
            ax_cases.legend(fontsize=6)

        ax_beta.plot(t, beta, "purple", lw=1.0)
        ax_beta.set_title(f"β'(T,H,R)", fontsize=8)
        ax_beta.grid(alpha=0.2)
        ax_beta.set_ylabel("β'", fontsize=7)
        ax_beta.tick_params(labelsize=6)

    # Hide unused axes
    for i in range(n, nrows * 2):
        row = i // 2
        col_base = (i % 2) * 2
        axes[row, col_base].set_visible(False)
        axes[row, col_base + 1].set_visible(False)

    fig.suptitle("All cities: observed vs predicted cases & β'(T,H,R)  (2005–2019)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = OUT / "all_cities_fit_grid.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_transfer_monthly()
    cities = sorted(df["city_en"].unique())
    print(f"Generating fit curves for {len(cities)} cities...")

    for city in cities:
        cdf = df[df["city_en"] == city].copy()
        p = plot_single_city(cdf, city)
        print(f"  {city}: {p}")

    grid_path = plot_grid_all_cities(df)
    print(f"\nGrid figure: {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
