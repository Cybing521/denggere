#!/usr/bin/env python3
"""External temporal validation using new BI data (2020-2026).

Validates models trained on 2005-2019 against independent 2020-2026 observations.
  - Brière β'(T) vs observed BI seasonal pattern
  - PySR M̂ formula vs observed BI
  - MOI supplementary analysis
  - Publication-ready figures
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/root/wenmei")
OUT = ROOT / "results" / "external_validation"


# ── Load & preprocess new BI data ────────────────────────────────────────

def load_new_bi():
    df = pd.read_excel(ROOT / "data" / "新BI - 气象.xlsx", sheet_name="Sheet1")
    df["year"] = df["Unnamed: 0"].ffill()
    df["month_raw"] = df["Unnamed: 1"].ffill()
    df["month"] = df["month_raw"].str.extract(r"(\d+)").astype(float)

    monthly = df.groupby(["year", "month"]).agg(
        BI=("BI", "mean"),
        MOI=("MOI", "mean"),
        tem=("温度", "mean"),
        rhu=("湿度", "mean"),
        pre=("降水", "mean"),
    ).reset_index().sort_values(["year", "month"]).reset_index(drop=True)

    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)
    return monthly


# ── Brière function ──────────────────────────────────────────────────────

def briere(T, c, T_min, T_max):
    T = np.asarray(T, dtype=np.float64)
    result = np.zeros_like(T)
    valid = (T > T_min) & (T < T_max)
    Tv = T[valid]
    result[valid] = c * Tv * (Tv - T_min) * np.sqrt(T_max - Tv)
    return np.maximum(result, 0.0)


# ── PySR formula loader ─────────────────────────────────────────────────

def load_pysr_formula():
    formula_file = ROOT / "results" / "part1_mosquito" / "best_formula.txt"
    text = formula_file.read_text()
    eq_str = None
    for line in text.splitlines():
        if line.startswith("Equation:"):
            eq_str = line.split(":", 1)[1].strip()
            break
    if eq_str is None:
        return None, None

    def predict_bi(T, H, R, Tm, Hm, Rm):
        expr = eq_str.replace("safe_sqrt", "np.sqrt").replace("sqrt", "np.sqrt")
        expr = expr.replace("exp(", "np.exp(").replace("cos(", "np.cos(")
        expr = expr.replace("^", "**")
        try:
            log1p_pred = eval(expr, {"np": np, "T": T, "H": H, "R": R,
                                     "Tm": Tm, "Hm": Hm, "Rm": Rm,
                                     "T_mean": Tm, "H_mean": Hm, "R_mean": Rm})
            return np.expm1(np.clip(np.maximum(np.asarray(log1p_pred, float), 0), 0, 10))
        except Exception as e:
            print(f"  WARNING: formula eval failed: {e}")
            return None

    return predict_bi, eq_str


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_briere_vs_bi_timeseries(monthly, beta, out_path):
    """Fig A: time series of Brière β'(T) vs observed BI."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    t = np.arange(len(monthly))
    bi_norm = monthly["BI"].values / monthly["BI"].mean()
    beta_norm = beta / beta.mean()

    ax = axes[0]
    ax.plot(t, bi_norm, "ko-", ms=4, lw=1.2, label="Observed BI (normalized)")
    ax.plot(t, beta_norm, "r-", lw=1.5, alpha=0.85, label="Brière β'(T) (normalized)")
    ax.set_ylabel("Normalized value")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)

    # Year boundaries
    for y in sorted(monthly["year"].unique()):
        idx = monthly[monthly["year"] == y].index[0]
        if idx > 0:
            ax.axvline(idx, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax.text(idx + 0.3, ax.get_ylim()[1] * 0.95, str(y), fontsize=8, color="gray")

    r_val = pearsonr(monthly["BI"].values, beta)[0]
    rho_val = spearmanr(monthly["BI"].values, beta)[0]
    ax.set_title(f"Temporal validation: Brière β'(T) vs observed BI (2020–2026)\n"
                 f"Pearson r = {r_val:.3f}, Spearman ρ = {rho_val:.3f}", fontsize=11)

    # Temperature subplot
    ax2 = axes[1]
    ax2.fill_between(t, monthly["tem"].values, alpha=0.3, color="orange")
    ax2.plot(t, monthly["tem"].values, "orange", lw=1)
    ax2.set_ylabel("T (°C)")
    ax2.set_xlabel("Month index")
    ax2.grid(alpha=0.2)

    # Tick labels
    tick_idx = [i for i in range(len(monthly)) if monthly.iloc[i]["month"] in [1, 7]]
    tick_labels = [f"{int(monthly.iloc[i]['year'])}-{int(monthly.iloc[i]['month']):02d}" for i in tick_idx]
    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_briere_vs_bi_scatter(monthly, beta, out_path):
    """Fig B: scatter plot with per-year coloring."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    years = sorted(monthly["year"].unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(years)))

    # Left: scatter
    ax = axes[0]
    for i, y in enumerate(years):
        mask = monthly["year"] == y
        ax.scatter(beta[mask], monthly.loc[mask, "BI"], s=30, alpha=0.7,
                   color=cmap[i], label=str(y), zorder=3)
    ax.set_xlabel("Brière β'(T)")
    ax.set_ylabel("Observed BI")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    r_val = pearsonr(monthly["BI"].values, beta)[0]
    ax.set_title(f"(a) Scatter: r = {r_val:.3f}")

    # Right: seasonal profile
    ax = axes[1]
    seasonal_bi = monthly.groupby("month")["BI"].agg(["mean", "std"])
    seasonal_beta = monthly.groupby("month").apply(
        lambda g: pd.Series({"mean": beta[g.index].mean(), "std": beta[g.index].std()}))

    months = seasonal_bi.index.values
    bi_m, bi_s = seasonal_bi["mean"].values, seasonal_bi["std"].values
    bt_m, bt_s = seasonal_beta["mean"].values, seasonal_beta["std"].values

    # Normalize for overlay
    bi_norm = bi_m / bi_m.max()
    bt_norm = bt_m / bt_m.max()

    ax.fill_between(months, bi_norm - bi_s / bi_m.max(), bi_norm + bi_s / bi_m.max(),
                    alpha=0.15, color="black")
    ax.plot(months, bi_norm, "ko-", ms=5, lw=1.5, label="BI (normalized)")
    ax.fill_between(months, bt_norm - bt_s / bt_m.max(), bt_norm + bt_s / bt_m.max(),
                    alpha=0.15, color="red")
    ax.plot(months, bt_norm, "r^-", ms=5, lw=1.5, label="β'(T) (normalized)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized value")
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    r_seasonal = pearsonr(bi_m, bt_m)[0]
    ax.set_title(f"(b) Seasonal profile: r = {r_seasonal:.3f}")

    fig.suptitle("External validation (2020–2026)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_per_year_validation(monthly, beta, out_path):
    """Fig C: per-year scatter subplots."""
    years = sorted(monthly["year"].unique())
    years = [y for y in years if (monthly["year"] == y).sum() >= 4]
    n = len(years)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for i, y in enumerate(years):
        ax = axes[i]
        mask = monthly["year"] == y
        bi_y = monthly.loc[mask, "BI"].values
        bt_y = beta[mask]
        ax.scatter(bt_y, bi_y, s=40, alpha=0.7, color="steelblue", zorder=3)

        # Month labels
        for _, row in monthly[mask].iterrows():
            ax.annotate(f"{int(row['month'])}月",
                        (beta[row.name], row["BI"]),
                        fontsize=7, textcoords="offset points", xytext=(4, 4))

        if np.std(bi_y) > 0 and np.std(bt_y) > 0:
            r_y = pearsonr(bi_y, bt_y)[0]
            rho_y = spearmanr(bi_y, bt_y)[0]
            ax.set_title(f"{y}  (r={r_y:.3f}, ρ={rho_y:.3f})", fontsize=10)
        else:
            ax.set_title(f"{y}", fontsize=10)

        ax.set_xlabel("β'(T)")
        ax.set_ylabel("BI")
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-year external validation (Brière β'(T) vs BI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_moi_analysis(monthly, beta, out_path):
    """Fig D: MOI supplementary analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # MOI vs BI
    ax = axes[0]
    ax.scatter(monthly["MOI"], monthly["BI"], s=25, alpha=0.6, color="teal")
    r_mb = pearsonr(monthly["MOI"], monthly["BI"])[0]
    ax.set_xlabel("MOI")
    ax.set_ylabel("BI")
    ax.set_title(f"(a) MOI vs BI: r = {r_mb:.3f}")
    ax.grid(alpha=0.2)

    # MOI vs β'(T)
    ax = axes[1]
    ax.scatter(beta, monthly["MOI"], s=25, alpha=0.6, color="coral")
    r_mt = pearsonr(monthly["MOI"], beta)[0]
    ax.set_xlabel("β'(T)")
    ax.set_ylabel("MOI")
    ax.set_title(f"(b) β'(T) vs MOI: r = {r_mt:.3f}")
    ax.grid(alpha=0.2)

    # Seasonal: BI, MOI, β'(T) overlay
    ax = axes[2]
    seasonal = monthly.groupby("month").agg(
        BI=("BI", "mean"), MOI=("MOI", "mean"), T=("tem", "mean")).reset_index()
    seasonal["beta"] = briere(seasonal["T"].values,
                              *pd.read_csv(ROOT / "results" / "part2_seir" / "seir_params.csv")[
                                  ["c", "T_min", "T_max"]].values[0])

    m = seasonal["month"].values
    ax.plot(m, seasonal["BI"] / seasonal["BI"].max(), "ko-", ms=5, lw=1.5, label="BI")
    ax.plot(m, seasonal["MOI"] / seasonal["MOI"].max(), "s-", ms=5, lw=1.5,
            color="teal", label="MOI")
    ax.plot(m, seasonal["beta"] / seasonal["beta"].max(), "r^-", ms=5, lw=1.5, label="β'(T)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized")
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=9)
    ax.set_title("(c) Seasonal: BI, MOI, β'(T)")
    ax.grid(alpha=0.2)

    fig.suptitle("MOI supplementary analysis (2020–2026)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pysr_validation(monthly, mhat_bi, out_path):
    """Fig E: PySR M̂ formula vs observed BI."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    t = np.arange(len(monthly))
    bi_norm = monthly["BI"].values / monthly["BI"].mean()
    mhat_norm = mhat_bi / mhat_bi.mean()
    ax.plot(t, bi_norm, "ko-", ms=3, lw=1, label="Observed BI")
    ax.plot(t, mhat_norm, "b-", lw=1.3, alpha=0.8, label="PySR M̂")
    ax.set_ylabel("Normalized")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    r_val = pearsonr(monthly["BI"].values, mhat_bi)[0]
    ax.set_title(f"(a) Time series: r = {r_val:.3f}")

    ax = axes[1]
    ax.scatter(mhat_bi, monthly["BI"].values, s=25, alpha=0.6, color="steelblue")
    ax.set_xlabel("PySR M̂ (BI scale)")
    ax.set_ylabel("Observed BI")
    rho_val = spearmanr(monthly["BI"].values, mhat_bi)[0]
    ax.set_title(f"(b) Scatter: ρ = {rho_val:.3f}")
    ax.grid(alpha=0.2)

    fig.suptitle("PySR M-hat formula validation (2020-2026)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("External Temporal Validation (2020-2026)")
    print("=" * 60)

    # Load new data
    print("\nLoading new BI data ...")
    monthly = load_new_bi()
    print(f"  {len(monthly)} months, {monthly['year'].min()}-{monthly['year'].max()}")

    # Load Brière params
    params = pd.read_csv(ROOT / "results" / "part2_seir" / "seir_params.csv")
    c, T_min, T_max = params["c"].values[0], params["T_min"].values[0], params["T_max"].values[0]
    print(f"  Brière: c={c:.2e}, T_min={T_min:.1f}, T_max={T_max:.1f}")

    beta = briere(monthly["tem"].values, c, T_min, T_max)
    monthly["beta_briere"] = beta

    # ── 1. Overall metrics ────────────────────────────────────────────
    print("\n--- Brière β'(T) vs observed BI ---")
    r_all = pearsonr(monthly["BI"].values, beta)[0]
    rho_all = spearmanr(monthly["BI"].values, beta)[0]
    print(f"  Overall: Pearson r = {r_all:.4f}, Spearman ρ = {rho_all:.4f}")

    # Per-year
    year_rows = []
    for y in sorted(monthly["year"].unique()):
        ydf = monthly[monthly["year"] == y]
        if len(ydf) >= 4:
            r_y = pearsonr(ydf["BI"].values, ydf["beta_briere"].values)[0]
            rho_y = spearmanr(ydf["BI"].values, ydf["beta_briere"].values)[0]
        else:
            r_y, rho_y = np.nan, np.nan
        year_rows.append({"year": y, "n_months": len(ydf),
                          "pearson_r": r_y, "spearman_rho": rho_y,
                          "BI_mean": ydf["BI"].mean(), "T_mean": ydf["tem"].mean()})
        print(f"  {y}: n={len(ydf):2d}, r={r_y:.3f}, ρ={rho_y:.3f}")

    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(OUT / "briere_per_year_metrics.csv", index=False)

    # Seasonal
    seasonal = monthly.groupby("month").agg(BI_mean=("BI", "mean")).reset_index()
    seasonal["beta_mean"] = monthly.groupby("month")["beta_briere"].mean().values
    r_seasonal = pearsonr(seasonal["BI_mean"].values, seasonal["beta_mean"].values)[0]
    print(f"  Seasonal profile: r = {r_seasonal:.4f}")

    # ── 2. PySR M̂ validation ─────────────────────────────────────────
    print("\n--- PySR M̂ formula vs observed BI ---")
    predict_bi_fn, eq_str = load_pysr_formula()
    mhat_bi = None
    if predict_bi_fn is not None:
        T = monthly["tem"].values
        H = monthly["rhu"].values
        R = monthly["pre"].values
        Tm = np.full_like(T, T.mean())
        Hm = np.full_like(H, H.mean())
        Rm = np.full_like(R, R.mean())
        mhat_bi = predict_bi_fn(T, H, R, Tm, Hm, Rm)
        if mhat_bi is not None:
            r_mhat = pearsonr(monthly["BI"].values, mhat_bi)[0]
            rho_mhat = spearmanr(monthly["BI"].values, mhat_bi)[0]
            print(f"  PySR M̂ vs BI: r = {r_mhat:.4f}, ρ = {rho_mhat:.4f}")
        else:
            print("  PySR formula evaluation failed")
    else:
        print("  PySR formula not found")

    # ── 3. MOI analysis ──────────────────────────────────────────────
    print("\n--- MOI supplementary ---")
    r_moi_bi = pearsonr(monthly["MOI"].values, monthly["BI"].values)[0]
    r_moi_beta = pearsonr(monthly["MOI"].values, beta)[0]
    r_moi_T = pearsonr(monthly["MOI"].values, monthly["tem"].values)[0]
    print(f"  MOI vs BI:    r = {r_moi_bi:.4f}")
    print(f"  MOI vs β'(T): r = {r_moi_beta:.4f}")
    print(f"  MOI vs T:     r = {r_moi_T:.4f}")

    # ── 4. Summary table ─────────────────────────────────────────────
    summary = {
        "metric": ["Brière β'(T) vs BI", "Brière seasonal", "PySR M̂ vs BI",
                    "MOI vs BI", "MOI vs β'(T)"],
        "pearson_r": [r_all, r_seasonal,
                      r_mhat if mhat_bi is not None else np.nan,
                      r_moi_bi, r_moi_beta],
        "spearman_rho": [rho_all,
                         spearmanr(seasonal["BI_mean"].values, seasonal["beta_mean"].values)[0],
                         rho_mhat if mhat_bi is not None else np.nan,
                         spearmanr(monthly["MOI"].values, monthly["BI"].values)[0],
                         spearmanr(monthly["MOI"].values, beta)[0]],
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "validation_summary.csv", index=False)
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))

    # Save processed data
    monthly.to_csv(OUT / "new_bi_monthly_processed.csv", index=False)

    # ── 5. Plots ─────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_briere_vs_bi_timeseries(monthly, beta, OUT / "fig_briere_vs_bi_timeseries.png")
    plot_briere_vs_bi_scatter(monthly, beta, OUT / "fig_briere_vs_bi_scatter.png")
    plot_per_year_validation(monthly, beta, OUT / "fig_per_year_validation.png")
    plot_moi_analysis(monthly, beta, OUT / "fig_moi_analysis.png")
    if mhat_bi is not None:
        plot_pysr_validation(monthly, mhat_bi, OUT / "fig_pysr_validation.png")

    print(f"\n{'=' * 60}")
    print(f"External validation complete!")
    print(f"  Output: {OUT}")
    print(f"  Figures: 4-5 PNG files")
    print(f"  Metrics: validation_summary.csv, briere_per_year_metrics.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
