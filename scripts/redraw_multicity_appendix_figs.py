from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
IN_ANNUAL = ROOT / "results" / "data2_1plus3" / "transfer_annual2014_data2.csv"
IN_MONTHLY = ROOT / "results" / "data2_1plus3" / "transfer_monthly_all_cities_data2.csv"
OUT = ROOT / "results" / "paper_extra"


def prepare() -> tuple[pd.DataFrame, float, float]:
    d = pd.read_csv(IN_ANNUAL).copy()
    d = d.sort_values("cases_2014", ascending=False).reset_index(drop=True)

    gz = d[d["city_en"] == "Guangzhou"]
    if gz.empty or float(gz["risk_2014"].iloc[0]) <= 0:
        raise RuntimeError("Cannot compute Guangzhou scaling anchor.")
    s_gz = float(gz["cases_2014"].iloc[0] / gz["risk_2014"].iloc[0])

    non = d[d["city_en"] != "Guangzhou"]
    x_non = non["risk_2014"].values.astype(float)
    y_non = non["cases_2014"].values.astype(float)
    s_nogz = float(np.sum(x_non * y_non) / (np.sum(x_non**2) + 1e-12))

    d["pred_gz"] = d["risk_2014"] * s_gz
    d["pred_nogz"] = d["risk_2014"] * s_nogz
    return d, s_gz, s_nogz


def plot_d12_all(d: pd.DataFrame) -> None:
    x = d["cases_2014"].values.astype(float)
    y = d["pred_nogz"].values.astype(float)
    rho, p = spearmanr(d["risk_2014"], x)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3))
    ax1, ax2 = axes

    # Left: linear view (keep comparability with y=x)
    ax1.scatter(x, y, s=44, c="#2ca25f", alpha=0.85)
    mx = max(x.max(), y.max()) * 1.05
    ax1.plot([0, mx], [0, mx], "k--", lw=1)
    for i, r in d.iterrows():
        offx = 10 if (i % 2 == 0) else -8
        offy = 8 if (i % 3 == 0) else -6
        ax1.annotate(
            r["city_en"],
            (r["cases_2014"], r["pred_nogz"]),
            textcoords="offset points",
            xytext=(offx, offy),
            fontsize=7,
        )
    ax1.set_title("Annual fit (all 16 cities) - linear")
    ax1.set_xlabel("Observed 2014 cases")
    ax1.set_ylabel("Predicted (no-GZ scale)")
    ax1.grid(alpha=0.25)

    # Right: log view (solve crowding of low-incidence cities)
    ax2.scatter(x, y, s=44, c="#1f78b4", alpha=0.85)
    min_v = 1.0
    max_v = mx
    ax2.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    for i, r in d.iterrows():
        offx = 8 if (i % 2 == 0) else -10
        offy = 8 if (i % 3 == 0) else -8
        ax2.annotate(
            r["city_en"],
            (max(float(r["cases_2014"]), 1.0), max(float(r["pred_nogz"]), 1.0)),
            textcoords="offset points",
            xytext=(offx, offy),
            fontsize=7,
        )
    ax2.set_title("Annual fit (all 16 cities) - log-log")
    ax2.set_xlabel("Observed 2014 cases (log)")
    ax2.set_ylabel("Predicted (no-GZ scale, log)")
    ax2.grid(alpha=0.25, which="both")

    fig.suptitle(
        f"Multicity annual fit (2014)  |  Spearman rho={rho:.3f} (p={p:.2g})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "D12_multicity_annual_fit_scatter.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_d12b_no_gz(d: pd.DataFrame) -> None:
    non = d[d["city_en"] != "Guangzhou"].copy()
    x = non["cases_2014"].values.astype(float)
    y = non["pred_nogz"].values.astype(float)
    rho, p = spearmanr(non["risk_2014"], x)
    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.scatter(x, y, s=58, c="#2ca25f", alpha=0.9, edgecolor="white", linewidth=0.6)

    mx = max(x.max(), y.max()) * 1.10
    ax.plot([1, mx], [1, mx], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, mx)
    ax.set_ylim(1, mx)

    # Label all cities with alternating offsets to reduce overlap.
    for i, r in non.reset_index(drop=True).iterrows():
        offx = 9 if (i % 2 == 0) else -10
        offy = 9 if (i % 3 == 0) else -8
        ax.annotate(
            r["city_en"],
            (max(float(r["cases_2014"]), 1.0), max(float(r["pred_nogz"]), 1.0)),
            textcoords="offset points",
            xytext=(offx, offy),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )

    ax.text(
        0.03,
        0.97,
        (
            f"Non-GZ cities (n={len(non)})\n"
            f"Spearman rho={rho:.3f} (p={p:.4g})\n"
            f"MAE={mae:.1f}\nRMSE={rmse:.1f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title("Other-city annual fit (excluding Guangzhou)")
    ax.set_xlabel("Observed 2014 cases (log scale)")
    ax.set_ylabel("Predicted cases (no-GZ scale, log scale)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "D12b_multicity_annual_fit_scatter_no_gz.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_d13_bars_broken_axis(d: pd.DataFrame) -> None:
    x = np.arange(len(d))
    w = 0.24

    non = d[d["city_en"] != "Guangzhou"].copy().reset_index(drop=True)
    x2 = np.arange(len(non))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.8, 5.8), gridspec_kw={"wspace": 0.24})

    # Left: full-city view on log scale (resolves vertical compression).
    ax_left.bar(x - w, d["cases_2014"], width=w, label="Observed", color="#333333", alpha=0.82)
    ax_left.bar(x, d["pred_gz"], width=w, label="Pred (with GZ scale)", color="#f28e2b", alpha=0.82)
    ax_left.bar(x + w, d["pred_nogz"], width=w, label="Pred (no-GZ scale)", color="#4e9ac7", alpha=0.82)
    ax_left.set_yscale("log")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(d["city_en"], rotation=40, ha="right")
    ax_left.set_title("All cities (log y-scale)")
    ax_left.set_ylabel("Cases (log scale)")
    ax_left.grid(axis="y", alpha=0.25, which="both")
    ax_left.legend(fontsize=8, ncol=1, loc="upper right")

    # Right: non-GZ zoomed linear panel (readable middle/low values).
    ax_right.bar(x2 - w, non["cases_2014"], width=w, label="Observed", color="#333333", alpha=0.82)
    ax_right.bar(x2, non["pred_gz"], width=w, label="Pred (with GZ scale)", color="#f28e2b", alpha=0.82)
    ax_right.bar(x2 + w, non["pred_nogz"], width=w, label="Pred (no-GZ scale)", color="#4e9ac7", alpha=0.82)
    y_max = float(
        np.max(
            [
                non["cases_2014"].max(),
                non["pred_gz"].max(),
                non["pred_nogz"].max(),
            ]
        )
    )
    ax_right.set_ylim(0, y_max * 1.15)
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(non["city_en"], rotation=40, ha="right")
    ax_right.set_title("Non-Guangzhou cities (linear y-scale)")
    ax_right.set_ylabel("Cases")
    ax_right.grid(axis="y", alpha=0.25)

    fig.suptitle("Other-city annual observed vs predicted (2014)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "D13_multicity_annual_fit_bars.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_d14_monthly_curves(d: pd.DataFrame) -> None:
    monthly = pd.read_csv(IN_MONTHLY)
    y14 = monthly[monthly["year"] == 2014].copy()
    non = d[d["city_en"] != "Guangzhou"].copy()
    s_nogz = float(np.sum(non["risk_2014"] * non["cases_2014"]) / (np.sum(non["risk_2014"] ** 2) + 1e-12))
    y14["pred_cases_nogz"] = y14["risk_monthly"] * s_nogz

    cities = sorted(y14["city_en"].unique())
    n = len(cities)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 3.2 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)
    months = np.arange(1, 13)

    for i, city in enumerate(cities):
        ax = axes[i]
        c = y14[y14["city_en"] == city].sort_values("month")
        obs = c["cases"].values.astype(float)
        pred = c["pred_cases_nogz"].values.astype(float)
        ax.plot(months, obs, color="#333333", lw=1.4, marker="o", ms=2.8, label="Obs")
        ax.plot(months, pred, color="#1f78b4", lw=1.4, marker="o", ms=2.8, label="Pred")
        ax.set_title(city, fontsize=9)
        ax.grid(alpha=0.25)
        if np.max(obs) > 300 or np.max(pred) > 300:
            ax.set_yscale("symlog", linthresh=10.0)
        ax.set_xlim(1, 12)
        ax.set_xticks([1, 3, 5, 7, 9, 11])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("2014 multicity monthly curves: observed vs predicted (no-GZ scaling)", y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.savefig(OUT / "D14_multicity_monthly_pred_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, _, _ = prepare()
    plot_d12_all(d)
    plot_d12b_no_gz(d)
    plot_d13_bars_broken_axis(d)
    plot_d14_monthly_curves(d)
    print("Rewrote D12/D12b/D13/D14 in", OUT)


if __name__ == "__main__":
    main()
