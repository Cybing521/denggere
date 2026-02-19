#!/usr/bin/env python3
"""Run 1+3 pipeline on data_2 processed datasets.

1) Single-city mechanism learning (Guangzhou):
   - Estimate beta(t) from cases and mosquito proxy
   - Train NN(T,H,R)->beta'
2) Symbolic regression:
   - Fit explicit quadratic formula to NN output
3) Transfer:
   - Apply fixed formula to other cities (no retraining)
   - Evaluate annual 2014 cross-city ranking and scale errors
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path("/root/wenmei")
DATA = ROOT / "data_2" / "processed"
OUT = ROOT / "results" / "data2_1plus3"
YEAR_START = 2005
YEAR_END = 2019


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Norm:
    w_min: np.ndarray
    w_max: np.ndarray


class TransmissionNN(nn.Module):
    def __init__(self, n_hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(DATA / "cases_weather_monthly_utf8.csv")
    b = pd.read_csv(DATA / "bi_guangdong_monthly_proxy.csv")
    for c in ["year", "month"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").astype(int)
        b[c] = pd.to_numeric(b[c], errors="coerce").astype(int)
    for c in ["cases", "tem", "rhu", "pre"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    b["index_norm_city"] = pd.to_numeric(b["index_norm_city"], errors="coerce")
    m = m.dropna(subset=["city_en", "year", "month", "cases", "tem", "rhu", "pre"]).copy()
    b = b.dropna(subset=["city_en", "year", "month", "index_norm_city"]).copy()
    m = m[(m["year"] >= YEAR_START) & (m["year"] <= YEAR_END)].copy()
    b = b[(b["year"] >= YEAR_START) & (b["year"] <= YEAR_END)].copy()
    return m, b


def attach_mosquito_proxy(city_df: pd.DataFrame, bi_df: pd.DataFrame) -> pd.DataFrame:
    out = city_df.merge(
        bi_df[["city_en", "year", "month", "index_norm_city"]],
        on=["city_en", "year", "month"],
        how="left",
    )
    if out["index_norm_city"].notna().any():
        out["index_norm_city"] = (
            out["index_norm_city"]
            .interpolate(limit_direction="both")
            .fillna(out["index_norm_city"].median())
        )
    else:
        out["index_norm_city"] = 1.0
    out["index_norm_city"] = np.clip(out["index_norm_city"], 0.05, 20.0)
    return out


def estimate_beta_series(cases: np.ndarray, m_norm: np.ndarray) -> Tuple[np.ndarray, float]:
    pool = np.ones_like(cases, dtype=float)
    for t in range(1, len(cases)):
        pool[t] = max(cases[t - 1] + 0.3 * (cases[t - 2] if t >= 2 else 0.0), 1.0)

    beta_raw = cases / (m_norm * pool + 1e-8)
    if (beta_raw > 0).any():
        p95 = np.percentile(beta_raw[beta_raw > 0], 95)
    else:
        p95 = 1.0
    beta_clip = np.clip(beta_raw, 0.0, p95)
    beta_smooth = gaussian_filter1d(beta_clip, sigma=1.2)
    beta_max = float(beta_smooth.max() + 1e-10)
    beta_norm = beta_smooth / beta_max
    return beta_norm, beta_max


def weather_norm(weather: np.ndarray, norm: Norm) -> np.ndarray:
    w_range = np.maximum(norm.w_max - norm.w_min, 1e-8)
    return np.clip((weather - norm.w_min) / w_range, 0.0, 1.0)


def train_nn(
    x_norm: np.ndarray,
    beta_target: np.ndarray,
    train_mask: np.ndarray,
    n_epochs: int = 2000,
    lr: float = 0.004,
) -> Tuple[TransmissionNN, List[float]]:
    model = TransmissionNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    x = torch.tensor(x_norm, dtype=torch.float32)
    y = torch.tensor(beta_target, dtype=torch.float32)
    m = torch.tensor(train_mask, dtype=torch.bool)

    best = None
    best_loss = np.inf
    losses: List[float] = []

    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(x).squeeze(-1)
        pred_m = pred[m]
        y_m = y[m]

        loss_mse = torch.mean((pred_m - y_m) ** 2)
        if pred_m.std() > 1e-6 and y_m.std() > 1e-6:
            pn = (pred_m - pred_m.mean()) / (pred_m.std() + 1e-6)
            yn = (y_m - y_m.mean()) / (y_m.std() + 1e-6)
            loss_corr = -torch.mean(pn * yn)
        else:
            loss_corr = torch.tensor(0.0)
        loss = loss_mse + 0.3 * loss_corr
        loss.backward()
        opt.step()
        sched.step()
        lv = float(loss.item())
        losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best is not None
    model.load_state_dict(best)
    return model, losses


def evaluate_cases(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.maximum(y_pred, 0.0)
    out = {
        "pearson_r": np.nan,
        "spearman_rho": np.nan,
        "r2_log": np.nan,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman_rho"] = float(spearmanr(y_true, y_pred)[0])
    out["r2_log"] = float(r2_score(np.log1p(y_true), np.log1p(y_pred)))
    return out


def poly2_formula(t: np.ndarray, h: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    a0, a_t, a_h, a_r, a_tt, a_hh, a_rr, a_th, a_tr, a_hr = p
    y = (
        a0
        + a_t * t
        + a_h * h
        + a_r * r
        + a_tt * t**2
        + a_hh * h**2
        + a_rr * r**2
        + a_th * t * h
        + a_tr * t * r
        + a_hr * h * r
    )
    return np.maximum(y, 0.0)


def fit_formula(weather_raw: np.ndarray, beta_nn: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    pf = PolynomialFeatures(degree=2, include_bias=True)
    xp = pf.fit_transform(weather_raw)
    reg = LinearRegression(fit_intercept=False).fit(xp, beta_nn)
    coef = reg.coef_
    # PolynomialFeatures order: [1,T,H,R,T^2,TH,TR,H^2,HR,R^2]
    params = coef[[0, 1, 2, 3, 4, 7, 9, 5, 6, 8]]
    pred = np.maximum(reg.predict(xp), 0.0)
    met = {
        "r2": float(r2_score(beta_nn, pred)),
        "corr": float(pearsonr(beta_nn, pred)[0]) if np.std(pred) > 0 and np.std(beta_nn) > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(beta_nn, pred))),
        "mae": float(np.mean(np.abs(beta_nn - pred))),
    }
    return params, met


def predict_city_risk_monthly(city_df: pd.DataFrame, params: np.ndarray) -> pd.DataFrame:
    t = city_df["tem"].values.astype(float)
    h = city_df["rhu"].values.astype(float)
    r = city_df["pre"].values.astype(float)
    beta = poly2_formula(t, h, r, params)
    m_norm = city_df["index_norm_city"].values.astype(float)
    cases = city_df["cases"].values.astype(float)

    pool = np.ones_like(cases)
    for i in range(1, len(cases)):
        pool[i] = max(cases[i - 1] + 0.3 * (cases[i - 2] if i >= 2 else 0.0), 1.0)

    risk = beta * m_norm * pool
    out = city_df[["city_en", "year", "month", "cases", "tem", "rhu", "pre", "index_norm_city"]].copy()
    out["beta_formula"] = beta
    out["risk_monthly"] = risk
    out["pool_obs_lag"] = pool
    return out


def compute_transfer_metrics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, mask in [("all", np.ones(len(annual), dtype=bool)), ("non_gz", annual["city_en"].values != "Guangzhou")]:
        d = annual[mask].copy()
        if len(d) < 3:
            continue
        x = d["risk_2014"].values.astype(float)
        y = d["cases_2014"].values.astype(float)

        # Scheme 1: scale with Guangzhou anchor
        gz = annual[annual["city_en"] == "Guangzhou"]
        if not gz.empty and float(gz["risk_2014"].iloc[0]) > 0:
            s_gz = float(gz["cases_2014"].iloc[0] / gz["risk_2014"].iloc[0])
        else:
            s_gz = np.nan
        pred_gz = x * s_gz

        # Scheme 2: linear no-GZ scaling (through origin)
        non_gz = annual[annual["city_en"] != "Guangzhou"]
        xn = non_gz["risk_2014"].values.astype(float)
        yn = non_gz["cases_2014"].values.astype(float)
        s_nogz = float(np.sum(xn * yn) / (np.sum(xn**2) + 1e-12))
        pred_nogz = x * s_nogz

        # Scheme 3: log-linear no-GZ scaling
        lx = np.log1p(xn)
        ly = np.log1p(yn)
        coef = np.polyfit(lx, ly, 1)
        pred_loglin = np.expm1(coef[0] * np.log1p(x) + coef[1])
        pred_loglin = np.maximum(pred_loglin, 0.0)

        rho, p = spearmanr(x, y)

        def err(pred: np.ndarray) -> Tuple[float, float, float]:
            mae = float(np.mean(np.abs(pred - y)))
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0)))
            return mae, rmse, mape

        for scheme, pred in [
            ("scale_with_guangzhou", pred_gz),
            ("scale_without_guangzhou", pred_nogz),
            ("loglinear_without_guangzhou", pred_loglin),
        ]:
            mae, rmse, mape = err(pred)
            rows.append(
                {
                    "subset": tag,
                    "scheme": scheme,
                    "N": len(d),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Spearman_rho": float(rho),
                    "Spearman_p": float(p),
                }
            )
    return pd.DataFrame(rows)


def plot_phase1(gz: pd.DataFrame, pred_cases: np.ndarray, phase1_metrics: Dict[str, float], losses: List[float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()
    t = np.arange(len(gz))
    y = gz["cases"].values.astype(float)
    m = gz["year"].values.astype(int) != 2014

    axes[0].plot(t, y, "k-", lw=1.8, label="Observed")
    axes[0].plot(t, pred_cases, "r-", lw=1.3, label="Pred (NN)")
    axes[0].set_yscale("symlog", linthresh=1.0)
    y0, y1 = int(gz["year"].min()), int(gz["year"].max())
    axes[0].set_title(f"Guangzhou monthly cases ({y0}-{y1})")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].scatter(y[m], pred_cases[m], s=20, alpha=0.7)
    mx = max(y[m].max(), pred_cases[m].max()) * 1.05
    axes[1].plot([0, mx], [0, mx], "k--", lw=1)
    axes[1].set_title(
        f"Phase1 metrics (exclude 2014)\n"
        f"r={phase1_metrics['pearson_r']:.3f}, rho={phase1_metrics['spearman_rho']:.3f}, "
        f"R2log={phase1_metrics['r2_log']:.3f}"
    )
    axes[1].grid(alpha=0.25)

    axes[2].plot(gz["date"], gz["beta_target"], "b-", lw=1.6, label="beta target")
    axes[2].plot(gz["date"], gz["beta_nn"], "r--", lw=1.5, label="beta nn")
    axes[2].set_title("Estimated beta vs NN beta")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    axes[3].plot(losses, lw=0.8)
    axes[3].set_yscale("log")
    axes[3].set_title("NN training loss")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "phase1_guangzhou_data2.png", dpi=220)
    plt.close(fig)


def plot_phase2_formula(beta_nn: np.ndarray, beta_formula: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.scatter(beta_nn, beta_formula, s=18, alpha=0.7)
    mx = max(beta_nn.max(), beta_formula.max()) * 1.03
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    r2 = r2_score(beta_nn, beta_formula)
    corr = pearsonr(beta_nn, beta_formula)[0] if np.std(beta_nn) > 0 and np.std(beta_formula) > 0 else np.nan
    ax.set_title(f"Formula vs NN beta\nR2={r2:.4f}, r={corr:.4f}")
    ax.set_xlabel("NN beta")
    ax.set_ylabel("Formula beta")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_formula_fit_data2.png", dpi=220)
    plt.close(fig)


def plot_transfer(annual: pd.DataFrame) -> None:
    # scatter no-gz scaling
    d = annual.sort_values("cases_2014", ascending=False).copy()
    non_gz = d[d["city_en"] != "Guangzhou"].copy()
    xn = non_gz["risk_2014"].values.astype(float)
    yn = non_gz["cases_2014"].values.astype(float)
    s_nogz = float(np.sum(xn * yn) / (np.sum(xn**2) + 1e-12))
    d["cases_pred_scaled_nogz"] = d["risk_2014"] * s_nogz

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.scatter(d["cases_2014"], d["cases_pred_scaled_nogz"], s=52, alpha=0.85)
    for _, r in d.iterrows():
        ax.text(r["cases_2014"] * 1.01, r["cases_pred_scaled_nogz"] * 1.01, r["city_en"], fontsize=7)
    mx = max(d["cases_2014"].max(), d["cases_pred_scaled_nogz"].max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    rho, p = spearmanr(d["risk_2014"], d["cases_2014"])
    ax.set_title(f"Transfer annual 2014 (16 cities)\nSpearman rho={rho:.3f}, p={p:.3g}")
    ax.set_xlabel("Observed cases")
    ax.set_ylabel("Predicted cases (no-GZ scaling)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_2014_scatter_data2.png", dpi=220)
    plt.close(fig)

    # bars
    gz = d[d["city_en"] == "Guangzhou"]
    if not gz.empty and float(gz["risk_2014"].iloc[0]) > 0:
        s_gz = float(gz["cases_2014"].iloc[0] / gz["risk_2014"].iloc[0])
    else:
        s_gz = np.nan
    d["cases_pred_scaled_gz"] = d["risk_2014"] * s_gz

    x = np.arange(len(d))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.bar(x - w, d["cases_2014"], width=w, label="Observed", color="black", alpha=0.75)
    ax.bar(x, d["cases_pred_scaled_gz"], width=w, label="Pred (GZ scale)", color="tab:orange", alpha=0.75)
    ax.bar(x + w, d["cases_pred_scaled_nogz"], width=w, label="Pred (no-GZ scale)", color="tab:blue", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(d["city_en"], rotation=35, ha="right")
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_title("Transfer annual 2014: observed vs predicted")
    ax.set_ylabel("Cases (symlog)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_2014_bars_data2.png", dpi=220)
    plt.close(fig)


def main() -> None:
    set_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    monthly, bi_proxy = load_data()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    if monthly.empty:
        raise RuntimeError("No monthly data left after year filtering.")
    if 2014 not in set(monthly["year"].unique()):
        raise RuntimeError("Year 2014 not available in filtered data; cannot run outbreak transfer diagnostics.")

    # 1) Single-city mechanism learning on Guangzhou
    gz = monthly[monthly["city_en"] == "Guangzhou"].sort_values("date").copy()
    gz = attach_mosquito_proxy(gz, bi_proxy)

    weather = gz[["tem", "rhu", "pre"]].values.astype(float)
    norm = Norm(w_min=weather.min(axis=0), w_max=weather.max(axis=0))
    x_norm = weather_norm(weather, norm)

    beta_target, beta_max = estimate_beta_series(gz["cases"].values.astype(float), gz["index_norm_city"].values.astype(float))
    gz["beta_target"] = beta_target
    train_mask = gz["year"].values != 2014

    model, losses = train_nn(x_norm, beta_target, train_mask)
    with torch.no_grad():
        beta_nn = model(torch.tensor(x_norm, dtype=torch.float32)).squeeze(-1).numpy()
    gz["beta_nn"] = beta_nn

    # direct monthly case reconstruction with learned beta
    cases = gz["cases"].values.astype(float)
    pool = np.ones_like(cases)
    for t in range(1, len(cases)):
        pool[t] = max(cases[t - 1] + 0.3 * (cases[t - 2] if t >= 2 else 0.0), 1.0)
    pred_cases = beta_nn * beta_max * gz["index_norm_city"].values.astype(float) * pool
    gz["cases_pred_nn"] = pred_cases

    phase1_metrics = evaluate_cases(cases[train_mask], pred_cases[train_mask])
    pd.DataFrame([phase1_metrics]).to_csv(OUT / "phase1_metrics_data2.csv", index=False)

    # 2) Symbolic regression (quadratic)
    params, f_metrics = fit_formula(weather, beta_nn)
    beta_formula_gz = poly2_formula(weather[:, 0], weather[:, 1], weather[:, 2], params)
    formula_obs_metrics = {
        "r2": float(r2_score(beta_nn, beta_formula_gz)),
        "corr": float(pearsonr(beta_nn, beta_formula_gz)[0]) if np.std(beta_nn) > 0 and np.std(beta_formula_gz) > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(beta_nn, beta_formula_gz))),
        "mae": float(np.mean(np.abs(beta_nn - beta_formula_gz))),
    }
    pd.DataFrame([f_metrics]).to_csv(OUT / "phase2_formula_fit_metrics_data2.csv", index=False)
    pd.DataFrame(
        {
            "parameter": ["a0", "aT", "aH", "aR", "aTT", "aHH", "aRR", "aTH", "aTR", "aHR"],
            "value": params,
        }
    ).to_csv(OUT / "phase2_formula_params_data2.csv", index=False)

    # 3) Transfer to all cities (no mechanism retraining)
    city_tables = []
    for city in sorted(monthly["city_en"].unique()):
        cdf = monthly[monthly["city_en"] == city].sort_values(["year", "month"]).copy()
        cdf = attach_mosquito_proxy(cdf, bi_proxy)
        city_tables.append(predict_city_risk_monthly(cdf, params))
    transfer_monthly = pd.concat(city_tables, ignore_index=True)
    transfer_monthly.to_csv(OUT / "transfer_monthly_all_cities_data2.csv", index=False)

    annual2014 = (
        transfer_monthly[transfer_monthly["year"] == 2014]
        .groupby("city_en", as_index=False)
        .agg(
            cases_2014=("cases", "sum"),
            risk_2014=("risk_monthly", "sum"),
            beta_mean_2014=("beta_formula", "mean"),
            months=("month", "count"),
        )
        .sort_values("cases_2014", ascending=False)
        .reset_index(drop=True)
    )
    annual2014.to_csv(OUT / "transfer_annual2014_data2.csv", index=False)

    transfer_metrics = compute_transfer_metrics(annual2014)
    transfer_metrics.to_csv(OUT / "transfer_metrics_data2.csv", index=False)

    # figures
    plot_phase1(gz, pred_cases, phase1_metrics, losses)
    plot_phase2_formula(beta_nn, beta_formula_gz)
    plot_transfer(annual2014)

    # export core training table
    gz[
        [
            "city_en",
            "year",
            "month",
            "date",
            "cases",
            "tem",
            "rhu",
            "pre",
            "index_norm_city",
            "beta_target",
            "beta_nn",
            "cases_pred_nn",
        ]
    ].to_csv(OUT / "phase1_guangzhou_predictions_data2.csv", index=False)

    print("=" * 72)
    print("1+3 pipeline on data_2 completed")
    print("=" * 72)
    print(f"Year window: {YEAR_START}-{YEAR_END}")
    print(f"Phase1 metrics: {OUT / 'phase1_metrics_data2.csv'}")
    print(f"Phase2 metrics: {OUT / 'phase2_formula_fit_metrics_data2.csv'}")
    print(f"Phase2 params:  {OUT / 'phase2_formula_params_data2.csv'}")
    print(f"Transfer annual: {OUT / 'transfer_annual2014_data2.csv'}")
    print(f"Transfer metrics: {OUT / 'transfer_metrics_data2.csv'}")
    print(f"Figures: {OUT / 'phase1_guangzhou_data2.png'}, {OUT / 'phase2_formula_fit_data2.png'},")
    print(f"         {OUT / 'transfer_2014_scatter_data2.png'}, {OUT / 'transfer_2014_bars_data2.png'}")


if __name__ == "__main__":
    main()
