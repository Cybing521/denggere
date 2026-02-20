#!/usr/bin/env python3
"""Run 1+3 pipeline on data_2 processed datasets.

1) Single-city mechanism learning (Guangzhou):
   - Estimate beta(t) from cases and mosquito proxy
   - Train NN(T,H,R)->beta'
2) Symbolic regression (PySR):
   - Use PySR to search for optimal formula approximating NN output
3) Transfer:
   - Apply discovered formula to other cities (no retraining)
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
from scipy.optimize import brentq
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from pysr import PySRRegressor

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

# ── SEIR dynamics constants ──────────────────────────────────────────────
SIGMA_H = 1.0 / 5.9       # latent-to-infectious rate (1/incubation period)
GAMMA = 1.0 / 14.0        # recovery rate
N_H = 1.426e7             # Guangzhou resident population
DAYS_PER_MONTH = 30       # days per time step for Euler integration
ETA_CANDIDATES = [0.1, 0.5, 1.0, 2.0, 5.0]  # import rate grid (persons/day)


def seir_forward(
    beta_series: np.ndarray,
    m_norm: np.ndarray,
    eta: float,
    n_h: float = N_H,
    days: int = DAYS_PER_MONTH,
) -> np.ndarray:
    """Run SEIR ODE forward simulation with Euler integration.

    Args:
        beta_series: transmission coefficient per time step (length T).
        m_norm: normalised mosquito density per time step (length T).
        eta: exogenous import rate (persons / day).
        n_h: host population size.
        days: number of daily Euler steps per time step.

    Returns:
        Monthly new case counts (length T).
    """
    T = len(beta_series)
    S = n_h - 1.0
    E = 0.0
    I = 1.0
    R = 0.0
    cases_out = np.zeros(T)

    for t in range(T):
        beta_t = beta_series[t]
        m_t = m_norm[t]
        month_cases = 0.0
        for _ in range(days):
            lam = beta_t * m_t * I / n_h
            new_exposed = lam * S
            dS = -new_exposed
            dE = new_exposed + eta - SIGMA_H * E
            dI = SIGMA_H * E - GAMMA * I
            dR = GAMMA * I
            new_cases = SIGMA_H * max(E, 0.0)
            month_cases += new_cases
            S = max(S + dS, 0.0)
            E = max(E + dE, 0.0)
            I = max(I + dI, 0.0)
            R = R + dR
        cases_out[t] = month_cases
    return cases_out


def _solve_beta_for_month(
    target_cases: float,
    m_t: float,
    eta: float,
    S: float,
    E: float,
    I: float,
    R: float,
    n_h: float = N_H,
    days: int = DAYS_PER_MONTH,
) -> Tuple[float, float, float, float, float]:
    """Solve for beta that produces target_cases in one month via bisection.

    Returns (beta, S_end, E_end, I_end, R_end) after the month.
    """

    def _simulate_month(beta_val: float) -> Tuple[float, float, float, float, float]:
        s, e, i, r = S, E, I, R
        mc = 0.0
        for _ in range(days):
            lam = beta_val * m_t * i / n_h
            ne = lam * s
            ds = -ne
            de = ne + eta - SIGMA_H * e
            di = SIGMA_H * e - GAMMA * i
            dr = GAMMA * i
            mc += SIGMA_H * max(e, 0.0)
            s = max(s + ds, 0.0)
            e = max(e + de, 0.0)
            i = max(i + di, 0.0)
            r = r + dr
        return mc, s, e, i, r

    # Edge case: if target is near zero, beta ~ 0
    if target_cases < 0.5:
        mc, s, e, i, r = _simulate_month(0.0)
        return 0.0, s, e, i, r

    # Find upper bound for bisection
    beta_lo, beta_hi = 0.0, 1.0
    for _ in range(20):
        mc_hi, *_ = _simulate_month(beta_hi)
        if mc_hi >= target_cases:
            break
        beta_hi *= 2.0
    else:
        # Could not bracket; return best effort
        mc, s, e, i, r = _simulate_month(beta_hi)
        return beta_hi, s, e, i, r

    # Bisection
    def residual(b: float) -> float:
        mc, *_ = _simulate_month(b)
        return mc - target_cases

    try:
        beta_opt = brentq(residual, beta_lo, beta_hi, xtol=1e-8, maxiter=200)
    except ValueError:
        beta_opt = beta_hi

    mc, s, e, i, r = _simulate_month(beta_opt)
    return beta_opt, s, e, i, r


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


def estimate_beta_series(
    cases: np.ndarray,
    m_norm: np.ndarray,
    eta_candidates: List[float] = ETA_CANDIDATES,
    n_h: float = N_H,
) -> Tuple[np.ndarray, float, float]:
    """Invert SEIR ODE to estimate beta(t) time series.

    For each candidate eta, solve beta(t) month-by-month via bisection so that
    the SEIR forward simulation reproduces the observed case counts.  Select the
    eta that minimises total squared error.

    Returns:
        (beta_norm, beta_max, best_eta)
    """
    best_err = np.inf
    best_beta: np.ndarray | None = None
    best_eta = eta_candidates[0]

    for eta in eta_candidates:
        S, E, I, R = n_h - 1.0, 0.0, 1.0, 0.0
        beta_raw = np.zeros(len(cases))
        sim_cases = np.zeros(len(cases))

        for t in range(len(cases)):
            beta_t, S, E, I, R = _solve_beta_for_month(
                target_cases=cases[t],
                m_t=m_norm[t],
                eta=eta,
                S=S, E=E, I=I, R=R,
                n_h=n_h,
            )
            beta_raw[t] = beta_t
            # Re-simulate to get actual predicted cases for error calc
            sim_cases[t] = seir_forward(
                beta_raw[t : t + 1], m_norm[t : t + 1], eta, n_h, DAYS_PER_MONTH
            )[0]

        err = float(np.sum((cases - sim_cases) ** 2))
        if err < best_err:
            best_err = err
            best_beta = beta_raw.copy()
            best_eta = eta

    assert best_beta is not None
    # Smooth and normalise
    beta_smooth = gaussian_filter1d(np.clip(best_beta, 0.0, None), sigma=1.2)
    beta_max = float(beta_smooth.max() + 1e-10)
    beta_norm = beta_smooth / beta_max
    return beta_norm, beta_max, best_eta


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


def fit_formula_pysr(
    weather_raw: np.ndarray,
    beta_nn: np.ndarray,
    niterations: int = 200,
    timeout: int = 600,
) -> Tuple[PySRRegressor, Dict[str, float]]:
    """Use PySR to search for the best symbolic formula: beta' = f(T, H, R)."""
    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "square", "cos"],
        niterations=niterations,
        maxsize=25,
        maxdepth=8,
        populations=30,
        population_size=50,
        model_selection="best",
        early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 15",
        timeout_in_seconds=timeout,
        progress=True,
        temp_equation_file=True,
    )
    model.fit(weather_raw, beta_nn, variable_names=["T", "H", "R"])

    pred = np.maximum(model.predict(weather_raw), 0.0)
    met = {
        "r2": float(r2_score(beta_nn, pred)),
        "corr": float(pearsonr(beta_nn, pred)[0]) if np.std(pred) > 0 and np.std(beta_nn) > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(beta_nn, pred))),
        "mae": float(np.mean(np.abs(beta_nn - pred))),
    }
    return model, met


def predict_with_pysr(model: PySRRegressor, weather_raw: np.ndarray) -> np.ndarray:
    """Predict beta' using the PySR best equation, clipped to >= 0."""
    return np.maximum(model.predict(weather_raw), 0.0)


def seir_reconstruct_cases(
    beta_series: np.ndarray,
    beta_max: float,
    m_norm: np.ndarray,
    eta: float,
    n_h: float = N_H,
) -> np.ndarray:
    """Reconstruct monthly cases using SEIR forward simulation.

    beta_series is in normalised [0,1] space; it is scaled by beta_max before
    feeding into the ODE.
    """
    return seir_forward(beta_series * beta_max, m_norm, eta, n_h)


def predict_city_risk_monthly(
    city_df: pd.DataFrame,
    sr_model: PySRRegressor,
    beta_max: float,
    eta: float,
    n_h: float = N_H,
) -> pd.DataFrame:
    weather = city_df[["tem", "rhu", "pre"]].values.astype(float)
    beta_norm = predict_with_pysr(sr_model, weather)
    m_norm = city_df["index_norm_city"].values.astype(float)

    pred_cases = seir_forward(beta_norm * beta_max, m_norm, eta, n_h)

    out = city_df[["city_en", "year", "month", "cases", "tem", "rhu", "pre", "index_norm_city"]].copy()
    out["beta_formula"] = beta_norm
    out["risk_monthly"] = pred_cases
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


def plot_phase2_formula(beta_nn: np.ndarray, beta_formula: np.ndarray, equation_str: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.scatter(beta_nn, beta_formula, s=18, alpha=0.7)
    mx = max(beta_nn.max(), beta_formula.max()) * 1.03
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    r2 = r2_score(beta_nn, beta_formula)
    corr = pearsonr(beta_nn, beta_formula)[0] if np.std(beta_nn) > 0 and np.std(beta_formula) > 0 else np.nan
    title = f"PySR Formula vs NN beta\nR2={r2:.4f}, r={corr:.4f}"
    if equation_str:
        # Truncate long equations for title readability
        eq_display = equation_str if len(equation_str) < 80 else equation_str[:77] + "..."
        title += f"\n{eq_display}"
    ax.set_title(title, fontsize=9)
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

    beta_target, beta_max, best_eta = estimate_beta_series(
        gz["cases"].values.astype(float),
        gz["index_norm_city"].values.astype(float),
    )
    gz["beta_target"] = beta_target
    print(f"  SEIR inversion: beta_max={beta_max:.6f}, best_eta={best_eta} persons/day")
    train_mask = gz["year"].values != 2014

    model, losses = train_nn(x_norm, beta_target, train_mask)
    with torch.no_grad():
        beta_nn = model(torch.tensor(x_norm, dtype=torch.float32)).squeeze(-1).numpy()
    gz["beta_nn"] = beta_nn

    # SEIR forward reconstruction with learned beta
    m_norm_gz = gz["index_norm_city"].values.astype(float)
    pred_cases = seir_reconstruct_cases(beta_nn, beta_max, m_norm_gz, best_eta)
    gz["cases_pred_nn"] = pred_cases

    cases = gz["cases"].values.astype(float)
    phase1_metrics = evaluate_cases(cases[train_mask], pred_cases[train_mask])
    phase1_metrics["best_eta"] = best_eta
    pd.DataFrame([phase1_metrics]).to_csv(OUT / "phase1_metrics_data2.csv", index=False)

    # Save SEIR parameters
    pd.DataFrame([{
        "sigma_h": SIGMA_H,
        "gamma": GAMMA,
        "N_h": N_H,
        "days_per_month": DAYS_PER_MONTH,
        "best_eta": best_eta,
        "beta_max": beta_max,
    }]).to_csv(OUT / "phase1_seir_params.csv", index=False)

    # 2) Symbolic regression (PySR)
    print("Phase 2: Running PySR symbolic regression search ...")
    sr_model, f_metrics = fit_formula_pysr(weather, beta_nn)
    beta_formula_gz = predict_with_pysr(sr_model, weather)
    formula_obs_metrics = {
        "r2": float(r2_score(beta_nn, beta_formula_gz)),
        "corr": float(pearsonr(beta_nn, beta_formula_gz)[0]) if np.std(beta_nn) > 0 and np.std(beta_formula_gz) > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(beta_nn, beta_formula_gz))),
        "mae": float(np.mean(np.abs(beta_nn - beta_formula_gz))),
    }
    pd.DataFrame([f_metrics]).to_csv(OUT / "phase2_formula_fit_metrics_data2.csv", index=False)

    # Save Pareto front
    pareto_df = sr_model.equations_[["complexity", "loss", "score", "equation"]].copy()
    pareto_df.to_csv(OUT / "phase2_pysr_pareto.csv", index=False)

    # Save best equation (sympy + LaTeX)
    best_sympy = sr_model.sympy()
    best_latex = sr_model.latex()
    with open(OUT / "phase2_pysr_best_equation.txt", "w") as f:
        f.write(f"sympy: {best_sympy}\n")
        f.write(f"latex:  {best_latex}\n")
        f.write(f"\nmetrics: {f_metrics}\n")
    print(f"  Best equation: {best_sympy}")

    # 3) Transfer to all cities (no mechanism retraining)
    city_tables = []
    for city in sorted(monthly["city_en"].unique()):
        cdf = monthly[monthly["city_en"] == city].sort_values(["year", "month"]).copy()
        cdf = attach_mosquito_proxy(cdf, bi_proxy)
        city_tables.append(predict_city_risk_monthly(cdf, sr_model, beta_max, best_eta))
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
    plot_phase2_formula(beta_nn, beta_formula_gz, equation_str=str(best_sympy))
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
    print("1+3 pipeline on data_2 completed (SEIR ODE + PySR)")
    print("=" * 72)
    print(f"Year window: {YEAR_START}-{YEAR_END}")
    print(f"SEIR params:    sigma_h={SIGMA_H:.4f}, gamma={GAMMA:.4f}, N_h={N_H:.0f}, eta={best_eta}")
    print(f"Phase1 metrics: {OUT / 'phase1_metrics_data2.csv'}")
    print(f"Phase1 SEIR:    {OUT / 'phase1_seir_params.csv'}")
    print(f"Phase2 metrics: {OUT / 'phase2_formula_fit_metrics_data2.csv'}")
    print(f"Phase2 Pareto:  {OUT / 'phase2_pysr_pareto.csv'}")
    print(f"Phase2 best eq: {OUT / 'phase2_pysr_best_equation.txt'}")
    print(f"Transfer annual: {OUT / 'transfer_annual2014_data2.csv'}")
    print(f"Transfer metrics: {OUT / 'transfer_metrics_data2.csv'}")
    print(f"Figures: {OUT / 'phase1_guangzhou_data2.png'}, {OUT / 'phase2_formula_fit_data2.png'},")
    print(f"         {OUT / 'transfer_2014_scatter_data2.png'}, {OUT / 'transfer_2014_bars_data2.png'}")


if __name__ == "__main__":
    main()
