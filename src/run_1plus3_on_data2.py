#!/usr/bin/env python3
"""Run 1+3 pipeline: NN-coupled SEIR dynamics + PySR symbolic regression.

1) Single-city mechanism learning (Guangzhou):
   - NN(T,H,R) → β(t) embedded in SEIR ODE
   - Train end-to-end: loss on observed case counts (not on β)
   - η (import rate) learned jointly as a trainable parameter
2) Symbolic regression (PySR):
   - Search for optimal formula approximating NN-learned β(t)
3) Transfer:
   - Apply discovered formula to other cities via SEIR forward simulation
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from pysr import PySRRegressor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path("/root/wenmei")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "data2_1plus3"
YEAR_START = 2005
YEAR_END = 2019

# ── SEIR dynamics constants ──────────────────────────────────────────────
SIGMA_H = 1.0 / 5.9       # latent → infectious rate (day⁻¹)
GAMMA = 1.0 / 14.0        # recovery rate (day⁻¹)
N_H = 1.426e7             # Guangzhou resident population
DAYS_PER_MONTH = 30


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Norm:
    w_min: np.ndarray
    w_max: np.ndarray


# ── Neural network ───────────────────────────────────────────────────────

class TransmissionNN(nn.Module):
    """MLP: (T_norm, H_norm, R_norm) → β_norm ∈ (0, 1).

    Architecture: 3 → 16 → 16 → 1 (Softplus, Softplus, Sigmoid).
    Total NN parameters: 353.
    """

    def __init__(self, n_hidden: int = 16):
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


class SEIRCoupledModel(nn.Module):
    """End-to-end NN-coupled SEIR model.

    The NN predicts β_norm ∈ (0,1), scaled by a learnable ``beta_scale``.
    The import rate η is also learnable.  SEIR ODE is integrated with
    daily Euler steps, aggregated to monthly new case counts.
    """

    def __init__(self, n_hidden: int = 16, n_h: float = N_H):
        super().__init__()
        self.nn_beta = TransmissionNN(n_hidden)
        self.log_beta_scale = nn.Parameter(torch.tensor(1.0))
        self.log_eta = nn.Parameter(torch.tensor(-1.0))
        self.n_h = n_h

    @property
    def beta_scale(self) -> torch.Tensor:
        return torch.exp(self.log_beta_scale)

    @property
    def eta(self) -> torch.Tensor:
        return torch.exp(self.log_eta)

    def forward(
        self,
        weather_norm: torch.Tensor,
        m_norm: torch.Tensor,
        days: int = DAYS_PER_MONTH,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: NN → β → SEIR ODE → predicted monthly cases.

        Returns (pred_cases, beta) where beta is the physical-scale β(t).
        """
        beta_norm = self.nn_beta(weather_norm).squeeze(-1)  # (T,)
        beta = beta_norm * self.beta_scale
        eta = self.eta
        n_h = self.n_h

        T = beta.shape[0]
        S = torch.tensor(n_h - 1.0)
        E = torch.tensor(0.0)
        I = torch.tensor(1.0)
        R = torch.tensor(0.0)

        cases_list: List[torch.Tensor] = []
        for t in range(T):
            month_new = torch.tensor(0.0)
            b_t = beta[t]
            m_t = m_norm[t]
            for _ in range(days):
                lam = b_t * m_t * I / n_h
                lam = torch.clamp(lam, max=0.01)
                new_exp = lam * S

                dS = -new_exp
                dE = new_exp + eta - SIGMA_H * E
                dI = SIGMA_H * torch.clamp(E, min=0.0) - GAMMA * I
                dR = GAMMA * torch.clamp(I, min=0.0)

                month_new = month_new + SIGMA_H * torch.clamp(E, min=0.0)

                S = torch.clamp(S + dS, min=0.0)
                E = torch.clamp(E + dE, min=0.0)
                I = torch.clamp(I + dI, min=0.0)
                R = R + dR

            cases_list.append(month_new)

        return torch.stack(cases_list), beta


# ── Numpy SEIR forward (for inference / transfer) ────────────────────────

def seir_forward(
    beta: np.ndarray,
    m_norm: np.ndarray,
    eta: float,
    n_h: float = N_H,
    days: int = DAYS_PER_MONTH,
) -> np.ndarray:
    """Non-differentiable SEIR forward simulation (numpy)."""
    T = len(beta)
    S, E, I, R = n_h - 1.0, 0.0, 1.0, 0.0
    cases_out = np.zeros(T)
    for t in range(T):
        mc = 0.0
        b_t, m_t = beta[t], m_norm[t]
        for _ in range(days):
            lam = min(b_t * m_t * I / n_h, 0.01)
            ne = lam * S
            dS = -ne
            dE = ne + eta - SIGMA_H * E
            dI = SIGMA_H * max(E, 0.0) - GAMMA * I
            dR = GAMMA * max(I, 0.0)
            mc += SIGMA_H * max(E, 0.0)
            S = max(S + dS, 0.0)
            E = max(E + dE, 0.0)
            I = max(I + dI, 0.0)
            R = R + dR
        cases_out[t] = mc
    return cases_out


# ── Data loading ─────────────────────────────────────────────────────────

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


def weather_norm(weather: np.ndarray, norm: Norm) -> np.ndarray:
    w_range = np.maximum(norm.w_max - norm.w_min, 1e-8)
    return np.clip((weather - norm.w_min) / w_range, 0.0, 1.0)


# ── Training ─────────────────────────────────────────────────────────────

def train_coupled_model(
    weather_norm_np: np.ndarray,
    m_norm_np: np.ndarray,
    cases_np: np.ndarray,
    train_mask_np: np.ndarray,
    n_epochs: int = 3000,
    lr: float = 1e-3,
    alpha: float = 0.5,
) -> Tuple[SEIRCoupledModel, List[float]]:
    """Train NN-coupled SEIR end-to-end on observed case counts."""
    model = SEIRCoupledModel(n_hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    w_t = torch.tensor(weather_norm_np, dtype=torch.float32)
    m_t = torch.tensor(m_norm_np, dtype=torch.float32)
    c_t = torch.tensor(cases_np, dtype=torch.float32)
    mask = torch.tensor(train_mask_np, dtype=torch.bool)

    best_state = None
    best_loss = float("inf")
    losses: List[float] = []

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()

        pred_cases, _ = model(w_t, m_t)

        p = pred_cases[mask]
        o = c_t[mask]

        loss_mse = torch.mean((torch.log1p(p) - torch.log1p(o)) ** 2)

        if p.std() > 1e-6 and o.std() > 1e-6:
            pn = (p - p.mean()) / (p.std() + 1e-6)
            on = (o - o.mean()) / (o.std() + 1e-6)
            loss_corr = -torch.mean(pn * on)
        else:
            loss_corr = torch.tensor(0.0)

        loss = loss_mse + alpha * loss_corr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        sched.step()

        lv = float(loss.item())
        losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 500 == 0:
            print(
                f"  Epoch {epoch + 1}/{n_epochs}: loss={lv:.4f}, "
                f"beta_scale={model.beta_scale.item():.4f}, "
                f"eta={model.eta.item():.4f} persons/day"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, losses


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_cases(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.maximum(y_pred, 0.0)
    out: Dict[str, float] = {
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


# ── PySR symbolic regression ────────────────────────────────────────────

def fit_formula_pysr(
    weather_raw: np.ndarray,
    beta: np.ndarray,
    niterations: int = 200,
    timeout: int = 600,
) -> Tuple[PySRRegressor, Dict[str, float]]:
    """Use PySR to search for the best symbolic formula: β = f(T, H, R)."""
    sr = PySRRegressor(
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
    sr.fit(weather_raw, beta, variable_names=["T", "H", "R"])

    pred = np.maximum(sr.predict(weather_raw), 0.0)
    met = {
        "r2": float(r2_score(beta, pred)),
        "corr": float(pearsonr(beta, pred)[0]) if np.std(pred) > 0 and np.std(beta) > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(beta, pred))),
        "mae": float(np.mean(np.abs(beta - pred))),
    }
    return sr, met


def predict_with_pysr(sr: PySRRegressor, weather_raw: np.ndarray) -> np.ndarray:
    return np.maximum(sr.predict(weather_raw), 0.0)


# ── Transfer to other cities ────────────────────────────────────────────

def predict_city_risk_monthly(
    city_df: pd.DataFrame,
    sr_model: PySRRegressor,
    eta: float,
    n_h: float = N_H,
) -> pd.DataFrame:
    """Predict monthly cases for a city using PySR formula + SEIR forward."""
    weather = city_df[["tem", "rhu", "pre"]].values.astype(float)
    beta = predict_with_pysr(sr_model, weather)
    m_norm = city_df["index_norm_city"].values.astype(float)
    pred_cases = seir_forward(beta, m_norm, eta, n_h)

    out = city_df[["city_en", "year", "month", "cases", "tem", "rhu", "pre", "index_norm_city"]].copy()
    out["beta_formula"] = beta
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

        gz = annual[annual["city_en"] == "Guangzhou"]
        if not gz.empty and float(gz["risk_2014"].iloc[0]) > 0:
            s_gz = float(gz["cases_2014"].iloc[0] / gz["risk_2014"].iloc[0])
        else:
            s_gz = np.nan
        pred_gz = x * s_gz

        non_gz = annual[annual["city_en"] != "Guangzhou"]
        xn = non_gz["risk_2014"].values.astype(float)
        yn = non_gz["cases_2014"].values.astype(float)
        s_nogz = float(np.sum(xn * yn) / (np.sum(xn**2) + 1e-12))
        pred_nogz = x * s_nogz

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


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_phase1(
    gz: pd.DataFrame,
    pred_cases: np.ndarray,
    beta: np.ndarray,
    phase1_metrics: Dict[str, float],
    losses: List[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()
    t = np.arange(len(gz))
    y = gz["cases"].values.astype(float)
    m = gz["year"].values.astype(int) != 2014

    axes[0].plot(t, y, "k-", lw=1.8, label="Observed")
    axes[0].plot(t, pred_cases, "r-", lw=1.3, label="Pred (SEIR+NN)")
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

    axes[2].plot(gz["date"], beta, "b-", lw=1.6, label="beta (NN+scale)")
    axes[2].set_title("Learned transmission coefficient beta(t)")
    axes[2].set_ylabel("beta (day^-1)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    axes[3].plot(losses, lw=0.8)
    axes[3].set_yscale("log")
    axes[3].set_title("Training loss (log1p MSE + corr)")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "phase1_guangzhou_data2.png", dpi=220)
    plt.close(fig)


def plot_phase2_formula(beta: np.ndarray, beta_formula: np.ndarray, equation_str: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.scatter(beta, beta_formula, s=18, alpha=0.7)
    mx = max(beta.max(), beta_formula.max()) * 1.03
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    r2 = r2_score(beta, beta_formula)
    corr = pearsonr(beta, beta_formula)[0] if np.std(beta) > 0 and np.std(beta_formula) > 0 else np.nan
    title = f"PySR Formula vs NN beta\nR2={r2:.4f}, r={corr:.4f}"
    if equation_str:
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


# ── Main pipeline ────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    monthly, bi_proxy = load_data()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    if monthly.empty:
        raise RuntimeError("No monthly data left after year filtering.")
    if 2014 not in set(monthly["year"].unique()):
        raise RuntimeError("Year 2014 not available.")

    # ── Phase 1: Single-city mechanism learning (Guangzhou) ──────────
    print("Phase 1: Training NN-coupled SEIR model on Guangzhou ...")
    gz = monthly[monthly["city_en"] == "Guangzhou"].sort_values("date").copy()
    gz = attach_mosquito_proxy(gz, bi_proxy)

    weather_raw = gz[["tem", "rhu", "pre"]].values.astype(float)
    norm = Norm(w_min=weather_raw.min(axis=0), w_max=weather_raw.max(axis=0))
    x_norm = weather_norm(weather_raw, norm)
    m_norm_gz = gz["index_norm_city"].values.astype(float)
    cases_obs = gz["cases"].values.astype(float)
    train_mask = gz["year"].values != 2014

    model, losses = train_coupled_model(x_norm, m_norm_gz, cases_obs, train_mask)

    with torch.no_grad():
        pred_cases_t, beta_t = model(
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(m_norm_gz, dtype=torch.float32),
        )
        pred_cases = pred_cases_t.numpy()
        beta_learned = beta_t.numpy()

    eta_val = float(model.eta.item())
    beta_scale_val = float(model.beta_scale.item())
    print(f"  Learned: beta_scale={beta_scale_val:.4f}, eta={eta_val:.4f} persons/day")

    gz["beta_learned"] = beta_learned
    gz["cases_pred_nn"] = pred_cases

    phase1_metrics = evaluate_cases(cases_obs[train_mask], pred_cases[train_mask])
    phase1_metrics["beta_scale"] = beta_scale_val
    phase1_metrics["eta"] = eta_val
    pd.DataFrame([phase1_metrics]).to_csv(OUT / "phase1_metrics_data2.csv", index=False)

    pd.DataFrame([{
        "sigma_h": SIGMA_H,
        "gamma": GAMMA,
        "N_h": N_H,
        "days_per_month": DAYS_PER_MONTH,
        "beta_scale": beta_scale_val,
        "eta": eta_val,
    }]).to_csv(OUT / "phase1_seir_params.csv", index=False)

    # ── Phase 2: Symbolic regression (PySR) ──────────────────────────
    print("Phase 2: Running PySR symbolic regression search ...")
    sr_model, f_metrics = fit_formula_pysr(weather_raw, beta_learned)
    beta_formula_gz = predict_with_pysr(sr_model, weather_raw)

    pd.DataFrame([f_metrics]).to_csv(OUT / "phase2_formula_fit_metrics_data2.csv", index=False)

    pareto_df = sr_model.equations_[["complexity", "loss", "score", "equation"]].copy()
    pareto_df.to_csv(OUT / "phase2_pysr_pareto.csv", index=False)

    best_sympy = sr_model.sympy()
    best_latex = sr_model.latex()
    with open(OUT / "phase2_pysr_best_equation.txt", "w") as f:
        f.write(f"sympy: {best_sympy}\n")
        f.write(f"latex:  {best_latex}\n")
        f.write(f"\nmetrics: {f_metrics}\n")
    print(f"  Best equation: {best_sympy}")

    # ── Phase 3: Transfer to all cities ──────────────────────────────
    print("Phase 3: Transferring to 16 cities ...")
    city_tables = []
    for city in sorted(monthly["city_en"].unique()):
        cdf = monthly[monthly["city_en"] == city].sort_values(["year", "month"]).copy()
        cdf = attach_mosquito_proxy(cdf, bi_proxy)
        city_tables.append(predict_city_risk_monthly(cdf, sr_model, eta_val))
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

    # ── Figures ──────────────────────────────────────────────────────
    plot_phase1(gz, pred_cases, beta_learned, phase1_metrics, losses)
    plot_phase2_formula(beta_learned, beta_formula_gz, equation_str=str(best_sympy))
    plot_transfer(annual2014)

    gz[
        [
            "city_en", "year", "month", "date", "cases",
            "tem", "rhu", "pre", "index_norm_city",
            "beta_learned", "cases_pred_nn",
        ]
    ].to_csv(OUT / "phase1_guangzhou_predictions_data2.csv", index=False)

    print("=" * 72)
    print("1+3 pipeline completed (end-to-end NN-SEIR + PySR)")
    print("=" * 72)
    print(f"SEIR: sigma_h={SIGMA_H:.4f}, gamma={GAMMA:.4f}, N_h={N_H:.0f}")
    print(f"Learned: beta_scale={beta_scale_val:.4f}, eta={eta_val:.4f} persons/day")
    print(f"Phase1: {OUT / 'phase1_metrics_data2.csv'}")
    print(f"Phase2: {OUT / 'phase2_pysr_pareto.csv'}")
    print(f"Phase2: {OUT / 'phase2_pysr_best_equation.txt'}")
    print(f"Transfer: {OUT / 'transfer_metrics_data2.csv'}")


if __name__ == "__main__":
    main()
