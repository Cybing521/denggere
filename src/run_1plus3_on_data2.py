#!/usr/bin/env python3
"""Run 1+3 pipeline: SEIR dynamics + NN + symbolic regression.

Architecture:
  - SEIR ODE inversion to estimate beta(t) from observed cases (bisection)
  - NN learns beta = f(weather, month, M) in log-space
  - Discrete-time formula for case prediction (avoids SEIR forward instability)
  - R0 analysis from SEIR-derived beta
  - Physics + polynomial dual-track formula discovery
  - Leave-One-Year-Out cross-validation
  - Baseline model comparisons
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
from scipy.optimize import brentq, minimize
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/root/wenmei")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "data2_1plus3"

# ── SEIR dynamics constants ──────────────────────────────────────────────
SIGMA_H = 1.0 / 5.9       # latent -> infectious (day^-1)
GAMMA = 1.0 / 14.0        # recovery rate (day^-1)
N_H = 1.426e7              # Guangzhou population
DAYS_PER_MONTH = 30
ETA_CANDIDATES = [0.01, 0.02, 0.05, 0.1, 0.2]


# ── SEIR ODE ─────────────────────────────────────────────────────────────

def seir_forward_step(beta, m_t, eta, s, e, i, r, days=DAYS_PER_MONTH):
    """One period of SEIR with daily Euler steps (normalized state s,e,i,r)."""
    mc = 0.0
    for _ in range(days):
        lam = beta * m_t * i
        ne = lam * s
        ds = -ne
        de = ne + eta / N_H - SIGMA_H * e
        di = SIGMA_H * max(e, 0) - GAMMA * i
        dr = GAMMA * max(i, 0)
        mc += SIGMA_H * max(e, 0) * N_H
        s = max(s + ds, 0); e = max(e + de, 0); i = max(i + di, 0); r = r + dr
    return mc, s, e, i, r


def invert_beta_seir(cases, m_norm, eta, days=DAYS_PER_MONTH):
    """Invert SEIR ODE: find beta(t) that reproduces observed cases via bisection."""
    T = len(cases)
    s, e, i, r = 1 - 1 / N_H, 0.0, 1 / N_H, 0.0
    betas = np.zeros(T)
    for t in range(T):
        tgt, mt = float(cases[t]), float(m_norm[t])
        if tgt <= 0:
            _, s, e, i, r = seir_forward_step(0, mt, eta, s, e, i, r, days)
            continue
        def res(b):
            mc, *_ = seir_forward_step(b, mt, eta, s, e, i, r, days)
            return mc - tgt
        mc_hi, *_ = seir_forward_step(200, mt, eta, s, e, i, r, days)
        if mc_hi < tgt:
            betas[t] = 200
            _, s, e, i, r = seir_forward_step(200, mt, eta, s, e, i, r, days)
            continue
        try:
            b = brentq(res, 0, 200, xtol=1e-4, maxiter=200)
        except ValueError:
            b = 0
        betas[t] = b
        _, s, e, i, r = seir_forward_step(b, mt, eta, s, e, i, r, days)
    return betas


def seir_forward(beta_series, m_norm, eta, days=DAYS_PER_MONTH):
    """Forward SEIR simulation (for R0 analysis, not case prediction)."""
    T = len(beta_series)
    s, e, i, r = 1 - 1 / N_H, 0.0, 1 / N_H, 0.0
    cases_out = np.zeros(T)
    for t in range(T):
        mc, s, e, i, r = seir_forward_step(
            float(beta_series[t]), float(m_norm[t]), eta, s, e, i, r, days)
        cases_out[t] = mc
    return cases_out


# ── Discrete-time case reconstruction ────────────────────────────────────

def reconstruct_cases(beta, m_norm, cases_obs):
    """Predict cases using discrete formula: pred(t) = beta(t)*M(t)*pool(t)."""
    T = len(beta)
    pred = np.zeros(T)
    for t in range(T):
        pool = max(cases_obs[t-1] + 0.3 * (cases_obs[t-2] if t >= 2 else 0), 1.0) if t >= 1 else 1.0
        pred[t] = beta[t] * m_norm[t] * pool
    return pred


# ── Data loading ─────────────────────────────────────────────────────────

def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)

def load_data():
    monthly = pd.read_csv(DATA / "cases_weather_monthly_utf8.csv")
    bi_proxy = pd.read_csv(DATA / "bi_guangdong_monthly_proxy.csv")
    return monthly, bi_proxy

@dataclass
class Norm:
    w_min: np.ndarray; w_max: np.ndarray

def weather_norm(raw, norm):
    rng = norm.w_max - norm.w_min; rng[rng == 0] = 1.0
    return (raw - norm.w_min) / rng

def build_extended_features(w_norm, months, m_norm):
    T = len(w_norm)
    rad = 2 * np.pi * months / 12.0
    t_lag = np.zeros(T); t_lag[1:] = w_norm[:-1, 0]; t_lag[0] = w_norm[0, 0]
    r_lag = np.zeros(T); r_lag[1:] = w_norm[:-1, 2]; r_lag[0] = w_norm[0, 2]
    m_feat = m_norm / (m_norm.max() + 1e-10)
    return np.column_stack([w_norm, np.sin(rad).reshape(-1,1), np.cos(rad).reshape(-1,1),
                            t_lag.reshape(-1,1), r_lag.reshape(-1,1), m_feat.reshape(-1,1)])

def attach_mosquito_proxy(city_df, bi_proxy, city):
    bi_city = bi_proxy[bi_proxy["city_en"] == city][["year", "month", "index_norm_city"]].copy()
    out = city_df.merge(bi_city, on=["year", "month"], how="left")
    if out["index_norm_city"].notna().any():
        out["index_norm_city"] = out["index_norm_city"].interpolate(limit_direction="both").fillna(
            out["index_norm_city"].median())
    else:
        out["index_norm_city"] = 1.0
    bi_mean = out["index_norm_city"].mean()
    if bi_mean > 0: out["index_norm_city"] = out["index_norm_city"] / bi_mean
    out["index_norm_city"] = np.clip(out["index_norm_city"], 0.1, 20.0)
    return out


# ── NN ───────────────────────────────────────────────────────────────────

class TransmissionNN(nn.Module):
    def __init__(self, n_input=8, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Softplus(),
            nn.Linear(n_hidden, n_hidden), nn.Softplus(),
            nn.Linear(n_hidden, 1), nn.Softplus())
    def forward(self, x): return self.net(x)

def train_nn_beta(x_ext, beta_target, train_mask, n_epochs=2000, lr=5e-3):
    model = TransmissionNN(n_input=x_ext.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    x_t = torch.tensor(x_ext, dtype=torch.float32)
    y_t = torch.tensor(np.log1p(beta_target).astype(np.float32))
    mask = torch.tensor(train_mask, dtype=torch.bool)
    huber = nn.SmoothL1Loss()
    best_loss, best_state, losses = float("inf"), None, []
    for epoch in range(n_epochs):
        model.train(); opt.zero_grad()
        pred = model(x_t).squeeze(-1); p, o = pred[mask], y_t[mask]
        loss_h = huber(p, o)
        loss_c = -torch.mean((p-p.mean())/(p.std()+1e-6)*(o-o.mean())/(o.std()+1e-6)) if p.std()>1e-6 and o.std()>1e-6 else torch.tensor(0.0)
        loss = loss_h + 0.5 * loss_c
        loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
        if loss.item() < best_loss: best_loss = loss.item(); best_state = {k:v.clone() for k,v in model.state_dict().items()}
        if (epoch+1) % 500 == 0: print(f"  NN Epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}", flush=True)
    if best_state: model.load_state_dict(best_state)
    return model, losses

def nn_predict_beta(model, x_ext):
    with torch.no_grad():
        log_b = model(torch.tensor(x_ext, dtype=torch.float32)).squeeze(-1).numpy()
    return np.expm1(np.maximum(log_b, 0.0))


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_cases(obs, pred):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    v = np.isfinite(o) & np.isfinite(p); o, p = o[v], p[v]
    if len(o) < 3 or np.std(o) == 0 or np.std(p) == 0:
        return {k: np.nan for k in ["pearson_r","spearman_rho","kendall_tau","r2_log","mae","rmse","wape","rmsle"]}
    lo, lp = np.log1p(o), np.log1p(p)
    return {"pearson_r": pearsonr(o,p)[0], "spearman_rho": spearmanr(o,p)[0],
            "kendall_tau": kendalltau(o,p)[0],
            "r2_log": 1-np.sum((lo-lp)**2)/(np.sum((lo-lo.mean())**2)+1e-12),
            "mae": float(np.mean(np.abs(o-p))), "rmse": float(np.sqrt(np.mean((o-p)**2))),
            "wape": float(np.sum(np.abs(o-p))/(np.sum(np.abs(o))+1e-12)),
            "rmsle": float(np.sqrt(np.mean((lo-lp)**2)))}


# ── Formula models ───────────────────────────────────────────────────────

class PolyFormulaModel:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.reg = LinearRegression(fit_intercept=False)
    def fit(self, X, y): self.poly.fit_transform(X); self.reg.fit(self.poly.transform(X), y); return self
    def predict(self, X): return np.maximum(self.reg.predict(self.poly.transform(X)), 0.0)
    def equation_str(self):
        names = self.poly.get_feature_names_out(["T","H","R"]).tolist()
        parts = [f"{c:.6e}*{n}" if n!="1" else f"{c:.6e}" for n,c in zip(names,self.reg.coef_) if abs(c)>1e-15]
        return "max(0, " + " + ".join(parts) + ")"
    def get_coefficients(self):
        return dict(zip(self.poly.get_feature_names_out(["T","H","R"]).tolist(), self.reg.coef_.tolist()))
    @property
    def n_params(self): return len(self.reg.coef_)

class PhysicsFormulaModel:
    param_names = ["beta0","T_opt","sigma_T","H_opt","sigma_H","k_R"]
    def __init__(self): self.params = None
    @staticmethod
    def _forward(X, p):
        b0,To,sT,Ho,sH,kR = p; T,H,R = X[:,0],X[:,1],X[:,2]
        return np.maximum(b0*np.exp(-0.5*((T-To)/(sT+1e-6))**2)*np.exp(-0.5*((H-Ho)/(sH+1e-6))**2)*(1-np.exp(-max(kR,1e-6)*np.maximum(R,0))), 0)
    def fit(self, X, y):
        def _loss(p): return float(np.mean((self._forward(X,p)-y)**2))
        res = minimize(_loss, [2,27,5,78,15,0.01], method="L-BFGS-B",
                       bounds=[(0.01,50),(15,40),(1,30),(50,100),(3,50),(1e-4,1)], options={"maxiter":5000})
        self.params = res.x; return self
    def predict(self, X): return self._forward(X, self.params)
    def equation_str(self):
        b0,To,sT,Ho,sH,kR = self.params
        return f"max(0, {b0:.4f}*exp(-0.5*((T-{To:.1f})/{sT:.1f})^2)*exp(-0.5*((H-{Ho:.1f})/{sH:.1f})^2)*(1-exp(-{kR:.4f}*R)))"
    def get_params_dict(self): return dict(zip(self.param_names, self.params.tolist()))
    @property
    def n_params(self): return 6

def fit_both_formulas(weather_raw, beta):
    poly = PolyFormulaModel().fit(weather_raw, beta)
    phys = PhysicsFormulaModel().fit(weather_raw, beta)
    n = len(beta); results = {}
    for name, model in [("polynomial",poly),("physics",phys)]:
        pred = model.predict(weather_raw); ss = np.sum((beta-pred)**2); k = model.n_params
        results[name] = {"r2": float(r2_score(beta,pred)),
            "corr": float(pearsonr(beta,pred)[0]) if np.std(pred)>0 else np.nan,
            "rmse": float(np.sqrt(np.mean((beta-pred)**2))), "mae": float(np.mean(np.abs(beta-pred))),
            "aic": float(n*np.log(ss/n+1e-30)+2*k), "bic": float(n*np.log(ss/n+1e-30)+k*np.log(n)),
            "n_params": k}
    return poly, phys, results


# ── Transfer + baselines ─────────────────────────────────────────────────

def predict_city_monthly(city_df, formula_model):
    w = city_df[["tem","rhu","pre"]].values.astype(float)
    beta = formula_model.predict(w)
    m = city_df["index_norm_city"].values.astype(float)
    pred = reconstruct_cases(beta, m, city_df["cases"].values.astype(float))
    out = city_df[["city_en","year","month","cases","tem","rhu","pre","index_norm_city"]].copy()
    out["beta_formula"] = beta; out["risk_monthly"] = pred
    return out

def baseline_historical_mean(cdf):
    train = cdf[cdf["year"]!=2014]
    mm = train.groupby("month")["cases"].mean().to_dict()
    return cdf["month"].map(mm).fillna(0).values

def baseline_linear_regression(cdf):
    train = cdf[cdf["year"]!=2014]
    reg = LinearRegression().fit(train[["tem","rhu","pre"]].values, train["cases"].values)
    return np.maximum(reg.predict(cdf[["tem","rhu","pre"]].values), 0)

def baseline_seasonal_naive(cdf):
    pred = np.zeros(len(cdf))
    for idx in range(len(cdf)):
        row = cdf.iloc[idx]
        prev = cdf[(cdf["year"]==row["year"]-1)&(cdf["month"]==row["month"])]
        if len(prev)>0: pred[idx] = prev["cases"].values[0]
    return pred


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_phase1(gz, pred_cases, beta, metrics, losses):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    t = np.arange(len(gz)); obs = gz["cases"].values
    ax = axes[0,0]; ax.plot(t,obs,"k-",lw=1.2,label="Observed"); ax.plot(t,pred_cases,"r-",lw=0.9,alpha=0.8,label="Predicted")
    ax.set_yscale("symlog",linthresh=1); ax.set_ylabel("Cases (symlog)")
    ax.set_title(f"(a) Guangzhou  r={metrics['pearson_r']:.3f}  rho={metrics['spearman_rho']:.3f}"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax = axes[0,1]; ax.plot(t,beta,"b-",lw=1); ax.set_ylabel("beta'"); ax.set_title("(b) SEIR-inverted beta'(t)"); ax.grid(alpha=0.2)
    m14 = gz["year"].values==2014
    if m14.any(): ax.axvspan(t[m14][0],t[m14][-1],alpha=0.15,color="red",label="2014"); ax.legend(fontsize=8)
    ax = axes[1,0]; ax.plot(losses,"g-",lw=0.6); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("(c) NN training loss"); ax.set_yscale("log"); ax.grid(alpha=0.2)
    ax = axes[1,1]; ax.scatter(np.log1p(obs),np.log1p(pred_cases),s=12,alpha=0.6)
    mx = max(np.log1p(obs).max(),np.log1p(pred_cases).max()); ax.plot([0,mx],[0,mx],"k--",lw=0.8)
    ax.set_xlabel("log(1+obs)"); ax.set_ylabel("log(1+pred)"); ax.set_title(f"(d) R2_log={metrics['r2_log']:.3f}"); ax.grid(alpha=0.2)
    fig.suptitle("Phase 1: SEIR-coupled NN (Guangzhou monthly)", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(OUT/"phase1_guangzhou_data2.png",dpi=200); plt.close(fig)

def plot_cv_boxplot(cv_df):
    fig, axes = plt.subplots(1,4,figsize=(14,4))
    for i,col in enumerate(["pearson_r","spearman_rho","r2_log","mae"]):
        ax = axes[i]; ax.boxplot(cv_df[col].dropna().values); ax.set_title(col); ax.grid(alpha=0.2)
        ax.axhline(cv_df[col].mean(),color="red",ls="--",lw=0.8,label=f"mean={cv_df[col].mean():.3f}"); ax.legend(fontsize=7)
    fig.suptitle("Leave-One-Year-Out Cross-Validation",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"phase1_cv_boxplot.png",dpi=200); plt.close(fig)

def plot_r0_analysis(gz, R0):
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    t = np.arange(len(gz))
    ax = axes[0]; ax.plot(t,R0,"b-",lw=1); ax.axhline(1,color="red",ls="--",lw=1,label="R0=1")
    ax.fill_between(t,0,R0,where=R0>1,alpha=0.2,color="red",label="R0>1")
    ax.set_ylabel("R0"); ax.set_title("(a) R0 time series"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    m14 = gz["year"].values==2014
    if m14.any(): ax.axvspan(t[m14][0],t[m14][-1],alpha=0.1,color="orange")
    ax = axes[1]
    for y in sorted(gz["year"].unique()):
        mask = gz["year"].values==y; months = gz["month"].values[mask]; r0y = R0[mask]
        c = "red" if y==2014 else "steelblue"; a = 1.0 if y==2014 else 0.3; lw = 2 if y==2014 else 0.7
        ax.plot(months,r0y,color=c,alpha=a,lw=lw,label=str(y) if y==2014 else None)
    ax.axhline(1,color="red",ls="--",lw=0.8)
    ax.set_xlabel("Month"); ax.set_ylabel("R0"); ax.set_title("(b) R0 seasonal profile"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("R0 = beta' * M_hat / gamma",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"phase1_R0_analysis.png",dpi=200); plt.close(fig)

def plot_phase2_dual(beta_inv, beta_poly, beta_phys, comp):
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    t = np.arange(len(beta_inv))
    ax = axes[0]; ax.plot(t,beta_inv,"k-",lw=1.2,label="NN"); ax.plot(t,beta_poly,"b--",lw=0.9,label=f"Poly R2={comp['polynomial']['r2']:.3f}")
    ax.plot(t,beta_phys,"r:",lw=0.9,label=f"Physics R2={comp['physics']['r2']:.3f}")
    ax.set_ylabel("beta'"); ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.set_title("(a) Time series")
    ax = axes[1]; ax.scatter(beta_inv,beta_poly,s=10,alpha=0.5,label="Poly"); ax.scatter(beta_inv,beta_phys,s=10,alpha=0.5,label="Physics")
    mx = max(beta_inv.max(),beta_poly.max(),beta_phys.max()); ax.plot([0,mx],[0,mx],"k--",lw=0.8)
    ax.set_xlabel("NN beta'"); ax.set_ylabel("Formula"); ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.set_title("(b) Scatter")
    ax = axes[2]; names = list(comp.keys()); x_pos = np.arange(len(names))
    ax.bar(x_pos-0.15,[comp[n]["r2"] for n in names],0.3,label="R2")
    ax.bar(x_pos+0.15,[comp[n]["bic"]/max(abs(comp[n]["bic"]) for n in names) for n in names],0.3,label="BIC(norm)")
    ax.set_xticks(x_pos); ax.set_xticklabels(names); ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.set_title("(c) Comparison")
    fig.suptitle("Phase 2: Formula discovery",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"phase2_formula_fit_data2.png",dpi=200); plt.close(fig)

def plot_response_surface(phys, poly, weather_raw):
    R_med = np.median(weather_raw[:,2])
    Tr = np.linspace(weather_raw[:,0].min(),weather_raw[:,0].max(),50)
    Hr = np.linspace(weather_raw[:,1].min(),weather_raw[:,1].max(),50)
    TT,HH = np.meshgrid(Tr,Hr); grid = np.column_stack([TT.ravel(),HH.ravel(),np.full(TT.size,R_med)])
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax,m,nm in [(axes[0],poly,"Polynomial"),(axes[1],phys,"Physics")]:
        Z = m.predict(grid).reshape(TT.shape)
        cs = ax.contourf(TT,HH,Z,levels=20,cmap="YlOrRd"); plt.colorbar(cs,ax=ax,label="beta'")
        ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Humidity (%)"); ax.set_title(f"{nm} (R={R_med:.0f}mm)")
    fig.suptitle("Beta response surface",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"phase2_response_surface.png",dpi=200); plt.close(fig)

def plot_transfer(annual):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    obs,pred = annual["cases_2014"].values, annual["risk_2014"].values
    ax = axes[0]; ax.scatter(np.log1p(obs),np.log1p(pred),s=30,alpha=0.7)
    for _,row in annual.iterrows(): ax.annotate(row["city_en"][:4],(np.log1p(row["cases_2014"]),np.log1p(row["risk_2014"])),fontsize=6)
    mx = max(np.log1p(obs).max(),np.log1p(pred).max())*1.05; ax.plot([0,mx],[0,mx],"k--",lw=0.8)
    rho,p = spearmanr(obs,pred); ax.set_xlabel("log(1+obs)"); ax.set_ylabel("log(1+pred)"); ax.set_title(f"2014 rho={rho:.3f}"); ax.grid(alpha=0.2)
    ax = axes[1]; idx = np.argsort(obs)[::-1]; cities = annual["city_en"].values[idx]
    ax.barh(range(len(cities)),np.log1p(obs[idx]),alpha=0.5,label="Obs"); ax.barh(range(len(cities)),np.log1p(pred[idx]),alpha=0.5,label="Pred")
    ax.set_yticks(range(len(cities))); ax.set_yticklabels(cities,fontsize=7); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT/"transfer_2014_scatter_data2.png",dpi=200); plt.close(fig)

def plot_outbreak_2014_beta(gz):
    gz = gz.copy(); is2014 = gz["year"]==2014
    other = gz[~is2014].groupby("month")["beta_inverted"].agg(["mean","std"]).reset_index()
    y2014 = gz[is2014][["month","beta_inverted"]].sort_values("month")
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.fill_between(other["month"],other["mean"]-other["std"],other["mean"]+other["std"],alpha=0.25,color="steelblue",label="Other years")
    ax.plot(other["month"],other["mean"],"b-",lw=1.5)
    ax.plot(y2014["month"].values,y2014["beta_inverted"].values,"r-o",lw=2,ms=5,label="2014")
    ax.set_xlabel("Month"); ax.set_ylabel("beta'"); ax.set_title("Monthly beta': 2014 vs other years")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_xticks(range(1,13))
    fig.tight_layout(); fig.savefig(OUT/"outbreak_2014_beta_compare_data2.png",dpi=220); plt.close(fig)

def plot_all_cities_grid(tm):
    cities = sorted(tm["city_en"].unique()); ncols,nrows = 4,(len(cities)+3)//4
    fig, axes = plt.subplots(nrows,ncols,figsize=(16,3.2*nrows)); axes = axes.ravel()
    for i,city in enumerate(cities):
        ax = axes[i]; cdf = tm[tm["city_en"]==city].sort_values(["year","month"]); t = np.arange(len(cdf))
        ax.plot(t,cdf["cases"].values,"k-",lw=1.2,label="Obs"); ax.plot(t,cdf["risk_monthly"].values,"r-",lw=0.9,alpha=0.8,label="Pred")
        ax.set_title(city,fontsize=9); ax.set_yscale("symlog",linthresh=1); ax.grid(alpha=0.2)
        if i==0: ax.legend(fontsize=7)
    for j in range(i+1,len(axes)): axes[j].set_visible(False)
    fig.suptitle("16 cities: observed vs predicted",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/"all_cities_fit_grid.png",dpi=180); plt.close(fig)

def plot_baseline_comparison(bdf):
    fig, axes = plt.subplots(1,3,figsize=(14,4.5)); models = bdf["model"].unique(); x = np.arange(len(models))
    for i,metric in enumerate(["spearman_rho","r2_log","wape"]):
        ax = axes[i]; vals = [bdf[bdf["model"]==m][metric].mean() for m in models]
        bars = ax.bar(x,vals,alpha=0.7); ax.set_xticks(x); ax.set_xticklabels(models,rotation=25,fontsize=8)
        ax.set_title(metric); ax.grid(alpha=0.2)
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{v:.3f}",ha="center",va="bottom",fontsize=7)
    fig.suptitle("Model comparison (16-city average)",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"transfer_baseline_comparison.png",dpi=200); plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    set_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    monthly, bi_proxy = load_data()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"],month=monthly["month"],day=1))

    gz = monthly[monthly["city_en"]=="Guangzhou"].sort_values("date").copy()
    gz = attach_mosquito_proxy(gz, bi_proxy, "Guangzhou")
    weather_raw = gz[["tem","rhu","pre"]].values.astype(float)
    norm_obj = Norm(w_min=weather_raw.min(axis=0), w_max=weather_raw.max(axis=0))
    x_norm = weather_norm(weather_raw, norm_obj)
    m_norm_gz = gz["index_norm_city"].values.astype(float)
    cases_obs = gz["cases"].values.astype(float)
    train_mask = gz["year"].values != 2014

    # ── Phase 1a: SEIR ODE inversion ─────────────────────────────────
    print("Phase 1a: SEIR ODE inversion — estimating beta(t) ...", flush=True)
    best_eta, best_r2, best_beta = None, -np.inf, None
    for eta_c in ETA_CANDIDATES:
        beta_c = invert_beta_seir(cases_obs, m_norm_gz, eta_c)
        pred_c = seir_forward(beta_c, m_norm_gz, eta_c)
        o, p = cases_obs[train_mask], pred_c[train_mask]
        lo, lp = np.log1p(o), np.log1p(p)
        r2 = 1 - np.sum((lo-lp)**2)/(np.sum((lo-lo.mean())**2)+1e-12)
        print(f"  eta={eta_c:.3f}: R2_log={r2:.4f}, beta=[{beta_c.min():.3f},{beta_c.max():.3f}]", flush=True)
        if r2 > best_r2: best_r2, best_eta, best_beta = r2, eta_c, beta_c
    eta_val = best_eta
    beta_raw = best_beta
    beta_smooth = gaussian_filter1d(beta_raw, sigma=2.0)
    beta_smooth = np.maximum(beta_smooth, 0.0)
    print(f"  Best eta={eta_val}, R2_log={best_r2:.4f}", flush=True)

    # R0 analysis
    R0 = beta_smooth * m_norm_gz / GAMMA
    gz["beta_inverted"] = beta_raw
    gz["beta_smooth"] = beta_smooth
    gz["R0"] = R0

    pd.DataFrame([{"sigma_h":SIGMA_H,"gamma":GAMMA,"N_h":N_H,"best_eta":eta_val,
                    "beta_max":beta_raw.max(),"R0_mean":R0.mean(),"R0_max":R0.max()}]).to_csv(
        OUT/"phase1_seir_params.csv", index=False)

    # ── Phase 1b: Train NN ───────────────────────────────────────────
    print("Phase 1b: Training NN beta = f(weather, month, M) ...", flush=True)
    x_ext = build_extended_features(x_norm, gz["month"].values, m_norm_gz)
    nn_model, losses = train_nn_beta(x_ext, beta_smooth, train_mask)
    beta_nn = nn_predict_beta(nn_model, x_ext)

    # Validate with discrete reconstruction
    pred_nn = reconstruct_cases(beta_nn, m_norm_gz, cases_obs)
    p1_met = evaluate_cases(cases_obs[train_mask], pred_nn[train_mask])
    print(f"  Phase 1: r={p1_met['pearson_r']:.3f}, rho={p1_met['spearman_rho']:.3f}, "
          f"R2_log={p1_met['r2_log']:.3f}, MAE={p1_met['mae']:.1f}", flush=True)

    gz["cases_pred_nn"] = pred_nn; gz["beta_nn"] = beta_nn
    gz[["city_en","year","month","cases","cases_pred_nn","beta_inverted","beta_smooth",
        "beta_nn","R0","tem","rhu","pre","index_norm_city"]].to_csv(
        OUT/"phase1_guangzhou_predictions_data2.csv", index=False)
    pd.DataFrame([p1_met]).to_csv(OUT/"phase1_metrics_data2.csv", index=False)

    # ── Phase 1c: Leave-One-Year-Out CV ──────────────────────────────
    print("Phase 1c: Leave-One-Year-Out CV ...", flush=True)
    cv_rows = []
    for test_year in sorted(gz["year"].unique()):
        cv_mask = gz["year"].values != test_year
        cv_model, _ = train_nn_beta(x_ext, beta_smooth, cv_mask, n_epochs=1000)
        cv_beta = nn_predict_beta(cv_model, x_ext)
        cv_pred = reconstruct_cases(cv_beta, m_norm_gz, cases_obs)
        met = evaluate_cases(cases_obs[~cv_mask], cv_pred[~cv_mask])
        met["test_year"] = test_year; cv_rows.append(met)
        print(f"  {test_year}: r={met['pearson_r']:.3f}, rho={met['spearman_rho']:.3f}", flush=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT/"phase1_cv_results.csv", index=False)
    print(f"  CV mean: r={cv_df['pearson_r'].mean():.3f}+/-{cv_df['pearson_r'].std():.3f}", flush=True)

    # ── Phase 2: Dual formula discovery ──────────────────────────────
    print("Phase 2: Fitting polynomial + physics formulas ...", flush=True)
    poly_model, phys_model, formula_comp = fit_both_formulas(weather_raw, beta_nn)
    beta_poly = poly_model.predict(weather_raw)
    beta_phys = phys_model.predict(weather_raw)

    pd.DataFrame([formula_comp["polynomial"]]).to_csv(OUT/"phase2_formula_fit_metrics_data2.csv", index=False)
    pd.DataFrame(formula_comp).T.to_csv(OUT/"phase2_physics_vs_poly.csv")
    pd.DataFrame([poly_model.get_coefficients()]).to_csv(OUT/"phase2_poly_coefficients.csv", index=False)
    pd.DataFrame([phys_model.get_params_dict()]).to_csv(OUT/"phase2_physics_params.csv", index=False)
    with open(OUT/"phase2_pysr_best_equation.txt","w") as f:
        f.write(f"Polynomial: {poly_model.equation_str()}\nPhysics: {phys_model.equation_str()}\n")
        f.write(f"\nComparison:\n{pd.DataFrame(formula_comp).T.to_string()}\n")
    for nm in formula_comp: print(f"  {nm}: R2={formula_comp[nm]['r2']:.4f}, AIC={formula_comp[nm]['aic']:.1f}", flush=True)

    # ── Phase 3: Transfer ────────────────────────────────────────────
    print("Phase 3: Transferring to 16 cities ...", flush=True)
    best_formula = poly_model if formula_comp["polynomial"]["r2"]>=formula_comp["physics"]["r2"] else phys_model
    city_tables = []
    for city in sorted(monthly["city_en"].unique()):
        cdf = monthly[monthly["city_en"]==city].sort_values(["year","month"]).copy()
        cdf = attach_mosquito_proxy(cdf, bi_proxy, city)
        city_tables.append(predict_city_monthly(cdf, best_formula))
    transfer_monthly = pd.concat(city_tables, ignore_index=True)
    transfer_monthly.to_csv(OUT/"transfer_monthly_all_cities_data2.csv", index=False)

    y14 = transfer_monthly[transfer_monthly["year"]==2014]
    annual = y14.groupby("city_en").agg(cases_2014=("cases","sum"),risk_2014=("risk_monthly","sum")).reset_index()
    annual.to_csv(OUT/"transfer_annual2014_data2.csv", index=False)
    obs14,pred14 = annual["cases_2014"].values, annual["risk_2014"].values
    rho_all,_ = spearmanr(obs14,pred14)
    non_gz = annual[annual["city_en"]!="Guangzhou"]
    rho_ng,_ = spearmanr(non_gz["cases_2014"].values, non_gz["risk_2014"].values)
    tr = [{"subset":"all","N":16,"MAE":float(np.mean(np.abs(obs14-pred14))),"RMSE":float(np.sqrt(np.mean((obs14-pred14)**2))),"Spearman_rho":rho_all},
          {"subset":"non_gz","N":15,"MAE":float(np.mean(np.abs(non_gz["cases_2014"].values-non_gz["risk_2014"].values))),"Spearman_rho":rho_ng}]
    pd.DataFrame(tr).to_csv(OUT/"transfer_metrics_data2.csv", index=False)

    city_metrics = []
    for city in sorted(transfer_monthly["city_en"].unique()):
        cdf = transfer_monthly[transfer_monthly["city_en"]==city]
        m = evaluate_cases(cdf["cases"].values, cdf["risk_monthly"].values); m["city_en"] = city; city_metrics.append(m)
    city_met_df = pd.DataFrame(city_metrics)
    city_met_df.to_csv(OUT/"transfer_city_monthly_metrics.csv", index=False)
    print("\nPer-city metrics:")
    print(city_met_df[["city_en","pearson_r","spearman_rho","r2_log","wape"]].to_string(index=False))

    # ── Baselines ────────────────────────────────────────────────────
    print("\nPhase 3b: Baselines ...", flush=True)
    bl_rows = []
    for city in sorted(monthly["city_en"].unique()):
        cdf = monthly[monthly["city_en"]==city].sort_values(["year","month"]).reset_index(drop=True)
        obs_c = cdf["cases"].values.astype(float)
        tdf = transfer_monthly[transfer_monthly["city_en"]==city]
        for nm,pred in [("SEIR+NN",tdf["risk_monthly"].values if len(tdf)==len(cdf) else np.zeros(len(cdf))),
                         ("HistMean",baseline_historical_mean(cdf)),("LinReg",baseline_linear_regression(cdf)),
                         ("SeasonNaive",baseline_seasonal_naive(cdf))]:
            m = evaluate_cases(obs_c,pred); m["model"]=nm; m["city_en"]=city; bl_rows.append(m)
    bl_df = pd.DataFrame(bl_rows); bl_df.to_csv(OUT/"transfer_baseline_comparison.csv", index=False)
    print("\nBaseline comparison (16-city mean):")
    print(bl_df.groupby("model")[["pearson_r","spearman_rho","r2_log","wape"]].mean().to_string())

    # Group analysis
    bi_cities = set(bi_proxy["city_en"].unique())
    city_met_df["has_bi"] = city_met_df["city_en"].isin(bi_cities)
    grp = city_met_df.groupby("has_bi")[["pearson_r","spearman_rho","r2_log","wape"]].mean()
    grp.index = ["No BI","Has BI"]; grp.to_csv(OUT/"transfer_group_analysis.csv")
    print("\nGroup analysis:"); print(grp.to_string())

    # ── Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots ...", flush=True)
    plot_phase1(gz, pred_nn, beta_nn, p1_met, losses)
    plot_cv_boxplot(cv_df)
    plot_r0_analysis(gz, R0)
    plot_phase2_dual(beta_nn, beta_poly, beta_phys, formula_comp)
    plot_response_surface(phys_model, poly_model, weather_raw)
    plot_transfer(annual)
    plot_outbreak_2014_beta(gz)
    plot_all_cities_grid(transfer_monthly)
    plot_baseline_comparison(bl_df)

    print(f"\n=== Done ===")
    print(f"Phase1: {OUT/'phase1_metrics_data2.csv'}")
    print(f"Phase1 CV: {OUT/'phase1_cv_results.csv'}")
    print(f"Phase2: {OUT/'phase2_pysr_best_equation.txt'}")
    print(f"Transfer: {OUT/'transfer_metrics_data2.csv'}")
    print(f"Baselines: {OUT/'transfer_baseline_comparison.csv'}")

if __name__ == "__main__":
    main()