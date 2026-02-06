#!/usr/bin/env python3
"""
Phase 1: 神经网络耦合动力学模型
=================================
参考: Zhang M, Wang X, Tang S (2024) PLoS Computational Biology
"Integrating dynamic models and neural networks to discover the
 mechanism of meteorological factors on Aedes population"

整体框架分两步:
  Step A: 蚊虫种群动力学 (NN耦合ODE → 拟合BI数据)
    dP/dt = NN(T,H,R)·A - dp(T)·P - mp(T)·P·(1+P/K)
    dA/dt = σ·dp(T)·P - ma(T)·A
    NN(T,H,R): 产卵率, 神经网络近似

  Step B: 在蚊虫动态基础上加入疾病传播 (SEIR)
    dEh/dt = β·b(T)·A_infected/Nh·Sh + imp - σh·Eh
    dIh/dt = σh·Eh - γ·Ih
    用病例数据校准 β 和 imp

关键: NN通过ODE求解器间接训练, 梯度穿过微分方程到达NN
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. 温度依赖函数 (来自文献, PyTorch可微分版本)
# ============================================================

def mp_T(T):
    """幼虫死亡率 (Otero et al. 2006), 高斯型"""
    return 0.05 + 0.003 * (T - 22.0)**2

def ma_T(T):
    """成蚊死亡率 (Brady et al. 2013), U型"""
    return 0.03 + 0.002 * (T - 26.0)**2

def dp_T(T):
    """幼虫→成蚊发育率 (Sharpe & DeMichele), 最适~27°C"""
    return torch.clamp(0.08 * torch.exp(-((T - 27.0) / 9.0)**2), min=0.005)

def b_transmit(T):
    """传播效率 b(T), Liu-Helmersson et al. 2014"""
    return 0.4 * torch.exp(-((T - 27.0) / 6.0)**2) * ((T > 14) & (T < 35)).float() + 0.001


# ============================================================
# 2. 产卵率神经网络
# ============================================================

class OvipositionNN(nn.Module):
    """
    产卵率NN: (T, H, R) → 日产卵率
    参照 Zhang et al.: 3层前馈, Sigmoid, 输出映射到生物学合理范围

    改进:
    - 使用Softplus作为内部激活 (梯度更好)
    - 输出用Softplus保证正值且可学习范围
    """
    def __init__(self, n_input=3, n_hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1),
        )
        # 初始化: 让输出初始值在合理范围
        with torch.no_grad():
            self.net[-1].bias.fill_(1.0)

    def forward(self, x):
        """
        x: (batch, 3) 归一化的 (T, H, R)
        returns: (batch, 1) 产卵率 > 0
        """
        raw = self.net(x)
        return torch.nn.functional.softplus(raw) + 0.01  # 保证 > 0.01


# ============================================================
# 3. Step A: 蚊虫种群动力学 (NN耦合ODE)
# ============================================================

class MosquitoDynamics(nn.Module):
    """
    蚊虫种群动力学模型 (类似Zhang et al. 2024)

    dP/dt = NN(T,H,R)·A - dp(T)·P - mp(T)·P·(1+P/K)
    dA/dt = σ·dp(T)·P - ma(T)·A

    P: 未成熟期 (卵+幼虫+蛹)
    A: 成蚊

    训练目标: A(t) 正比于 BI(t)
    """
    def __init__(self, nn_model):
        super().__init__()
        self.nn_model = nn_model

        # 可训练参数
        self.log_sigma = nn.Parameter(torch.tensor(-0.5))    # 羽化存活率 ~0.6
        self.log_K = nn.Parameter(torch.tensor(7.0))         # 承载力 (对数)
        self.log_P0_scale = nn.Parameter(torch.tensor(4.0))  # 初始P
        self.log_A0_scale = nn.Parameter(torch.tensor(3.0))  # 初始A

        # 线性映射: A → BI (可训练的线性变换)
        self.bi_scale = nn.Parameter(torch.tensor(0.0))      # log(A→BI比例)
        self.bi_offset = nn.Parameter(torch.tensor(1.0))     # BI偏移

    @property
    def sigma(self):
        return torch.sigmoid(self.log_sigma)  # ∈ (0, 1)

    @property
    def K(self):
        return torch.exp(self.log_K)

    def forward(self, weather_norm, weather_raw, dt=1.0, days_per_step=15):
        """
        运行蚊虫ODE, 每半月一步 (匹配BI数据时间尺度)

        Args:
            weather_norm: (n_months, 3) 归一化气象
            weather_raw: (n_months, 3) 原始气象
            dt: 积分步长 (天)
            days_per_step: 每半月天数

        Returns:
            A_series: (n_months,) 月末成蚊数
            P_series: (n_months,) 月末幼虫数
            ovi_series: (n_months,) NN产卵率
        """
        n = weather_norm.shape[0]

        # 初始状态
        P = torch.exp(self.log_P0_scale) * 1000.0
        A = torch.exp(self.log_A0_scale) * 100.0

        A_series = []
        P_series = []
        ovi_series = []

        K = self.K * 1e6  # 承载力
        sigma = self.sigma

        for m in range(n):
            # NN产卵率
            nn_input = weather_norm[m:m+1]
            ovi = self.nn_model(nn_input).squeeze()
            ovi_series.append(ovi)

            T_raw = weather_raw[m, 0]

            # 温度依赖率
            mp = mp_T(T_raw)
            ma = ma_T(T_raw)
            dp = dp_T(T_raw)

            # 半月积分 (多步Euler, 提高稳定性)
            for _ in range(days_per_step):
                # 密度依赖
                density = 1.0 + P / K

                # 幼虫: 产卵 - 发育 - 死亡
                dP = ovi * A - dp * P - mp * P * density

                # 成蚊: 羽化 - 死亡
                emergence = sigma * dp * P
                dA = emergence - ma * A

                # 自适应步长限制 (防止爆炸)
                max_dP = 0.3 * P + 100.0
                max_dA = 0.3 * A + 100.0
                dP = torch.clamp(dP * dt, min=-max_dP, max=max_dP)
                dA = torch.clamp(dA * dt, min=-max_dA, max=max_dA)

                P = torch.clamp(P + dP, min=1.0)
                A = torch.clamp(A + dA, min=1.0)

            A_series.append(A)
            P_series.append(P)

        return torch.stack(A_series), torch.stack(P_series), torch.stack(ovi_series)

    def predict_bi(self, A_series):
        """成蚊数 → BI预测"""
        log_A = torch.log(A_series + 1.0)
        return torch.exp(self.bi_scale) * log_A + self.bi_offset


# ============================================================
# 4. Step B: 疾病传播动力学 (在蚊虫动态基础上)
# ============================================================

class DiseaseDynamics(nn.Module):
    """
    基于蚊虫动态的SEIR疾病模型

    给定 A(t) (成蚊种群, 来自Step A)
    计算月度新增病例

    改进: 用蚊虫种群的相对变化驱动传播力,
    避免绝对量级问题
    """
    def __init__(self, N_h=14_000_000):
        super().__init__()
        self.N_h = N_h
        self.log_beta = nn.Parameter(torch.tensor(0.0))     # 基础传播率
        self.log_import = nn.Parameter(torch.tensor(2.0))    # 输入病例
        self.log_amp = nn.Parameter(torch.tensor(3.0))       # 蚊虫→病例放大系数
        self.sigma_h = 1.0 / 5.5
        self.gamma = 1.0 / 7.0

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    @property
    def import_rate(self):
        return torch.exp(self.log_import)

    @property
    def amp(self):
        return torch.exp(self.log_amp)

    def forward(self, A_series, weather_raw, dt=1.0, days_per_month=30):
        """
        运行SEIR模型
        """
        n = len(A_series)
        beta = torch.clamp(self.beta, min=1e-6, max=50.0)
        imp = self.import_rate
        amp = self.amp

        Eh = torch.tensor(0.0)
        Ih = torch.tensor(1.0)
        Rh = torch.tensor(0.0)

        # 归一化蚊虫种群 (用均值归一化, 避免量级问题)
        A_mean = A_series.mean().detach()
        A_norm = A_series / (A_mean + 1.0)

        cases_monthly = []

        for m in range(n):
            T = weather_raw[m, 0]
            b_T = b_transmit(T)
            A_rel = A_norm[m]  # 相对蚊虫密度

            month_cases = torch.tensor(0.0)
            imp_daily = imp / 30.0

            for _ in range(days_per_month):
                Sh = torch.clamp(self.N_h - Eh - Ih - Rh, min=0.0)

                # 传染力: β * b(T) * A_rel (相对蚊虫密度驱动)
                force = beta * b_T * A_rel / self.N_h
                force = torch.clamp(force, max=0.01)

                dEh = force * Sh * amp + imp_daily - self.sigma_h * Eh
                dIh = self.sigma_h * torch.clamp(Eh, min=0) - self.gamma * Ih
                dRh = self.gamma * torch.clamp(Ih, min=0)

                new_cases = self.sigma_h * torch.clamp(Eh, min=0) * dt
                month_cases = month_cases + new_cases

                # Euler更新
                delta_Eh = torch.clamp(dEh * dt, min=-0.5 * Eh,
                                       max=torch.tensor(self.N_h * 0.001))
                Eh = torch.clamp(Eh + delta_Eh, min=0.0)

                delta_Ih = torch.clamp(dIh * dt, min=-0.5 * Ih,
                                       max=torch.tensor(self.N_h * 0.001))
                Ih = torch.clamp(Ih + delta_Ih, min=0.0)
                Rh = torch.clamp(Rh + dRh * dt, min=0.0)

            cases_monthly.append(month_cases)

        return torch.stack(cases_monthly)


# ============================================================
# 5. 数据加载
# ============================================================

def load_data(exclude_2014=False):
    """加载月度数据"""
    log("\n[1] 加载数据...")

    case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
    if exclude_2014:
        # 排除2014年暴发数据 (该年占总病例90%，严重干扰模型训练)
        df = case_df[((case_df['year'] >= 2006) & (case_df['year'] <= 2019)) &
                      (case_df['year'] != 2014)].copy().reset_index(drop=True)
        log("  *** 排除2014年暴发数据 ***")
    else:
        df = case_df[(case_df['year'] >= 2006) & (case_df['year'] <= 2014)].copy().reset_index(drop=True)

    bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']

    df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
    df['has_bi'] = df['bi'].notna()
    df['bi'] = df['bi'].fillna(0)

    weather_raw = df[['temperature', 'humidity', 'precipitation']].values.astype(np.float32)
    w_min = weather_raw.min(axis=0)
    w_max = weather_raw.max(axis=0)
    w_range = np.maximum(w_max - w_min, 1e-8)
    weather_norm = (weather_raw - w_min) / w_range

    log(f"  数据: {df['year'].min()}-{df['year'].max()}, {len(df)}个月")
    log(f"  BI数据: {df['has_bi'].sum()}/{len(df)}个月")
    log(f"  总病例: {df['cases'].sum():,}")
    log(f"  温度: [{w_min[0]:.1f}, {w_max[0]:.1f}]°C")

    return df, weather_raw, weather_norm, w_min, w_max


# ============================================================
# 6. Step A 训练: 蚊虫种群拟合BI
# ============================================================

def train_step_a(mosquito_model, weather_norm_t, weather_raw_t,
                 obs_bi_t, has_bi_mask, n_epochs=800, lr=0.005):
    """
    Step A: 训练NN耦合蚊虫ODE → 拟合BI数据
    """
    log("\n" + "=" * 60)
    log("Step A: 训练蚊虫种群模型 (NN耦合ODE → BI)")
    log("=" * 60)

    optimizer = torch.optim.Adam(mosquito_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2)

    best_loss = float('inf')
    best_state = None
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        A_series, P_series, ovi_series = mosquito_model(weather_norm_t, weather_raw_t)
        pred_bi = mosquito_model.predict_bi(A_series)

        # === 损失函数 ===
        # 1. BI拟合 (有BI数据的月份)
        mask = has_bi_mask
        if mask.sum() > 5:
            pred_m = pred_bi[mask]
            obs_m = obs_bi_t[mask]
            # MSE
            loss_bi = torch.mean((pred_m - obs_m)**2)
            # 相关性 (越高越好)
            pred_n = (pred_m - pred_m.mean()) / (pred_m.std() + 1e-6)
            obs_n = (obs_m - obs_m.mean()) / (obs_m.std() + 1e-6)
            loss_corr = -torch.mean(pred_n * obs_n)
        else:
            loss_bi = torch.tensor(0.0)
            loss_corr = torch.tensor(0.0)

        # 2. NN平滑性 (ovi_series is already a stacked tensor from forward())
        loss_smooth = torch.mean((ovi_series[1:] - ovi_series[:-1])**2)

        # 3. NN应该有季节性变化 (不应该恒定)
        loss_variance = -torch.log(ovi_series.std() + 1e-6)

        # 总损失
        loss = 1.0 * loss_bi + 2.0 * loss_corr + 0.05 * loss_smooth + 0.5 * loss_variance

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(mosquito_model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in mosquito_model.state_dict().items()}

        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                A_s, _, ovi_s = mosquito_model(weather_norm_t, weather_raw_t)
                pb = mosquito_model.predict_bi(A_s).numpy()
                ob = obs_bi_t.numpy()
                m = has_bi_mask.numpy().astype(bool)
                if m.sum() > 5 and np.std(pb[m]) > 0:
                    corr = np.corrcoef(ob[m], pb[m])[0, 1]
                    r2 = r2_score(ob[m], pb[m])
                else:
                    corr, r2 = 0, -999
                ovi_arr = ovi_s.numpy()

            log(f"  Epoch {epoch+1:4d}: Loss={loss.item():.4f}, "
                f"BI_corr={corr:.4f}, BI_R²={r2:.4f}, "
                f"ovi=[{ovi_arr.min():.3f},{ovi_arr.max():.3f}]")

    mosquito_model.load_state_dict(best_state)
    log(f"\n  Step A完成! 最优Loss={best_loss:.4f}")
    return losses


# ============================================================
# 7. Step B 训练: 疾病模型拟合病例
# ============================================================

def train_step_b(disease_model, mosquito_model, weather_norm_t, weather_raw_t,
                 obs_cases_t, n_epochs=500, lr=0.005):
    """
    Step B: 固定蚊虫模型, 训练疾病参数 → 拟合病例
    """
    log("\n" + "=" * 60)
    log("Step B: 训练疾病传播模型 (SEIR → 病例)")
    log("=" * 60)

    # 固定蚊虫模型
    mosquito_model.eval()
    with torch.no_grad():
        A_series, _, _ = mosquito_model(weather_norm_t, weather_raw_t)

    optimizer = torch.optim.Adam(disease_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    best_loss = float('inf')
    best_state = None
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_cases = disease_model(A_series.detach(), weather_raw_t)

        # 损失: log空间MSE + 相关性
        pred_c = torch.clamp(pred_cases, min=0)
        loss_mse = torch.mean((torch.log1p(pred_c) - torch.log1p(obs_cases_t))**2)

        if pred_c.std() > 1e-6:
            pn = (pred_c - pred_c.mean()) / (pred_c.std() + 1e-6)
            on = (obs_cases_t - obs_cases_t.mean()) / (obs_cases_t.std() + 1e-6)
            loss_corr = -torch.mean(pn * on)
        else:
            loss_corr = torch.tensor(0.0)

        # 鼓励有变化
        loss_var = -torch.log(pred_c.std() + 1e-6)

        loss = 1.0 * loss_mse + 1.0 * loss_corr + 0.1 * loss_var

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(disease_model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in disease_model.state_dict().items()}

        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                pc = disease_model(A_series.detach(), weather_raw_t)
                pc_np = np.maximum(pc.numpy(), 0)
                oc_np = obs_cases_t.numpy()
                pc_clean = np.nan_to_num(pc_np, 0)
                if np.std(pc_clean) > 0:
                    corr = np.corrcoef(oc_np, pc_clean)[0, 1]
                    r2l = r2_score(np.log1p(oc_np), np.log1p(pc_clean))
                else:
                    corr, r2l = 0, -999

            log(f"  Epoch {epoch+1:4d}: Loss={loss.item():.4f}, "
                f"r={corr:.4f}, R²(log)={r2l:.4f}, "
                f"β={disease_model.beta.item():.4f}, imp={disease_model.import_rate.item():.2f}, "
                f"amp={disease_model.amp.item():.2f}")

    disease_model.load_state_dict(best_state)
    log(f"\n  Step B完成! 最优Loss={best_loss:.4f}")
    return losses


# ============================================================
# 8. Step C: 联合微调 (Joint Fine-tuning)
# ============================================================

def train_joint(mosquito_model, disease_model, weather_norm_t, weather_raw_t,
                obs_bi_t, obs_cases_t, has_bi_mask, n_epochs=400, lr=0.001):
    """
    联合微调: 同时优化蚊虫模型和疾病模型
    """
    log("\n" + "=" * 60)
    log("Step C: 联合微调 (蚊虫+疾病 同时优化)")
    log("=" * 60)

    all_params = list(mosquito_model.parameters()) + list(disease_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float('inf')
    best_state_m = None
    best_state_d = None
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # 蚊虫前向
        A_series, P_series, ovi_series = mosquito_model(weather_norm_t, weather_raw_t)
        pred_bi = mosquito_model.predict_bi(A_series)

        # 疾病前向
        pred_cases = disease_model(A_series, weather_raw_t)

        # === 多目标损失 ===
        mask = has_bi_mask

        # BI拟合
        if mask.sum() > 5:
            loss_bi = torch.mean((pred_bi[mask] - obs_bi_t[mask])**2)
            pn = (pred_bi[mask] - pred_bi[mask].mean()) / (pred_bi[mask].std() + 1e-6)
            on = (obs_bi_t[mask] - obs_bi_t[mask].mean()) / (obs_bi_t[mask].std() + 1e-6)
            loss_bi_corr = -torch.mean(pn * on)
        else:
            loss_bi = torch.tensor(0.0)
            loss_bi_corr = torch.tensor(0.0)

        # 病例拟合
        pred_c = torch.clamp(pred_cases, min=0)
        loss_cases = torch.mean((torch.log1p(pred_c) - torch.log1p(obs_cases_t))**2)
        if pred_c.std() > 1e-6:
            pn = (pred_c - pred_c.mean()) / (pred_c.std() + 1e-6)
            on = (obs_cases_t - obs_cases_t.mean()) / (obs_cases_t.std() + 1e-6)
            loss_cases_corr = -torch.mean(pn * on)
        else:
            loss_cases_corr = torch.tensor(0.0)

        # NN平滑+变异 (ovi_series is already a stacked tensor)
        loss_smooth = torch.mean((ovi_series[1:] - ovi_series[:-1])**2)
        loss_var = -torch.log(ovi_series.std() + 1e-6)

        loss = (0.5 * loss_bi + 1.0 * loss_bi_corr +
                1.0 * loss_cases + 1.0 * loss_cases_corr +
                0.02 * loss_smooth + 0.2 * loss_var)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state_m = {k: v.cpu().clone() for k, v in mosquito_model.state_dict().items()}
            best_state_d = {k: v.cpu().clone() for k, v in disease_model.state_dict().items()}

        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                pc = np.maximum(np.nan_to_num(pred_cases.numpy(), 0), 0)
                oc = obs_cases_t.numpy()
                pb = pred_bi.numpy()
                ob = obs_bi_t.numpy()
                m = has_bi_mask.numpy().astype(bool)
                corr_c = np.corrcoef(oc, pc)[0, 1] if np.std(pc) > 0 else 0
                corr_bi = np.corrcoef(ob[m], pb[m])[0, 1] if np.std(pb[m]) > 0 else 0
                ovi_arr = ovi_series.numpy()
            log(f"  Epoch {epoch+1:4d}: Loss={loss.item():.4f}, "
                f"BI_r={corr_bi:.4f}, Case_r={corr_c:.4f}, "
                f"ovi=[{ovi_arr.min():.2f},{ovi_arr.max():.2f}]")

    mosquito_model.load_state_dict(best_state_m)
    disease_model.load_state_dict(best_state_d)
    log(f"\n  联合微调完成! 最优Loss={best_loss:.4f}")
    return losses


# ============================================================
# 9. 评估与可视化
# ============================================================

def evaluate_and_visualize(mosquito_model, disease_model, weather_norm_t, weather_raw_t,
                            df, obs_bi_t, obs_cases_t, has_bi_mask,
                            weather_raw, w_min, w_max, all_losses):
    """全面评估"""
    log("\n[评估与可视化]")

    mosquito_model.eval()
    disease_model.eval()

    with torch.no_grad():
        A_series, P_series, ovi_series = mosquito_model(weather_norm_t, weather_raw_t)
        pred_bi = mosquito_model.predict_bi(A_series)
        pred_cases = disease_model(A_series, weather_raw_t)

    pred_bi_np = pred_bi.numpy()
    pred_cases_np = np.maximum(pred_cases.numpy(), 0)
    A_np = A_series.numpy()
    P_np = P_series.numpy()
    ovi_np = ovi_series.numpy()
    obs_bi_np = obs_bi_t.numpy()
    obs_cases_np = obs_cases_t.numpy()
    has_bi = has_bi_mask.numpy().astype(bool)

    # 指标
    if has_bi.sum() > 5 and np.std(pred_bi_np[has_bi]) > 0:
        corr_bi, _ = pearsonr(obs_bi_np[has_bi], pred_bi_np[has_bi])
        r2_bi = r2_score(obs_bi_np[has_bi], pred_bi_np[has_bi])
    else:
        corr_bi, r2_bi = 0, -999

    if np.std(pred_cases_np) > 0:
        corr_cases, pval = pearsonr(obs_cases_np, pred_cases_np)
        r2_cases_log = r2_score(np.log1p(obs_cases_np), np.log1p(pred_cases_np))
    else:
        corr_cases, pval, r2_cases_log = 0, 1, -999

    log(f"\n  === 最终性能 ===")
    log(f"  【BI拟合】r={corr_bi:.4f}, R²={r2_bi:.4f}")
    log(f"  【病例拟合】r={corr_cases:.4f} (p={pval:.2e}), R²(log)={r2_cases_log:.4f}")
    log(f"  【NN产卵率】范围=[{ovi_np.min():.3f}, {ovi_np.max():.3f}]")
    log(f"  【参数】β={disease_model.beta.item():.4f}, imp={disease_model.import_rate.item():.2f}/月")

    # =========== 可视化 ===========
    fig = plt.figure(figsize=(22, 24))
    months = np.arange(len(obs_cases_np))
    years = df['year'].values

    # --- 1. BI拟合 ---
    ax = fig.add_subplot(4, 3, 1)
    bi_months = np.where(has_bi)[0]
    ax.plot(bi_months, obs_bi_np[has_bi], 'go-', lw=2, ms=5, label='BI (observed)')
    ax.plot(months, pred_bi_np, 'r-', lw=2, alpha=0.8, label='Model (predicted)')
    ax.fill_between(bi_months, 0, obs_bi_np[has_bi], alpha=0.15, color='green')
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Breteau Index')
    ax.set_title(f'Step A: BI Fitting (r={corr_bi:.3f}, R²={r2_bi:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 2. 病例拟合 ---
    ax = fig.add_subplot(4, 3, 2)
    ax.plot(months, obs_cases_np, 'b-', lw=2, label='Observed', marker='o', ms=3, alpha=0.8)
    ax.plot(months, pred_cases_np, 'r-', lw=2, label='Model', marker='s', ms=3, alpha=0.8)
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Cases')
    ax.set_title(f'Step B+C: Cases (r={corr_cases:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. 病例 (log) ---
    ax = fig.add_subplot(4, 3, 3)
    ax.semilogy(months, obs_cases_np + 1, 'b-', lw=2, label='Observed', marker='o', ms=3)
    ax.semilogy(months, pred_cases_np + 1, 'r-', lw=2, label='Model', marker='s', ms=3)
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Cases (log)')
    ax.set_title(f'Log Scale (R²={r2_cases_log:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. NN产卵率 ---
    ax = fig.add_subplot(4, 3, 4)
    ax.plot(months, ovi_np, 'purple', lw=2.5, label='NN(T,H,R)')
    ax2 = ax.twinx()
    ax2.plot(months, weather_raw[:, 0], 'orange', lw=1.5, alpha=0.6, label='Temp')
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Oviposition Rate', color='purple')
    ax2.set_ylabel('Temperature (°C)', color='orange')
    ax.set_title('Neural Network: Learned Oviposition Rate')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- 5. NN vs 温度散点 ---
    ax = fig.add_subplot(4, 3, 5)
    sc = ax.scatter(weather_raw[:, 0], ovi_np, c=weather_raw[:, 2],
                    cmap='Blues', s=80, edgecolors='black', lw=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, label='Precipitation (mm)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Oviposition Rate')
    ax.set_title('NN: Oviposition ~ Temperature')
    ax.grid(True, alpha=0.3)

    # --- 6. 成蚊种群 ---
    ax = fig.add_subplot(4, 3, 6)
    ax.plot(months, A_np, 'g-', lw=2, label='Adults (A)')
    ax.plot(months, P_np, 'b-', lw=1.5, alpha=0.6, label='Immatures (P)')
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Population')
    ax.set_title('Simulated Mosquito Population')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 7. NN热力图 ---
    ax = fig.add_subplot(4, 3, 7)
    T_grid = np.linspace(0, 1, 50)
    R_grid = np.linspace(0, 1, 50)
    TT, RR = np.meshgrid(T_grid, R_grid)
    H_mid = 0.5
    grid_inp = np.column_stack([TT.ravel(), np.full(TT.size, H_mid), RR.ravel()]).astype(np.float32)
    with torch.no_grad():
        grid_ovi = mosquito_model.nn_model(torch.from_numpy(grid_inp)).numpy().reshape(50, 50)
    T_real = w_min[0] + TT * (w_max[0] - w_min[0])
    R_real = w_min[2] + RR * (w_max[2] - w_min[2])
    im = ax.contourf(T_real, R_real, grid_ovi, levels=20, cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='Oviposition Rate')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Precipitation (mm)')
    ax.set_title('NN Learned: Oviposition(T, R)')

    # --- 8. 年度对比 ---
    ax = fig.add_subplot(4, 3, 8)
    df_eval = pd.DataFrame({'year': years, 'actual': obs_cases_np, 'predicted': pred_cases_np})
    yearly = df_eval.groupby('year').agg({'actual': 'sum', 'predicted': 'sum'}).reset_index()
    x = range(len(yearly))
    width = 0.35
    ax.bar([i - width/2 for i in x], yearly['actual'], width, label='Actual', color='steelblue')
    ax.bar([i + width/2 for i in x], yearly['predicted'], width, label='Model', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(yearly['year'])
    ax.set_ylabel('Annual Cases')
    ax.set_title('Annual Cases Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 9. 散点图 ---
    ax = fig.add_subplot(4, 3, 9)
    ax.scatter(obs_cases_np, pred_cases_np, alpha=0.6, c='steelblue', s=50, edgecolors='white')
    max_val = max(obs_cases_np.max(), pred_cases_np.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2)
    ax.set_xlabel('Observed Cases')
    ax.set_ylabel('Predicted Cases')
    ax.set_title(f'Scatter (r={corr_cases:.3f})')
    ax.grid(True, alpha=0.3)

    # --- 10. 训练损失 ---
    ax = fig.add_subplot(4, 3, 10)
    loss_a, loss_b, loss_c = all_losses
    offset = 0
    for label, ls, color in [('A: Mosquito', loss_a, 'green'),
                               ('B: Disease', loss_b, 'blue'),
                               ('C: Joint', loss_c, 'red')]:
        x_ax = range(offset, offset + len(ls))
        ax.plot(x_ax, ls, color=color, lw=0.5, alpha=0.5)
        from scipy.ndimage import gaussian_filter1d
        if len(ls) > 10:
            ax.plot(x_ax, gaussian_filter1d(ls, sigma=5), color=color, lw=2, label=label)
        offset += len(ls)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (3 Steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # --- 11. BI散点 ---
    ax = fig.add_subplot(4, 3, 11)
    if has_bi.sum() > 5:
        ax.scatter(obs_bi_np[has_bi], pred_bi_np[has_bi], c='green', s=60, alpha=0.7, edgecolors='black')
        max_bi = max(obs_bi_np[has_bi].max(), pred_bi_np[has_bi].max()) * 1.1
        ax.plot([0, max_bi], [0, max_bi], 'r--', lw=2)
        ax.set_xlabel('Observed BI')
        ax.set_ylabel('Predicted BI')
        ax.set_title(f'BI Scatter (R²={r2_bi:.3f})')
    ax.grid(True, alpha=0.3)

    # --- 12. 框架总结 ---
    ax = fig.add_subplot(4, 3, 12)
    ax.axis('off')
    summary = f"""
    Coupled NN-ODE Dynamics Model
    (Zhang et al. 2024 framework)
    =====================================

    Step A: Mosquito Population (NN+ODE)
      dP/dt = NN(T,H,R)·A - dp(T)·P - mp(T)·P
      dA/dt = σ·dp(T)·P - ma(T)·A
      BI corr: {corr_bi:.4f}, R²: {r2_bi:.4f}

    Step B+C: Disease Dynamics (SEIR)
      dEh/dt = β·b(T)·A/(Nh+A)·Sh + imp - σh·Eh
      dIh/dt = σh·Eh - γ·Ih
      Case corr: {corr_cases:.4f}
      R² (log):  {r2_cases_log:.4f}

    NN Architecture: 3→16→16→1 (Softplus)
    NN Oviposition: [{ovi_np.min():.3f}, {ovi_np.max():.3f}]

    Learned Parameters:
      β = {disease_model.beta.item():.4f}
      import = {disease_model.import_rate.item():.2f}/month
      σ (emergence) = {mosquito_model.sigma.item():.4f}

    Next: Phase 2 Symbolic Regression
      NN(T,H,R) → analytical formula
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Phase 1: Neural Network Coupled Dynamics Model\n'
                 '(Mosquito Population + Disease Transmission)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/figures/phase1_coupled_model.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log("  已保存: results/figures/phase1_coupled_model.png")

    return {
        'corr_bi': corr_bi, 'r2_bi': r2_bi,
        'corr_cases': corr_cases, 'r2_cases_log': r2_cases_log,
        'pred_cases': pred_cases_np, 'pred_bi': pred_bi_np,
        'ovi_rates': ovi_np, 'A_series': A_np, 'P_series': P_np
    }


# ============================================================
# 10. 保存NN数据供Phase 2
# ============================================================

def save_for_phase2(mosquito_model, disease_model, weather_raw, w_min, w_max, df, results):
    """保存数据供Phase 2符号回归"""
    log("\n[保存Phase 2数据]")

    mosquito_model.eval()

    # 1. 网格数据
    T_range = np.linspace(w_min[0], w_max[0], 40)
    H_range = np.linspace(w_min[1], w_max[1], 10)
    R_range = np.linspace(w_min[2], w_max[2], 40)
    w_range = np.maximum(w_max - w_min, 1e-8)

    records = []
    for T in T_range:
        for H in H_range:
            for R in R_range:
                T_n = (T - w_min[0]) / w_range[0]
                H_n = (H - w_min[1]) / w_range[1]
                R_n = (R - w_min[2]) / w_range[2]
                inp = torch.tensor([[T_n, H_n, R_n]], dtype=torch.float32)
                with torch.no_grad():
                    ovi = mosquito_model.nn_model(inp).item()
                records.append({'temperature': T, 'humidity': H, 'precipitation': R,
                               'T_norm': T_n, 'H_norm': H_n, 'R_norm': R_n,
                               'oviposition_rate': ovi})

    pd.DataFrame(records).to_csv('/root/wenmei/results/data/nn_grid_output.csv', index=False)
    log(f"  网格: {len(records)} → nn_grid_output.csv")

    # 2. 观测点数据
    obs_records = []
    for i in range(len(weather_raw)):
        T, H, R = weather_raw[i]
        inp = torch.tensor([[(T-w_min[0])/w_range[0], (H-w_min[1])/w_range[1],
                              (R-w_min[2])/w_range[2]]], dtype=torch.float32)
        with torch.no_grad():
            ovi = mosquito_model.nn_model(inp).item()
        obs_records.append({'temperature': T, 'humidity': H, 'precipitation': R,
                           'oviposition_rate': ovi})
    pd.DataFrame(obs_records).to_csv('/root/wenmei/results/data/nn_obs_output.csv', index=False)
    log(f"  观测: {len(obs_records)} → nn_obs_output.csv")

    # 3. 模型权重
    torch.save({
        'mosquito': mosquito_model.state_dict(),
        'disease': disease_model.state_dict(),
    }, '/root/wenmei/results/data/phase1_model.pt')

    # 4. 归一化参数
    np.savez('/root/wenmei/results/data/phase1_norm_params.npz', w_min=w_min, w_max=w_max)

    # 5. 预测结果
    pd.DataFrame({
        'year': df['year'].values, 'month': df['month'].values,
        'temperature': weather_raw[:, 0], 'humidity': weather_raw[:, 1],
        'precipitation': weather_raw[:, 2],
        'bi_obs': df['bi'].values, 'cases_obs': df['cases'].values,
        'cases_pred': results['pred_cases'],
        'bi_pred': results['pred_bi'],
        'mosquito_A': results['A_series'],
        'oviposition_nn': results['ovi_rates']
    }).to_csv('/root/wenmei/results/data/phase1_predictions.csv', index=False)
    log("  预测 → phase1_predictions.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 70)
    log("Phase 1: 神经网络耦合动力学模型")
    log("参考: Zhang, Wang & Tang (2024) PLoS Computational Biology")
    log("=" * 70)
    log("""
    ┌──────────────────────────────────────────────────────────┐
    │  框架: 动力学模型 (ODE) 是主体                            │
    │                                                          │
    │  Step A: 蚊虫ODE + NN(产卵率) → 拟合BI                   │
    │    dP/dt = NN(T,H,R)·A - dp(T)·P - mp(T)·P              │
    │    dA/dt = σ·dp(T)·P - ma(T)·A                          │
    │                                                          │
    │  Step B: 固定蚊虫 → SEIR拟合病例                          │
    │    dEh/dt = β·b(T)·A·Sh/(Nh+A) + imp - σh·Eh           │
    │    dIh/dt = σh·Eh - γ·Ih                                │
    │                                                          │
    │  Step C: 联合微调 (蚊虫+疾病同时优化)                     │
    └──────────────────────────────────────────────────────────┘
    """)

    # 1. 数据 (排除2014年暴发)
    df, weather_raw, weather_norm, w_min, w_max = load_data(exclude_2014=True)
    weather_norm_t = torch.from_numpy(weather_norm.astype(np.float32))
    weather_raw_t = torch.from_numpy(weather_raw.astype(np.float32))
    obs_cases_t = torch.from_numpy(df['cases'].values.astype(np.float32))
    obs_bi_t = torch.from_numpy(df['bi'].values.astype(np.float32))
    has_bi_mask = torch.from_numpy(df['has_bi'].values)

    # 2. 构建模型
    log("\n[2] 构建模型...")
    nn_model = OvipositionNN(n_input=3, n_hidden=16)
    mosquito_model = MosquitoDynamics(nn_model)
    disease_model = DiseaseDynamics(N_h=14_000_000)

    nn_params = sum(p.numel() for p in nn_model.parameters())
    m_params = sum(p.numel() for p in mosquito_model.parameters())
    d_params = sum(p.numel() for p in disease_model.parameters())
    log(f"  NN参数: {nn_params}")
    log(f"  蚊虫模型参数: {m_params}")
    log(f"  疾病模型参数: {d_params}")
    log(f"  总计: {m_params + d_params}")

    # 3. Step A: 蚊虫拟合
    loss_a = train_step_a(mosquito_model, weather_norm_t, weather_raw_t,
                          obs_bi_t, has_bi_mask, n_epochs=800, lr=0.005)

    # 4. Step B: 疾病拟合
    loss_b = train_step_b(disease_model, mosquito_model, weather_norm_t, weather_raw_t,
                          obs_cases_t, n_epochs=500, lr=0.01)

    # 5. Step C: 联合微调
    loss_c = train_joint(mosquito_model, disease_model, weather_norm_t, weather_raw_t,
                         obs_bi_t, obs_cases_t, has_bi_mask, n_epochs=300, lr=0.002)

    # 6. 评估
    results = evaluate_and_visualize(
        mosquito_model, disease_model, weather_norm_t, weather_raw_t,
        df, obs_bi_t, obs_cases_t, has_bi_mask,
        weather_raw, w_min, w_max, (loss_a, loss_b, loss_c))

    # 7. 保存
    save_for_phase2(mosquito_model, disease_model, weather_raw, w_min, w_max, df, results)

    # 8. 总结
    log("\n" + "=" * 70)
    log("Phase 1 总结")
    log("=" * 70)
    log(f"""
    【模型框架】(参考 Zhang et al. 2024)
      动力学 (ODE) 是主体, NN嵌入其中:
        蚊虫: dP/dt = NN(T,H,R)·A - dp(T)·P - mp(T)·P
              dA/dt = σ·dp(T)·P - ma(T)·A
        疾病: SEIR传播

    【训练策略】
      Step A: 蚊虫ODE+NN → 拟合BI (NN间接训练)
      Step B: 固定蚊虫 → SEIR拟合病例
      Step C: 联合微调

    【性能】
      BI拟合:   r={results['corr_bi']:.4f}, R²={results['r2_bi']:.4f}
      病例拟合: r={results['corr_cases']:.4f}, R²(log)={results['r2_cases_log']:.4f}

    【NN产卵率】
      范围: [{results['ovi_rates'].min():.3f}, {results['ovi_rates'].max():.3f}]

    【下一步: Phase 2】
      符号回归: NN(T,H,R) → 解析公式 f(T,H,R)
      数据已保存: nn_grid_output.csv, nn_obs_output.csv
    """)
    log("Phase 1 完成!\n")
    return mosquito_model, disease_model, results


if __name__ == "__main__":
    m_model, d_model, results = main()
