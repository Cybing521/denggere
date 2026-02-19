#!/usr/bin/env python3
"""
完整蚊虫生命周期 + 人群 SEI-SEIR 模型

蚊虫水生阶段: E (卵) → L (幼虫) → P (蛹)
成蚊阶段: S_m → E_m → I_m (SEI)
人群: S_h → E_h → I_h → R_h (SEIR)

校正: 用 BI 数据校正幼虫密度 L (相对指标)
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 70)
log("完整蚊虫-人群 SEI-SEIR 动力学模型")
log("=" * 70)

# ============================================================
# 1. 温度依赖参数 (来自文献)
# ============================================================
class TempParams:
    """温度依赖的生物学参数"""
    
    @staticmethod
    def phi(T):
        """产卵率 φ(T)"""
        if T < 14 or T > 35:
            return 0.1
        return max(0.1, -0.02*T**2 + 1.1*T - 8.5)
    
    @staticmethod
    def f_e(T):
        """卵孵化率"""
        if T < 10 or T > 40:
            return 0.01
        return max(0.01, 0.0022*T*(T-10)*np.sqrt(max(0.1, 40-T)))
    
    @staticmethod
    def f_l(T):
        """幼虫发育率"""
        if T < 10 or T > 40:
            return 0.01
        return max(0.01, 0.00085*T*(T-10)*np.sqrt(max(0.1, 40-T)))
    
    @staticmethod
    def f_p(T):
        """蛹发育率"""
        if T < 10 or T > 40:
            return 0.01
        return max(0.01, 0.0021*T*(T-10)*np.sqrt(max(0.1, 40-T)))
    
    @staticmethod
    def mu_e(T):
        """卵死亡率"""
        return 0.1
    
    @staticmethod
    def mu_l(T):
        """幼虫死亡率"""
        if T < 10 or T > 40:
            return 0.5
        return max(0.01, 0.0025*T**2 - 0.094*T + 0.96)
    
    @staticmethod
    def mu_p(T):
        """蛹死亡率"""
        if T < 10 or T > 40:
            return 0.5
        return max(0.01, 0.0003*T**2 - 0.0126*T + 0.14)
    
    @staticmethod
    def mu_m(T):
        """成蚊死亡率"""
        if T < 10 or T > 40:
            return 0.5
        return max(0.02, 0.0006*T**2 - 0.028*T + 0.37)
    
    @staticmethod
    def epsilon(T):
        """蚊虫EIP转化率 (1/外潜伏期)"""
        if T < 18 or T > 34:
            return 0.05
        # EIP约7-14天，取10天
        return 0.1
    
    @staticmethod
    def b_hv(T):
        """人→蚊传播率 = a × c"""
        if T < 14 or T > 35:
            return 0.01
        a = max(0.01, 0.0005*T*(T-14)*np.sqrt(max(0.1, 35-T)))
        c = max(0.01, 0.0007*T*(T-12)*np.sqrt(max(0.1, 37-T)))
        return a * c
    
    @staticmethod
    def b_vh(T):
        """蚊→人传播率 = a × b"""
        if T < 14 or T > 35:
            return 0.01
        a = max(0.01, 0.0005*T*(T-14)*np.sqrt(max(0.1, 35-T)))
        b = max(0.01, 0.0008*T*(T-17)*np.sqrt(max(0.1, 36-T)))
        return a * b


# 固定参数
SIGMA = 0.5       # 雌蚊比例
MU_EM = 0.1       # 羽化死亡率
DELTA = 1/5       # 人潜伏期转化率 (5天)
GAMMA = 1/6       # 康复率 (6天)
D = 0             # 人自然死亡率 (忽略，短期模型)
A = 0             # 人出生率 (忽略)

N_H = 14_000_000  # 人口

# ============================================================
# 2. 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()

# BI数据
bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
gz_bi.columns = ['year', 'month', 'bi']

# 合并
df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
df['bi'] = df['bi'].fillna(df['bi'].mean())

# 用温度填补缺失
for idx in df[df['bi'].isna()].index:
    T = df.loc[idx, 'temperature']
    df.loc[idx, 'bi'] = 2 * np.exp(-((T - 27) / 8) ** 2)

log(f"  数据范围: {df['year'].min()}-{df['year'].max()}")
log(f"  月数据: {len(df)}个月")
log(f"  总病例: {df['cases'].sum():,}")
log(f"  BI范围: {df['bi'].min():.2f} - {df['bi'].max():.2f}")

# 周数据
n_months = len(df)
n_weeks = int(n_months * 4.33)

monthly_idx = np.arange(n_months) * 4.33
weekly_idx = np.arange(n_weeks)

temp_interp = interp1d(monthly_idx, df['temperature'].values, kind='cubic', fill_value='extrapolate')
bi_interp = interp1d(monthly_idx, df['bi'].values, kind='cubic', fill_value='extrapolate')
cases_interp = interp1d(monthly_idx, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')

weekly_temp = temp_interp(weekly_idx)
weekly_bi = np.maximum(0.1, bi_interp(weekly_idx))
weekly_cases = np.maximum(0, cases_interp(weekly_idx))

# BI 归一化 (相对指标)
bi_mean = weekly_bi.mean()
bi_normalized = weekly_bi / bi_mean  # 相对于平均值

log(f"  周数据: {n_weeks}周")
log(f"  BI均值: {bi_mean:.2f}, 归一化范围: {bi_normalized.min():.2f} - {bi_normalized.max():.2f}")

# ============================================================
# 3. ODE 模型
# ============================================================
def sei_seir_ode(y, t, params_func, bi_func, est_params, N_h):
    """
    完整 SEI-SEIR 模型
    
    状态变量 (10个):
    [E, L, P, S_m, E_m, I_m, S_h, E_h, I_h, R_h]
    
    E, L, P: 蚊虫水生阶段 (卵、幼虫、蛹)
    S_m, E_m, I_m: 成蚊 SEI
    S_h, E_h, I_h, R_h: 人群 SEIR
    """
    E, L, P, S_m, E_m, I_m, S_h, E_h, I_h, R_h = y
    
    # 当前温度和BI
    idx = min(int(t / 7), len(weekly_temp) - 1)
    T = weekly_temp[idx]
    bi_ratio = bi_func(t)  # BI 相对值
    
    # 温度依赖参数
    p = TempParams()
    phi = p.phi(T)
    f_e = p.f_e(T)
    f_l = p.f_l(T)
    f_p = p.f_p(T)
    mu_e = p.mu_e(T)
    mu_l = p.mu_l(T)
    mu_p = p.mu_p(T)
    mu_m = p.mu_m(T)
    epsilon = p.epsilon(T)
    b_hv_t = p.b_hv(T)
    b_vh_t = p.b_vh(T)
    
    # 估计参数
    k_l, k_p, b_h_scale, imp = est_params
    
    # 环境容纳量
    K_l = k_l * N_h  # 幼虫容纳量
    K_p = k_p * N_h  # 蛹容纳量
    
    # 传播系数缩放
    b_hv_eff = b_hv_t * b_h_scale
    b_vh_eff = b_vh_t * b_h_scale
    
    # 成蚊总数
    M_total = S_m + E_m + I_m
    
    # === 蚊虫水生阶段 ===
    # 卵: 产卵 - 孵化 - 死亡
    dE = phi * SIGMA * M_total - (f_e + mu_e) * E
    
    # 幼虫: 孵化 - 发育 - 死亡 (密度依赖)
    # 用 BI 校正幼虫动态
    mu_l_eff = mu_l * (1 + L / K_l)  # 密度依赖死亡
    dL = f_e * E * bi_ratio - (f_l + mu_l_eff) * L  # bi_ratio 作为校正因子
    
    # 蛹: 发育 - 羽化 - 死亡
    dP = f_l * L - (f_p + mu_p) * P
    
    # === 成蚊 SEI ===
    # 羽化率 (考虑密度依赖存活)
    emergence = SIGMA * f_p * P * np.exp(-MU_EM * (1 + P / K_p))
    
    # 感染率
    infection_rate_m = b_hv_eff * (I_h + imp) / N_h
    
    # 易感成蚊
    dS_m = emergence - infection_rate_m * S_m - mu_m * S_m
    
    # 暴露成蚊
    dE_m = infection_rate_m * S_m - epsilon * E_m - mu_m * E_m
    
    # 感染成蚊
    dI_m = epsilon * E_m - mu_m * I_m
    
    # === 人群 SEIR ===
    # 感染率
    infection_rate_h = b_vh_eff * I_m / N_h
    
    # 易感人群
    dS_h = -infection_rate_h * S_h
    
    # 暴露人群
    dE_h = infection_rate_h * S_h - DELTA * E_h
    
    # 感染人群
    dI_h = DELTA * E_h - GAMMA * I_h
    
    # 恢复人群
    dR_h = GAMMA * I_h
    
    return [dE, dL, dP, dS_m, dE_m, dI_m, dS_h, dE_h, dI_h, dR_h]


# ============================================================
# 4. 参数估计
# ============================================================
log("\n[2] 参数估计...")
log("  待估计参数:")
log("    k_l: 幼虫环境容纳量系数")
log("    k_p: 蛹环境容纳量系数")
log("    b_h_scale: 传播系数缩放")
log("    imp: 输入病例")

time_days = np.arange(n_weeks) * 7

# BI 插值函数
bi_ratio_interp = interp1d(time_days, bi_normalized, kind='linear', fill_value='extrapolate')

def objective(params):
    """目标函数"""
    k_l, k_p, b_h_scale, imp = params
    
    # 初始条件
    # 蚊虫 (根据初始温度和BI估计)
    T0 = weekly_temp[0]
    p = TempParams()
    M0 = N_H * 0.01  # 初始成蚊
    P0 = M0 * 2
    L0 = M0 * 3 * bi_normalized[0]  # 用BI校正
    E0 = M0 * 5
    
    S_m0 = M0 * 0.97
    E_m0 = M0 * 0.02
    I_m0 = M0 * 0.01
    
    # 人群
    I_h0 = 100
    E_h0 = I_h0 * 2
    S_h0 = N_H - E_h0 - I_h0
    R_h0 = 0
    
    y0 = [E0, L0, P0, S_m0, E_m0, I_m0, S_h0, E_h0, I_h0, R_h0]
    
    try:
        sol = odeint(sei_seir_ode, y0, time_days,
                     args=(TempParams, bi_ratio_interp, params, N_H),
                     rtol=1e-6, atol=1e-9)
        
        I_h = sol[:, 8]
        
        # 周新增病例
        weekly_pred = GAMMA * I_h * 7
        weekly_pred = np.maximum(weekly_pred, 0)
        
        # 对数空间MSE
        obs_log = np.log1p(weekly_cases)
        pred_log = np.log1p(weekly_pred)
        mse = np.mean((obs_log - pred_log) ** 2)
        
        # 相关性奖励
        if np.std(weekly_pred) > 1e-6:
            corr, _ = pearsonr(weekly_cases, weekly_pred)
            if not np.isnan(corr) and corr > 0:
                mse -= 0.3 * corr
        
        # BI-幼虫相关性约束
        L = sol[:, 1]
        L_normalized = L / L.mean() if L.mean() > 0 else L
        corr_bi_L, _ = pearsonr(bi_normalized, L_normalized)
        if corr_bi_L < 0.3:  # BI和幼虫应该正相关
            mse += 1.0
        
        return mse
        
    except Exception as e:
        return 1e10

# 优化
log("  优化中...")

bounds = [
    (0.0001, 0.01),   # k_l
    (0.0001, 0.01),   # k_p
    (0.1, 100),       # b_h_scale
    (0, 20),          # imp
]

iter_count = [0]
def callback(xk, convergence=None):
    iter_count[0] += 1
    if iter_count[0] % 30 == 0:
        log(f"    迭代 {iter_count[0]}...")
    return False

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=150,
    tol=1e-6,
    workers=1,
    callback=callback,
    polish=True
)

k_l, k_p, b_h_scale, imp = result.x

log(f"\n  估计结果:")
log(f"    k_l = {k_l:.6f}")
log(f"    k_p = {k_p:.6f}")
log(f"    b_h_scale = {b_h_scale:.2f}")
log(f"    imp = {imp:.2f} /周")
log(f"    Loss = {result.fun:.4f}")

# ============================================================
# 5. 运行最终模型
# ============================================================
log("\n[3] 运行最终模型...")

# 初始条件
T0 = weekly_temp[0]
M0 = N_H * 0.01
P0, L0, E0 = M0 * 2, M0 * 3 * bi_normalized[0], M0 * 5
S_m0, E_m0, I_m0 = M0 * 0.97, M0 * 0.02, M0 * 0.01
I_h0, E_h0 = 100, 200
S_h0, R_h0 = N_H - E_h0 - I_h0, 0

y0 = [E0, L0, P0, S_m0, E_m0, I_m0, S_h0, E_h0, I_h0, R_h0]

sol = odeint(sei_seir_ode, y0, time_days,
             args=(TempParams, bi_ratio_interp, result.x, N_H))

# 提取结果
E_mosq, L_mosq, P_mosq = sol[:, 0], sol[:, 1], sol[:, 2]
S_m, E_m, I_m = sol[:, 3], sol[:, 4], sol[:, 5]
S_h, E_h, I_h, R_h = sol[:, 6], sol[:, 7], sol[:, 8], sol[:, 9]

M_total = S_m + E_m + I_m
weekly_pred = GAMMA * I_h * 7
weekly_pred = np.maximum(weekly_pred, 0)

# ============================================================
# 6. 评估
# ============================================================
log("\n[4] 模型评估...")

# 病例预测
corr_cases, pval = pearsonr(weekly_cases, weekly_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(weekly_pred))

# BI-幼虫相关性
L_normalized = L_mosq / L_mosq.mean() if L_mosq.mean() > 0 else L_mosq
corr_bi_L, _ = pearsonr(bi_normalized, L_normalized)

log(f"  病例预测:")
log(f"    相关系数: {corr_cases:.4f} (p={pval:.2e})")
log(f"    R² (对数): {r2_log:.4f}")

log(f"  BI校正效果:")
log(f"    BI-幼虫相关: {corr_bi_L:.4f}")

# R0 估计
p = TempParams()
R0_arr = []
for i in range(n_weeks):
    T = weekly_temp[i]
    M = M_total[i]
    b_vh = p.b_vh(T) * b_h_scale
    b_hv = p.b_hv(T) * b_h_scale
    mu_m = p.mu_m(T)
    eps = p.epsilon(T)
    
    # R0 ≈ sqrt(b_vh × b_hv × M / (μ_m × γ × N_h))
    R0 = np.sqrt(b_vh * b_hv * M / (mu_m * GAMMA * N_H + 1e-10))
    R0_arr.append(R0)

R0_arr = np.array(R0_arr)
log(f"  R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")
log(f"  R0>1 周数: {(R0_arr > 1).sum()}/{n_weeks}")

# ============================================================
# 7. 可视化
# ============================================================
log("\n[5] 生成可视化...")

fig = plt.figure(figsize=(18, 16))
weeks = range(n_weeks)

# 1. 病例对比
ax = fig.add_subplot(3, 3, 1)
ax.plot(weeks, weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weeks, weekly_pred, 'r-', lw=1.5, label='Model', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(f'Weekly Cases (r={corr_cases:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数尺度
ax = fig.add_subplot(3, 3, 2)
ax.semilogy(weeks, weekly_cases + 1, 'b-', lw=1.5, label='Observed')
ax.semilogy(weeks, weekly_pred + 1, 'r-', lw=1.5, label='Model')
ax.set_xlabel('Week')
ax.set_ylabel('Cases (log)')
ax.set_title(f'Log Scale (R²={r2_log:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. R0
ax = fig.add_subplot(3, 3, 3)
ax.plot(weeks, R0_arr, 'g-', lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2)
ax.fill_between(weeks, 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red')
ax.set_xlabel('Week')
ax.set_ylabel('R0')
ax.set_title('Reproduction Number R0(t)')
ax.grid(True, alpha=0.3)

# 4. BI vs 幼虫
ax = fig.add_subplot(3, 3, 4)
ax.plot(weeks, bi_normalized, 'g-', lw=1.5, label='BI (normalized)', alpha=0.8)
ax.plot(weeks, L_normalized, 'b-', lw=1.5, label='L (model)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Normalized Value')
ax.set_title(f'BI vs Larvae (r={corr_bi_L:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 蚊虫水生阶段
ax = fig.add_subplot(3, 3, 5)
ax.plot(weeks, E_mosq / 1e6, 'b-', lw=1, label='E (eggs)', alpha=0.7)
ax.plot(weeks, L_mosq / 1e6, 'g-', lw=1, label='L (larvae)', alpha=0.7)
ax.plot(weeks, P_mosq / 1e6, 'y-', lw=1, label='P (pupae)', alpha=0.7)
ax.set_xlabel('Week')
ax.set_ylabel('Population (millions)')
ax.set_title('Aquatic Stages')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 成蚊动态
ax = fig.add_subplot(3, 3, 6)
ax.plot(weeks, S_m / 1e6, 'g-', lw=1, label='S_m', alpha=0.7)
ax.plot(weeks, E_m / 1e3, 'y-', lw=1, label='E_m (×1000)', alpha=0.7)
ax.plot(weeks, I_m / 1e3, 'r-', lw=1, label='I_m (×1000)', alpha=0.7)
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Adult Mosquito SEI')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. 人群动态
ax = fig.add_subplot(3, 3, 7)
ax.plot(weeks, E_h, 'y-', lw=1.5, label='E_h', alpha=0.8)
ax.plot(weeks, I_h, 'r-', lw=1.5, label='I_h', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Human E and I')
ax.legend()
ax.grid(True, alpha=0.3)

# 8. 散点图
ax = fig.add_subplot(3, 3, 8)
ax.scatter(weekly_cases, weekly_pred, alpha=0.5, s=30)
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter')
ax.grid(True, alpha=0.3)

# 9. 温度
ax = fig.add_subplot(3, 3, 9)
ax.plot(weeks, weekly_temp, 'orange', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/full_sei_seir_model.png', dpi=150, bbox_inches='tight')
plt.close()

log("  已保存: results/figures/full_sei_seir_model.png")

# 保存结果
results_df = pd.DataFrame({
    'week': weeks,
    'temperature': weekly_temp,
    'bi': weekly_bi,
    'bi_normalized': bi_normalized,
    'observed_cases': weekly_cases,
    'predicted_cases': weekly_pred,
    'E_mosq': E_mosq,
    'L_mosq': L_mosq,
    'P_mosq': P_mosq,
    'M_total': M_total,
    'I_m': I_m,
    'I_h': I_h,
    'R0': R0_arr
})
results_df.to_csv('/root/wenmei/results/data/full_sei_seir_results.csv', index=False)
log("  已保存: results/data/full_sei_seir_results.csv")

# ============================================================
# 8. 总结
# ============================================================
log("\n" + "=" * 70)
log("模型总结")
log("=" * 70)
log(f"""
【模型结构】
  蚊虫水生阶段: E (卵) → L (幼虫) → P (蛹)
  成蚊 SEI: S_m → E_m → I_m
  人群 SEIR: S_h → E_h → I_h → R_h

【BI 校正】
  BI 作为幼虫密度的相对指标
  校正方式: dL/dt 中的孵化项乘以 BI/BI_mean
  BI-幼虫相关性: {corr_bi_L:.4f}

【估计参数】
  k_l = {k_l:.6f} (幼虫容纳量系数)
  k_p = {k_p:.6f} (蛹容纳量系数)
  b_h_scale = {b_h_scale:.2f} (传播系数缩放)
  imp = {imp:.2f} /周 (输入病例)

【模型性能】
  病例相关系数: {corr_cases:.4f}
  R² (对数): {r2_log:.4f}
  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]

【物理意义】
  - BI↑ → 幼虫L↑ → 成蚊M↑ → 病例↑
  - 温度影响所有发育率和死亡率
  - 传播系数由温度和校正因子共同决定
""")

log("\n完成!")
