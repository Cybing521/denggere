#!/usr/bin/env python3
"""
SEI-SEIR 模型 (Zhang et al. 2025 风格)

关键特点:
1. 温度依赖参数用文献公式: a(T), b(T), c(T), μ_m(T)
2. 只估计 b_h (蚊→人传播系数) 和 imp (输入病例)
3. b_vh(t) = a(T) × b(T) × b_h
4. 周尺度建模
"""
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 65)
log("SEI-SEIR 模型 (Zhang et al. 2025 风格)")
log("=" * 65)

# ============================================================
# 1. 温度依赖参数 (Table 2 公式)
# ============================================================
def a_T(T):
    """叮咬率 a(T) - 公式(9)"""
    if T < 14 or T > 35:
        return 0.001
    return max(0.001, 0.0005 * T * (T - 14) * np.sqrt(35 - T))

def b_T(T):
    """感染概率 蚊→人 b(T) - 公式(10)"""
    if T < 17.05 or T > 35.83:
        return 0.001
    return max(0.001, 0.0008 * T * (T - 17.05) * np.sqrt(35.83 - T))

def c_T(T):
    """感染概率 人→蚊 c(T) - 公式(11)"""
    if T < 12.22 or T > 37.46:
        return 0.001
    return max(0.001, 0.0007 * T * (T - 12.22) * np.sqrt(37.46 - T))

def mu_m_T(T):
    """成蚊死亡率 μ_m(T) - 公式(8)"""
    if T < 10 or T > 40:
        return 0.5
    return max(0.02, 0.0006 * T**2 - 0.028 * T + 0.37)

def phi_T(T):
    """产卵率 φ(T) - 公式(2)"""
    if T < 14 or T > 35:
        return 0.01
    return max(0.01, -0.02 * T**2 + 1.1 * T - 8.5)

# 固定参数 (Table 2)
SIGMA = 0.5       # 雌蚊比例
DELTA = 1/5       # 人潜伏期转化率 (5天)
GAMMA = 1/6       # 康复率 (6天)
B_V = 1.0         # 人→蚊传播系数 (固定为1)

log("\n【固定参数 (文献)】")
log(f"  σ (雌蚊比例) = {SIGMA}")
log(f"  δ (人潜伏期转化率) = {DELTA:.3f} /天")
log(f"  γ (康复率) = {GAMMA:.4f} /天")
log(f"  b_v (人→蚊传播系数) = {B_V}")

# ============================================================
# 2. 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()

N_H = 14_000_000  # 广州人口

log(f"  月数据: {len(df)}个月")
log(f"  病例总数: {df['cases'].sum():,}")
log(f"  人口: {N_H:,}")

# 转周数据 (每月约4.33周)
n_months = len(df)
n_weeks = int(n_months * 4.33)

# 插值
monthly_idx = np.arange(n_months) * 4.33
weekly_idx = np.arange(n_weeks)

temp_interp = interp1d(monthly_idx, df['temperature'].values, kind='cubic', fill_value='extrapolate')
weekly_temp = temp_interp(weekly_idx)

cases_interp = interp1d(monthly_idx, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')
weekly_cases = np.maximum(0, cases_interp(weekly_idx))

log(f"  周数据: {n_weeks}周")
log(f"  温度范围: {weekly_temp.min():.1f} - {weekly_temp.max():.1f}°C")
log(f"  周均病例: {weekly_cases.mean():.1f}")

# ============================================================
# 3. 计算温度依赖参数序列
# ============================================================
log("\n[2] 计算温度依赖参数...")

a_arr = np.array([a_T(T) for T in weekly_temp])
b_arr = np.array([b_T(T) for T in weekly_temp])
c_arr = np.array([c_T(T) for T in weekly_temp])
mu_m_arr = np.array([mu_m_T(T) for T in weekly_temp])
phi_arr = np.array([phi_T(T) for T in weekly_temp])

log(f"  a(T) 叮咬率: {a_arr.min():.4f} - {a_arr.max():.4f}")
log(f"  b(T) 蚊→人: {b_arr.min():.4f} - {b_arr.max():.4f}")
log(f"  c(T) 人→蚊: {c_arr.min():.4f} - {c_arr.max():.4f}")
log(f"  μ_m(T) 死亡率: {mu_m_arr.min():.4f} - {mu_m_arr.max():.4f}")

# ============================================================
# 4. 蚊虫动态 (简化: 准稳态)
# ============================================================
log("\n[3] 估计蚊虫密度...")

# 平衡态蚊虫数: M* ≈ φ × K / μ_m (简化)
K = N_H * 0.1  # 环境容纳量 (调小，更合理)

# 蚊虫密度随温度变化
M_arr = phi_arr * K / (mu_m_arr + 0.01)
M_arr = np.clip(M_arr, 1e4, N_H * 0.5)  # 限制在合理范围

# 归一化
M_normalized = M_arr / M_arr.max()

log(f"  蚊虫密度范围: {M_arr.min():.2e} - {M_arr.max():.2e}")
log(f"  蚊虫/人口比: {(M_arr/N_H).min():.4f} - {(M_arr/N_H).max():.4f}")

# ============================================================
# 5. SEI-SEIR 模型
# ============================================================
def seir_model(y, t, params_arr, b_h, imp, N_h):
    """
    简化SEIR模型
    
    状态: [S, E, I, R]
    
    传播率: b_vh(t) = a(t) × b(t) × b_h
    感染力: λ(t) = b_vh(t) × M(t) × I / N_h
    """
    S, E, I, R = y
    
    # 当前时间索引
    idx = min(int(t / 7), len(params_arr['a']) - 1)
    
    # 温度依赖参数
    a_t = params_arr['a'][idx]
    b_t = params_arr['b'][idx]
    M_t = params_arr['M'][idx]
    
    # 有效传播率 (Zhang公式)
    b_vh = a_t * b_t * b_h
    
    # 感染蚊比例估计 (简化: 与I/N成正比)
    # 这里假设感染蚊比例 ∝ 人群感染率
    c_t = params_arr['c'][idx]
    i_v = c_t * a_t * I / N_h  # 感染蚊比例
    
    # 感染力
    lambda_t = b_vh * M_t * i_v / N_h
    
    # 输入病例 (每天)
    imp_daily = imp / 7.0
    
    # SEIR方程
    dS = -lambda_t * S
    dE = lambda_t * S + imp_daily - DELTA * E
    dI = DELTA * E - GAMMA * I
    dR = GAMMA * I
    
    return [dS, dE, dI, dR]


# ============================================================
# 6. 参数估计
# ============================================================
log("\n[4] 参数估计...")
log("  待估计参数:")
log("    b_h: 蚊→人有效传播系数")
log("    imp: 周输入病例数")

time_days = np.arange(n_weeks) * 7

params_arr = {
    'a': a_arr,
    'b': b_arr,
    'c': c_arr,
    'M': M_arr
}

def objective(x):
    """目标函数"""
    b_h, imp, I0_log = x
    I0 = 10 ** I0_log
    
    y0 = [N_H - I0 * 3, I0 * 2, I0, 0]
    
    try:
        sol = odeint(seir_model, y0, time_days, 
                     args=(params_arr, b_h, imp, N_H),
                     rtol=1e-6, atol=1e-9)
        
        I_h = sol[:, 2]
        
        # 周新增病例
        weekly_pred = GAMMA * I_h * 7
        
        # 确保非负
        weekly_pred = np.maximum(weekly_pred, 0)
        
        # 对数空间MSE
        obs_log = np.log1p(weekly_cases)
        pred_log = np.log1p(weekly_pred)
        
        mse = np.mean((obs_log - pred_log) ** 2)
        
        # 相关性奖励
        if np.std(weekly_pred) > 1e-6:
            corr, _ = pearsonr(weekly_cases, weekly_pred)
            if not np.isnan(corr) and corr > 0:
                mse -= 0.5 * corr  # 奖励正相关
        
        return mse
        
    except Exception as e:
        return 1e10

# 使用差分进化优化
log("  优化中...")

bounds = [
    (1, 500),       # b_h: 传播系数 (需要较大值来补偿)
    (0, 20),        # imp: 周输入病例
    (-7, -3),       # log10(I0)
]

# 回调函数显示进度
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
    maxiter=200,
    tol=1e-7,
    workers=1,
    callback=callback,
    polish=True
)

b_h_opt, imp_opt, I0_log_opt = result.x
I0_opt = 10 ** I0_log_opt

log(f"\n  估计结果:")
log(f"    b_h = {b_h_opt:.2f}")
log(f"    imp = {imp_opt:.2f} /周")
log(f"    I0 = {I0_opt:.2e}")
log(f"    Loss = {result.fun:.4f}")

# ============================================================
# 7. 运行最终模型
# ============================================================
log("\n[5] 运行最终模型...")

y0 = [N_H - I0_opt * 3, I0_opt * 2, I0_opt, 0]
sol = odeint(seir_model, y0, time_days, 
             args=(params_arr, b_h_opt, imp_opt, N_H))

S_h, E_h, I_h, R_h = sol.T
weekly_pred = GAMMA * I_h * 7
weekly_pred = np.maximum(weekly_pred, 0)

# ============================================================
# 8. 计算 R0(t)
# ============================================================
# R0 ≈ b_vh × M × (a × c) / (μ_m × γ × N_h)
# 简化: R0 ∝ b_h × a² × b × c × M / (μ_m × γ × N_h)

R0_arr = []
for i in range(n_weeks):
    a_t = a_arr[i]
    b_t = b_arr[i]
    c_t = c_arr[i]
    M_t = M_arr[i]
    mu_m_t = mu_m_arr[i]
    
    R0 = b_h_opt * (a_t ** 2) * b_t * c_t * M_t / (mu_m_t * GAMMA * N_H)
    R0_arr.append(R0)

R0_arr = np.array(R0_arr)

# ============================================================
# 9. 评估
# ============================================================
log("\n[6] 模型评估...")

corr, pval = pearsonr(weekly_cases, weekly_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(weekly_pred))
r2_linear = r2_score(weekly_cases, weekly_pred)

# 趋势准确率
trend_obs = np.diff(weekly_cases) > 0
trend_pred = np.diff(weekly_pred) > 0
trend_acc = np.mean(trend_obs == trend_pred)

log(f"  相关系数: {corr:.4f} (p={pval:.2e})")
log(f"  R² (对数): {r2_log:.4f}")
log(f"  R² (线性): {r2_linear:.4f}")
log(f"  趋势准确率: {trend_acc:.2%}")
log(f"  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")
log(f"  R0 > 1 周数: {(R0_arr > 1).sum()}/{n_weeks}")

# ============================================================
# 10. 可视化
# ============================================================
log("\n[7] 生成可视化...")

fig = plt.figure(figsize=(18, 14))

# 1. 病例对比
ax = fig.add_subplot(3, 3, 1)
weeks = range(n_weeks)
ax.plot(weeks, weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weeks, weekly_pred, 'r-', lw=1.5, label='Model', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(f'Weekly Cases (r={corr:.3f})')
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

# 3. R0(t)
ax = fig.add_subplot(3, 3, 3)
ax.plot(weeks, R0_arr, 'g-', lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2, label='R0=1')
ax.fill_between(weeks, 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red')
ax.set_xlabel('Week')
ax.set_ylabel('R0(t)')
ax.set_title('Reproduction Number')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 传播参数
ax = fig.add_subplot(3, 3, 4)
b_vh_arr = a_arr * b_arr * b_h_opt
ax.plot(weeks, b_vh_arr, 'purple', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('b_vh(t)')
ax.set_title(f'Effective Transmission Rate\nb_vh = a(T) × b(T) × b_h')
ax.grid(True, alpha=0.3)

# 5. 温度 vs 病例
ax = fig.add_subplot(3, 3, 5)
ax2 = ax.twinx()
ax.plot(weeks, weekly_temp, 'orange', lw=1.5, label='Temperature')
ax2.plot(weeks, weekly_cases, 'b-', lw=1, alpha=0.7, label='Cases')
ax.set_xlabel('Week')
ax.set_ylabel('Temperature (°C)', color='orange')
ax2.set_ylabel('Cases', color='blue')
ax.set_title('Temperature vs Cases')
ax.grid(True, alpha=0.3)

# 6. 蚊虫密度
ax = fig.add_subplot(3, 3, 6)
ax.plot(weeks, M_arr / 1e6, 'm-', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('Mosquitoes (millions)')
ax.set_title('Estimated Mosquito Density')
ax.grid(True, alpha=0.3)

# 7. 散点图
ax = fig.add_subplot(3, 3, 7)
colors = np.arange(n_weeks)
sc = ax.scatter(weekly_cases, weekly_pred, c=colors, cmap='viridis', s=30, alpha=0.6)
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed Cases')
ax.set_ylabel('Predicted Cases')
ax.set_title('Scatter Plot')
plt.colorbar(sc, ax=ax, label='Week')
ax.grid(True, alpha=0.3)

# 8. 人群动态
ax = fig.add_subplot(3, 3, 8)
ax.plot(weeks, E_h, 'y-', lw=1, label='E (Exposed)')
ax.plot(weeks, I_h, 'r-', lw=1, label='I (Infectious)')
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Human E and I Compartments')
ax.legend()
ax.grid(True, alpha=0.3)

# 9. 年度对比
ax = fig.add_subplot(3, 3, 9)
# 按年汇总
year_labels = 2015 + (np.arange(n_weeks) // 52)
df_yearly = pd.DataFrame({
    'year': year_labels,
    'observed': weekly_cases,
    'predicted': weekly_pred
})
yearly = df_yearly.groupby('year').sum().reset_index()

x = range(len(yearly))
width = 0.35
ax.bar([i - width/2 for i in x], yearly['observed'], width, label='Observed', color='steelblue')
ax.bar([i + width/2 for i in x], yearly['predicted'], width, label='Predicted', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(yearly['year'].astype(int))
ax.set_xlabel('Year')
ax.set_ylabel('Annual Cases')
ax.set_title('Annual Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/zhang_style_model.png', dpi=150, bbox_inches='tight')
plt.close()

log("  已保存: results/figures/zhang_style_model.png")

# 保存数据
results_df = pd.DataFrame({
    'week': weeks,
    'temperature': weekly_temp,
    'observed': weekly_cases,
    'predicted': weekly_pred,
    'M': M_arr,
    'R0': R0_arr,
    'a_T': a_arr,
    'b_T': b_arr,
    'c_T': c_arr,
    'b_vh': a_arr * b_arr * b_h_opt,
    'I_h': I_h,
    'E_h': E_h
})
results_df.to_csv('/root/wenmei/results/data/zhang_style_model.csv', index=False)
log("  已保存: results/data/zhang_style_model.csv")

# ============================================================
# 11. 模型总结
# ============================================================
log("\n" + "=" * 65)
log("模型总结")
log("=" * 65)
log(f"""
【模型结构】
  人群: S → E → I → R
  传播率: b_vh(t) = a(T) × b(T) × b_h

【温度依赖参数 (文献公式)】
  a(T): 叮咬率 - 公式(9)
  b(T): 感染概率(蚊→人) - 公式(10)
  c(T): 感染概率(人→蚊) - 公式(11)
  μ_m(T): 成蚊死亡率 - 公式(8)

【固定参数 (文献)】
  δ = {DELTA:.3f} /天 (潜伏期转化率, 1/5天)
  γ = {GAMMA:.4f} /天 (康复率, 1/6天)
  N_h = {N_H:,}

【估计参数】
  b_h = {b_h_opt:.2f} (蚊→人传播系数)
  imp = {imp_opt:.2f} /周 (输入病例)

【模型性能】
  相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  趋势准确率: {trend_acc:.2%}
  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]

【公式】
  b_vh(t) = a(T) × b(T) × b_h
  λ(t) = b_vh(t) × M(t) × i_v(t) / N_h
  R0(t) = b_h × a² × b × c × M / (μ_m × γ × N_h)
""")

log("\n完成!")
