#!/usr/bin/env python3
"""
实用的登革热预测模型

问题分析:
- 2015-2019年病例数少(年均2600例)，没有大暴发
- 标准ODE模型假设持续传播，不适合这种情况
- 实际可能是: 输入病例 → 短暂本地传播 → 消退

解决方案:
- 不用复杂的ODE耦合
- 直接用温度依赖的传播势能 + 输入驱动
- 类似 PNAS 的 β(t) = β' × M(t) 但更简单
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 65)
log("实用登革热预测模型")
log("=" * 65)

# ============================================================
# 1. 温度依赖函数 (来自文献)
# ============================================================
def transmission_potential(T):
    """
    温度依赖的传播势能
    综合了 a(T), b(T), c(T), 1/μ_m(T)
    """
    if T < 15 or T > 35:
        return 0.01
    
    # 各参数
    a = max(0.01, 0.0005 * T * (T - 14) * np.sqrt(max(0.01, 35 - T)))
    b = max(0.01, 0.0008 * T * (T - 17) * np.sqrt(max(0.01, 36 - T)))
    c = max(0.01, 0.0007 * T * (T - 12) * np.sqrt(max(0.01, 37 - T)))
    mu_m = max(0.02, 0.0006 * T**2 - 0.028 * T + 0.37)
    
    # 综合传播势能 ∝ a² × b × c / μ_m
    potential = (a ** 2) * b * c / mu_m
    return potential

# ============================================================
# 2. 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()

log(f"  月数据: {len(df)}个月")
log(f"  病例: {df['cases'].sum():,}")

# 月数据
monthly_temp = df['temperature'].values
monthly_cases = df['cases'].values
n_months = len(df)

# 计算传播势能
P_arr = np.array([transmission_potential(T) for T in monthly_temp])

log(f"  温度范围: {monthly_temp.min():.1f} - {monthly_temp.max():.1f}°C")
log(f"  传播势能范围: {P_arr.min():.4f} - {P_arr.max():.4f}")

# ============================================================
# 3. 分析传播势能与病例的滞后相关性
# ============================================================
log("\n[2] 滞后相关性分析...")

max_lag = 4
best_lag = 0
best_corr = -1

for lag in range(0, max_lag + 1):
    if lag == 0:
        corr, p = pearsonr(P_arr, monthly_cases)
    else:
        corr, p = pearsonr(P_arr[:-lag], monthly_cases[lag:])
    
    log(f"  滞后 {lag} 月: r = {corr:.4f}")
    
    if corr > best_corr:
        best_corr = corr
        best_lag = lag

log(f"  最佳滞后: {best_lag} 月, r = {best_corr:.4f}")

# ============================================================
# 4. 简单预测模型
# ============================================================
log("\n[3] 构建预测模型...")

# 特征: 传播势能 + 滞后
features = []
targets = []

for i in range(3, n_months):  # 从第4个月开始，需要滞后特征
    feat = [
        P_arr[i],           # 当月传播势能
        P_arr[i-1],         # 滞后1月
        P_arr[i-2],         # 滞后2月
        monthly_cases[i-1], # 上月病例
        monthly_cases[i-2], # 滞后2月病例
        np.sin(2 * np.pi * (i % 12) / 12),  # 季节性
        np.cos(2 * np.pi * (i % 12) / 12),
    ]
    features.append(feat)
    targets.append(monthly_cases[i])

X = np.array(features)
y = np.array(targets)

log(f"  特征数: {X.shape[1]}")
log(f"  样本数: {X.shape[0]}")

# 对数变换
y_log = np.log1p(y)

# 训练/测试分割
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
y_log_train, y_log_test = y_log[:split], y_log[split:]

# 岭回归 (对数空间)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_log_train)

# 预测
y_log_pred_train = model.predict(X_train_scaled)
y_log_pred_test = model.predict(X_test_scaled)

y_pred_train = np.expm1(y_log_pred_train)
y_pred_test = np.expm1(y_log_pred_test)

# 全量预测
X_all_scaled = scaler.transform(X)
y_log_pred_all = model.predict(X_all_scaled)
y_pred_all = np.expm1(y_log_pred_all)

# ============================================================
# 5. 评估
# ============================================================
log("\n[4] 模型评估...")

# 训练集
corr_train, _ = pearsonr(y_train, y_pred_train)
r2_train = r2_score(y_log_train, y_log_pred_train)

# 测试集
corr_test, _ = pearsonr(y_test, y_pred_test)
r2_test = r2_score(y_log_test, y_log_pred_test)

# 全量
corr_all, _ = pearsonr(y, y_pred_all)
r2_all = r2_score(y_log, y_log_pred_all)

log(f"  训练集 (n={len(y_train)}):")
log(f"    相关系数: {corr_train:.4f}")
log(f"    R² (对数): {r2_train:.4f}")

log(f"  测试集 (n={len(y_test)}):")
log(f"    相关系数: {corr_test:.4f}")
log(f"    R² (对数): {r2_test:.4f}")

log(f"  全量 (n={len(y)}):")
log(f"    相关系数: {corr_all:.4f}")
log(f"    R² (对数): {r2_all:.4f}")

# 特征重要性
feature_names = ['P(t)', 'P(t-1)', 'P(t-2)', 'Cases(t-1)', 'Cases(t-2)', 'sin', 'cos']
log("\n  特征权重:")
for name, coef in zip(feature_names, model.coef_):
    log(f"    {name}: {coef:.4f}")

# ============================================================
# 6. 基于传播势能的R0估计
# ============================================================
log("\n[5] R0 估计...")

# R0 ∝ P(T) × 常数
# 校准: 使 R0 在合理范围 (0.5 - 3)
P_max = P_arr.max()
R0_scale = 2.5 / P_max  # 使最大R0约为2.5

R0_arr = P_arr * R0_scale

log(f"  R0 范围: [{R0_arr.min():.3f}, {R0_arr.max():.3f}]")
log(f"  R0 > 1 月数: {(R0_arr > 1).sum()}/{n_months}")

# ============================================================
# 7. 可视化
# ============================================================
log("\n[6] 生成可视化...")

fig = plt.figure(figsize=(16, 12))

# 1. 病例对比
ax = fig.add_subplot(2, 3, 1)
months = range(3, n_months)
ax.plot(months, y, 'b-o', ms=4, lw=1.5, label='Observed', alpha=0.8)
ax.plot(months, y_pred_all, 'r-s', ms=4, lw=1.5, label='Predicted', alpha=0.8)
ax.axvline(x=3+split, color='green', ls='--', label='Train/Test')
ax.set_xlabel('Month')
ax.set_ylabel('Cases')
ax.set_title(f'Monthly Cases (r={corr_all:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数尺度
ax = fig.add_subplot(2, 3, 2)
ax.semilogy(months, y + 1, 'b-o', ms=4, lw=1.5, label='Observed')
ax.semilogy(months, y_pred_all + 1, 'r-s', ms=4, lw=1.5, label='Predicted')
ax.axvline(x=3+split, color='green', ls='--')
ax.set_xlabel('Month')
ax.set_ylabel('Cases (log)')
ax.set_title(f'Log Scale (R²={r2_all:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. R0(t)
ax = fig.add_subplot(2, 3, 3)
ax.plot(R0_arr, 'g-o', ms=4, lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2, label='R0=1')
ax.fill_between(range(n_months), 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red')
ax.set_xlabel('Month')
ax.set_ylabel('R0(t)')
ax.set_title('Estimated R0 from Transmission Potential')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 传播势能 vs 病例 (滞后)
ax = fig.add_subplot(2, 3, 4)
if best_lag > 0:
    ax.scatter(P_arr[:-best_lag], monthly_cases[best_lag:], alpha=0.6, s=50)
else:
    ax.scatter(P_arr, monthly_cases, alpha=0.6, s=50)
ax.set_xlabel(f'Transmission Potential P(t-{best_lag})')
ax.set_ylabel('Cases')
ax.set_title(f'P vs Cases (lag={best_lag}, r={best_corr:.3f})')
ax.grid(True, alpha=0.3)

# 5. 散点图
ax = fig.add_subplot(2, 3, 5)
ax.scatter(y[:split], y_pred_all[:split], c='blue', s=50, alpha=0.6, label='Train')
ax.scatter(y[split:], y_pred_all[split:], c='red', s=50, alpha=0.6, label='Test')
max_val = max(y.max(), y_pred_all.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 温度 vs 传播势能
ax = fig.add_subplot(2, 3, 6)
ax2 = ax.twinx()
ax.plot(monthly_temp, 'orange', lw=1.5, label='Temperature')
ax2.plot(P_arr, 'purple', lw=1.5, label='P(T)')
ax.set_xlabel('Month')
ax.set_ylabel('Temperature (°C)', color='orange')
ax2.set_ylabel('Transmission Potential', color='purple')
ax.set_title('Temperature vs Transmission Potential')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/practical_model.png', dpi=150, bbox_inches='tight')
plt.close()

log("  已保存: results/figures/practical_model.png")

# 保存数据
results_df = pd.DataFrame({
    'month': range(n_months),
    'year': df['year'].values,
    'temperature': monthly_temp,
    'cases': monthly_cases,
    'transmission_potential': P_arr,
    'R0_estimated': R0_arr
})
results_df.to_csv('/root/wenmei/results/data/practical_model.csv', index=False)
log("  已保存: results/data/practical_model.csv")

# 保存传播势能公式供符号回归
formula_df = pd.DataFrame({
    'temperature': monthly_temp,
    'P_T': P_arr,
    'cases': monthly_cases
})
formula_df.to_csv('/root/wenmei/results/data/transmission_potential_for_symbolic.csv', index=False)
log("  已保存: results/data/transmission_potential_for_symbolic.csv")

# ============================================================
# 8. 总结
# ============================================================
log("\n" + "=" * 65)
log("模型总结")
log("=" * 65)
log(f"""
【方法】
  温度依赖传播势能 + 滞后特征回归

【传播势能公式 (来自文献)】
  P(T) = a(T)² × b(T) × c(T) / μ_m(T)
  
  其中:
    a(T) = 0.0005 × T × (T-14) × √(35-T)  [叮咬率]
    b(T) = 0.0008 × T × (T-17) × √(36-T)  [蚊→人概率]
    c(T) = 0.0007 × T × (T-12) × √(37-T)  [人→蚊概率]
    μ_m(T) = 0.0006T² - 0.028T + 0.37     [死亡率]

【R0 估计】
  R0(t) = k × P(T)
  校准后 R0 范围: [{R0_arr.min():.3f}, {R0_arr.max():.3f}]

【传播势能-病例相关性】
  最佳滞后: {best_lag} 月
  相关系数: {best_corr:.4f}

【预测模型性能】
  训练集 r = {corr_train:.4f}, R² = {r2_train:.4f}
  测试集 r = {corr_test:.4f}, R² = {r2_test:.4f}

【下一步: 符号回归】
  输入: transmission_potential_for_symbolic.csv
  目标: 简化P(T)公式或找到更好的表达式
""")

log("\n完成!")
