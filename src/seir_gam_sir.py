"""
基于PNAS论文的气候驱动登革热动力学模型
参考: Li et al. (2019) PNAS - Climate-driven variation in mosquito density predicts dengue dynamics

核心思路:
1. GAM/TCN 预测蚊虫密度 M(t) — 统计模型
2. SIR/SEIR 模拟人群动态 — 动力学模型
3. 传播率 β(t) = β' × M(t) — 耦合机制

生物学参数从文献获取并固定，只估计向量效率 β'
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 生物学参数 (来自文献，固定不变)
# ============================================================
class BiologicalParameters:
    """
    登革热生物学参数 - 来自文献
    
    References:
    - Mordecai et al. (2017) PLOS NTD: Temperature effects on mosquito-borne diseases
    - Chan & Johansson (2012): Seasonality of dengue
    - Li et al. (2019) PNAS
    """
    
    # 人群参数
    INFECTIOUS_PERIOD_HUMAN = 5.0       # 人感染期 (天), 范围4-7天
    INTRINSIC_INCUBATION = 5.5          # 人潜伏期 (天), 范围4-7天
    
    # 蚊虫参数  
    EXTRINSIC_INCUBATION_25C = 10.0     # 外潜伏期@25°C (天)
    MOSQUITO_LIFESPAN = 14.0            # 蚊虫寿命 (天), 范围10-20天
    BITING_RATE = 0.5                   # 叮咬率 (次/天)
    
    # 传播概率
    PROB_MOSQUITO_TO_HUMAN = 0.5        # 蚊→人传播概率
    PROB_HUMAN_TO_MOSQUITO = 0.5        # 人→蚊传播概率
    
    @staticmethod
    def get_EIP(temperature):
        """
        温度依赖的外潜伏期 (天)
        基于 Mordecai et al. (2017) 的经验公式
        """
        if temperature < 18 or temperature > 34:
            return 30  # 极端温度下延长
        # 简化的温度-EIP关系
        EIP = 4 + 1.09 * np.exp(-(temperature - 25) / 10)
        return max(4, min(EIP * 2, 30))
    
    @staticmethod
    def get_mosquito_mortality(temperature):
        """
        温度依赖的蚊虫死亡率 (1/天)
        """
        optimal_temp = 25
        # 偏离最适温度时死亡率增加
        mortality = 0.05 + 0.01 * ((temperature - optimal_temp) / 10) ** 2
        return min(0.3, max(0.03, mortality))


# ============================================================
# 2. SIR 动力学模型 (简化版，参考PNAS)
# ============================================================
class DengueSIRModel:
    """
    登革热SIR模型
    
    dS/dt = -β(t) * S * I / N
    dI/dt = β(t) * S * I / N - γ * I
    dR/dt = γ * I
    
    其中 β(t) = β' × M(t)
    - β': 向量效率 (待估计)
    - M(t): 蚊虫密度 (从GAM/TCN预测)
    """
    
    def __init__(self, N, gamma=None):
        """
        Args:
            N: 人口总数
            gamma: 恢复率 (1/感染期), 默认从文献取值
        """
        self.N = N
        self.gamma = gamma or 1.0 / BiologicalParameters.INFECTIOUS_PERIOD_HUMAN
        
    def equations(self, t, y, beta_func):
        """SIR微分方程"""
        S, I, R = y
        N = self.N
        
        beta_t = beta_func(t)
        
        dSdt = -beta_t * S * I / N
        dIdt = beta_t * S * I / N - self.gamma * I
        dRdt = self.gamma * I
        
        return [dSdt, dIdt, dRdt]
    
    def solve(self, y0, t_span, t_eval, beta_func):
        """求解ODE"""
        sol = solve_ivp(
            self.equations,
            t_span,
            y0,
            args=(beta_func,),
            t_eval=t_eval,
            method='RK45'
        )
        return sol
    
    def compute_R0(self, beta):
        """计算基本再生数 R0 = β/γ"""
        return beta / self.gamma


# ============================================================
# 3. SEIR 动力学模型 (完整版)
# ============================================================
class DengueSEIRModel:
    """
    登革热SEIR模型 (只针对人群，蚊虫密度作为外部驱动)
    
    dS/dt = -λ(t) * S
    dE/dt = λ(t) * S - σ * E
    dI/dt = σ * E - γ * I
    dR/dt = γ * I
    
    感染力 λ(t) = β' × M(t) × I / N
    - β': 向量效率 (待估计的唯一参数)
    - M(t): 蚊虫密度指数 (从BI数据或GAM预测)
    """
    
    def __init__(self, N, sigma=None, gamma=None):
        """
        Args:
            N: 人口总数
            sigma: 暴露→感染率 (1/潜伏期)
            gamma: 恢复率 (1/感染期)
        """
        self.N = N
        self.sigma = sigma or 1.0 / BiologicalParameters.INTRINSIC_INCUBATION
        self.gamma = gamma or 1.0 / BiologicalParameters.INFECTIOUS_PERIOD_HUMAN
        
    def equations(self, t, y, beta_prime, M_func):
        """SEIR微分方程"""
        S, E, I, R = y
        N = self.N
        
        # 蚊虫密度 (从插值函数获取)
        M_t = M_func(t)
        
        # 感染力: λ(t) = β' × M(t) × I / N
        lambda_t = beta_prime * M_t * I / N
        
        dSdt = -lambda_t * S
        dEdt = lambda_t * S - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        
        return [dSdt, dEdt, dIdt, dRdt]
    
    def solve(self, y0, t_span, t_eval, beta_prime, M_func):
        """求解ODE"""
        sol = solve_ivp(
            self.equations,
            t_span,
            y0,
            args=(beta_prime, M_func),
            t_eval=t_eval,
            method='RK45',
            max_step=1.0
        )
        return sol
    
    def get_new_cases(self, sol, beta_prime, M_func):
        """
        计算新增病例 (每个时间步的新感染)
        新增 = σ × E (从E到I的流量)
        """
        E = sol.y[1]
        new_cases = self.sigma * E
        return new_cases
    
    def compute_Rt(self, beta_prime, M_t):
        """
        计算时变再生数 R(t)
        R(t) = β' × M(t) / γ
        """
        return beta_prime * M_t / self.gamma


# ============================================================
# 4. 参数估计 (只估计向量效率 β')
# ============================================================
class ParameterEstimator:
    """
    参数估计器
    只估计向量效率 β'，其他生物学参数固定
    """
    
    def __init__(self, model, observed_cases, M_data, time_points):
        """
        Args:
            model: SEIR模型实例
            observed_cases: 观测病例数 (数组)
            M_data: 蚊虫密度数据 (数组，与time_points对应)
            time_points: 时间点 (天)
        """
        self.model = model
        self.observed = observed_cases
        self.M_data = M_data
        self.time_points = time_points
        
        # 创建蚊虫密度插值函数
        from scipy.interpolate import interp1d
        self.M_func = interp1d(
            time_points, M_data, 
            kind='linear', 
            fill_value='extrapolate'
        )
    
    def objective(self, params):
        """
        目标函数: 最小化预测与观测的差异
        
        Args:
            params: [beta_prime, I0_fraction]
                - beta_prime: 向量效率
                - I0_fraction: 初始感染比例
        """
        beta_prime, I0_frac = params
        
        N = self.model.N
        I0 = max(1, N * I0_frac)
        E0 = I0 * 2  # 假设E0是I0的2倍
        R0 = 0
        S0 = N - E0 - I0 - R0
        
        y0 = [S0, E0, I0, R0]
        t_span = (self.time_points[0], self.time_points[-1])
        
        try:
            sol = self.model.solve(y0, t_span, self.time_points, beta_prime, self.M_func)
            
            if sol.status != 0:
                return 1e10
            
            # 计算累积病例变化 (近似新增)
            predicted_cumulative = sol.y[3]  # R compartment
            predicted_new = np.diff(predicted_cumulative, prepend=0)
            predicted_new = np.maximum(predicted_new, 0)
            
            # 使用对数空间的MSE
            obs_log = np.log1p(self.observed)
            pred_log = np.log1p(predicted_new)
            
            mse = np.mean((obs_log - pred_log) ** 2)
            return mse
            
        except Exception as e:
            return 1e10
    
    def estimate(self, method='differential_evolution'):
        """
        估计参数
        
        Returns:
            dict: 包含 beta_prime 和 I0_fraction
        """
        # 参数边界
        bounds = [
            (0.001, 1.0),    # beta_prime
            (1e-8, 1e-4),    # I0_fraction
        ]
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-6,
                workers=1,
                updating='deferred'
            )
        else:
            from scipy.optimize import minimize
            x0 = [0.1, 1e-6]
            result = minimize(
                self.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
        
        return {
            'beta_prime': result.x[0],
            'I0_fraction': result.x[1],
            'loss': result.fun
        }


# ============================================================
# 5. 完整工作流
# ============================================================
def run_analysis():
    """
    运行完整分析流程
    """
    print("=" * 70)
    print("气候驱动的登革热动力学模型")
    print("参考: Li et al. (2019) PNAS")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
    case_df['date'] = pd.to_datetime(case_df['date'])
    
    bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    
    # 2. 选择分析时段
    # 方案A: 排除2014年，用2006-2013年训练，2015-2019年测试
    # 方案B: 只用2015-2019年
    
    print("\n分析方案: 2015-2019年 (排除2014年异常暴发)")
    
    # 合并数据
    df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()
    df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
    
    # BI数据可能不完整，用气象数据估计
    df['bi'] = df['bi'].fillna(df['bi'].mean())
    
    # 如果BI数据缺失太多，用温度估计
    if df['bi'].isna().sum() > len(df) * 0.5:
        print("  警告: BI数据缺失，使用温度估计蚊虫密度")
        df['bi'] = estimate_mosquito_density(df['temperature'].values)
    
    print(f"  时间范围: 2015-2019年")
    print(f"  数据点: {len(df)}个月")
    print(f"  总病例: {df['cases'].sum():,}")
    print(f"  年度分布:")
    for year in range(2015, 2020):
        year_cases = df[df['year'] == year]['cases'].sum()
        print(f"    {year}: {year_cases:,} 例")
    
    # 3. 准备模型输入
    print("\n[2] 准备模型...")
    
    N = 14_000_000  # 广州人口
    
    # 时间点 (天)
    n_months = len(df)
    days_per_month = 30
    time_days = np.arange(n_months) * days_per_month
    
    # 蚊虫密度 (标准化)
    M_raw = df['bi'].values
    M_normalized = (M_raw - M_raw.min()) / (M_raw.max() - M_raw.min() + 1e-6)
    M_normalized = M_normalized + 0.01  # 避免零值
    
    # 观测病例
    observed_cases = df['cases'].values
    
    print(f"  人口: {N:,}")
    print(f"  蚊虫密度范围: {M_raw.min():.2f} - {M_raw.max():.2f}")
    
    # 4. 创建SEIR模型
    print("\n[3] 参数估计...")
    print("  固定参数 (来自文献):")
    print(f"    σ (暴露→感染): {1/BiologicalParameters.INTRINSIC_INCUBATION:.3f}/天")
    print(f"    γ (恢复率): {1/BiologicalParameters.INFECTIOUS_PERIOD_HUMAN:.3f}/天")
    print("  待估计参数:")
    print("    β' (向量效率)")
    
    model = DengueSEIRModel(N)
    
    # 5. 估计参数
    estimator = ParameterEstimator(
        model, 
        observed_cases, 
        M_normalized, 
        time_days
    )
    
    params = estimator.estimate()
    beta_prime = params['beta_prime']
    I0_frac = params['I0_fraction']
    
    print(f"\n  估计结果:")
    print(f"    β' = {beta_prime:.4f}")
    print(f"    I0 = {N * I0_frac:.1f}")
    print(f"    Loss = {params['loss']:.4f}")
    
    # 6. 模型模拟
    print("\n[4] 模型模拟...")
    
    I0 = max(1, N * I0_frac)
    E0 = I0 * 2
    S0 = N - E0 - I0
    y0 = [S0, E0, I0, 0]
    
    from scipy.interpolate import interp1d
    M_func = interp1d(time_days, M_normalized, kind='linear', fill_value='extrapolate')
    
    sol = model.solve(y0, (time_days[0], time_days[-1]), time_days, beta_prime, M_func)
    
    # 新增病例
    predicted_R = sol.y[3]
    predicted_new = np.diff(predicted_R, prepend=0)
    predicted_new = np.maximum(predicted_new, 0)
    
    # 7. 计算R(t)
    Rt = np.array([model.compute_Rt(beta_prime, M_func(t)) for t in time_days])
    
    print(f"  R(t) 范围: {Rt.min():.2f} - {Rt.max():.2f}")
    print(f"  R(t) > 1 的月份: {(Rt > 1).sum()}/{len(Rt)}")
    
    # 8. 评估
    print("\n[5] 模型评估...")
    
    # 相关性
    corr, pval = pearsonr(observed_cases, predicted_new)
    
    # 对数空间R²
    obs_log = np.log1p(observed_cases)
    pred_log = np.log1p(predicted_new)
    r2_log = r2_score(obs_log, pred_log)
    
    # 线性R²
    r2_linear = r2_score(observed_cases, predicted_new)
    
    print(f"  相关系数: {corr:.4f} (p={pval:.4f})")
    print(f"  R² (对数): {r2_log:.4f}")
    print(f"  R² (线性): {r2_linear:.4f}")
    
    # 9. 可视化
    print("\n[6] 生成可视化...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 时间标签
    time_labels = [f"{row['year']}-{row['month']:02d}" for _, row in df.iterrows()]
    x_ticks = range(0, len(time_labels), 6)
    x_labels = [time_labels[i] for i in x_ticks]
    
    # 1. 病例对比 (线性)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(observed_cases, 'b-o', lw=2, ms=4, label='Observed')
    ax1.plot(predicted_new, 'r-s', lw=2, ms=4, label='SEIR Predicted')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Cases')
    ax1.set_title(f'Monthly Cases (r={corr:.3f})')
    ax1.legend()
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. 病例对比 (对数)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(observed_cases + 1, 'b-o', lw=2, ms=4, label='Observed')
    ax2.semilogy(predicted_new + 1, 'r-s', lw=2, ms=4, label='SEIR Predicted')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Cases (log scale)')
    ax2.set_title(f'Cases (Log Scale, R²={r2_log:.3f})')
    ax2.legend()
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. R(t) 时间序列
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(Rt, 'g-o', lw=2, ms=4)
    ax3.axhline(y=1, color='red', ls='--', lw=2, label='R(t)=1')
    ax3.fill_between(range(len(Rt)), 0, Rt, where=Rt>1, alpha=0.3, color='red', label='R(t)>1')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('R(t)')
    ax3.set_title('Time-varying Reproduction Number')
    ax3.legend()
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 散点图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(observed_cases, predicted_new, c='blue', s=50, alpha=0.6)
    max_val = max(observed_cases.max(), predicted_new.max())
    ax4.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect fit')
    ax4.set_xlabel('Observed Cases')
    ax4.set_ylabel('Predicted Cases')
    ax4.set_title('Scatter Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 蚊虫密度 vs 病例
    ax5 = fig.add_subplot(2, 3, 5)
    ax5_twin = ax5.twinx()
    ln1 = ax5.plot(M_raw, 'g-o', lw=2, ms=4, label='BI (Mosquito)')
    ln2 = ax5_twin.plot(observed_cases, 'b-s', lw=2, ms=4, label='Cases')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Breteau Index', color='green')
    ax5_twin.set_ylabel('Cases', color='blue')
    ax5.set_title('Mosquito Density vs Cases')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='upper left')
    ax5.set_xticks(x_ticks)
    ax5.set_xticklabels(x_labels, rotation=45)
    
    # 6. 年度汇总
    ax6 = fig.add_subplot(2, 3, 6)
    df['predicted'] = predicted_new
    yearly = df.groupby('year').agg({'cases': 'sum', 'predicted': 'sum'}).reset_index()
    x = range(len(yearly))
    width = 0.35
    ax6.bar([i - width/2 for i in x], yearly['cases'], width, label='Observed', color='steelblue')
    ax6.bar([i + width/2 for i in x], yearly['predicted'], width, label='Predicted', color='coral')
    ax6.set_xticks(x)
    ax6.set_xticklabels(yearly['year'])
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Annual Cases')
    ax6.set_title('Annual Cases Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/figures/seir_dynamics_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  已保存: results/figures/seir_dynamics_model.png")
    
    # 10. 保存结果
    results_df = pd.DataFrame({
        'year': df['year'].values,
        'month': df['month'].values,
        'observed_cases': observed_cases,
        'predicted_cases': predicted_new,
        'bi': M_raw,
        'Rt': Rt
    })
    results_df.to_csv('/root/wenmei/results/data/seir_dynamics_results.csv', index=False)
    print("  已保存: results/data/seir_dynamics_results.csv")
    
    # 打印模型总结
    print("\n" + "=" * 70)
    print("模型总结")
    print("=" * 70)
    print(f"""
模型类型: SEIR (人群动力学)
驱动变量: 蚊虫密度 M(t) — 来自布雷图指数

微分方程组:
  dS/dt = -λ(t) × S
  dE/dt = λ(t) × S - σ × E  
  dI/dt = σ × E - γ × I
  dR/dt = γ × I
  
  其中 λ(t) = β' × M(t) × I / N

固定参数 (来自文献):
  σ = {model.sigma:.3f} /天 (1/潜伏期)
  γ = {model.gamma:.3f} /天 (1/感染期)
  
估计参数:
  β' = {beta_prime:.4f} (向量效率)

模型性能:
  相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  R(t) 范围: {Rt.min():.2f} - {Rt.max():.2f}
""")
    
    return {
        'beta_prime': beta_prime,
        'correlation': corr,
        'r2_log': r2_log,
        'Rt': Rt,
        'observed': observed_cases,
        'predicted': predicted_new
    }


def estimate_mosquito_density(temperature):
    """
    用温度估计蚊虫密度 (当BI数据缺失时)
    基于温度适宜性
    """
    suitability = np.exp(-((temperature - 27) / 8) ** 2)
    return suitability * 10  # 缩放到合理范围


if __name__ == '__main__':
    results = run_analysis()
