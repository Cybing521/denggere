"""
SEI-SEIR 蚊媒-人群动力学模型

蚊媒部分 (SEI):
    dS_v/dt = Λ_v(t) - μ_v*S_v - β_v*b*(I_h/N_h)*S_v
    dE_v/dt = β_v*b*(I_h/N_h)*S_v - (μ_v + σ_v)*E_v
    dI_v/dt = σ_v*E_v - μ_v*I_v

人群部分 (SEIR):
    dS_h/dt = -β_h*b*(I_v/N_h)*S_h
    dE_h/dt = β_h*b*(I_v/N_h)*S_h - σ_h*E_h
    dI_h/dt = σ_h*E_h - γ*I_h
    dR_h/dt = γ*I_h

参数说明:
    Λ_v(t): 蚊虫出生率 (气象驱动, 由TCN预测)
    μ_v: 蚊虫死亡率 (温度依赖)
    β_v: 人→蚊传播概率
    β_h: 蚊→人传播概率
    b: 叮咬率
    σ_v: 蚊虫潜伏期转化率 (1/EIP)
    σ_h: 人潜伏期转化率
    γ: 人康复率
    N_h: 总人口数
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Tuple, Optional


@dataclass
class SEISEIRParameters:
    """SEI-SEIR模型参数"""
    # 蚊媒参数
    mu_v: float = 0.1          # 蚊虫死亡率 (day^-1), 约10天寿命
    beta_v: float = 0.33       # 人→蚊传播概率
    sigma_v: float = 0.1       # 蚊虫潜伏期转化率 (day^-1), EIP约10天
    b: float = 0.5             # 叮咬率 (bites/day)
    
    # 人群参数
    beta_h: float = 0.5        # 蚊→人传播概率
    sigma_h: float = 0.2       # 人潜伏期转化率 (day^-1), 约5天
    gamma: float = 0.143       # 康复率 (day^-1), 约7天
    
    # 人口参数
    N_h: float = 1e6           # 总人口数
    
    def get_mu_v_temp_dependent(self, temperature: float) -> float:
        """
        温度依赖的蚊虫死亡率
        基于 Mordecai et al. (2017) 的公式
        """
        T = temperature
        if T < 10 or T > 40:
            return 1.0  # 极端温度下高死亡率
        # 简化的温度依赖关系
        mu_v = 0.8692 - 0.159*T + 0.01116*T**2 - 0.0003408*T**3 + 0.000003809*T**4
        return max(0.05, min(1.0, mu_v))
    
    def get_EIP_temp_dependent(self, temperature: float) -> float:
        """
        温度依赖的外潜伏期 (Extrinsic Incubation Period)
        """
        T = temperature
        if T < 18:
            return 20.0  # 低温时EIP很长
        elif T > 35:
            return 5.0   # 高温时EIP短但蚊子死亡率高
        # EIP随温度升高而缩短
        EIP = 4 + np.exp(5.15 - 0.123*T)
        return max(3.0, min(20.0, EIP))


class SEISEIRModel:
    """SEI-SEIR 蚊媒-人群动力学模型"""
    
    def __init__(self, params: SEISEIRParameters, 
                 lambda_v_func: Optional[Callable[[float], float]] = None):
        """
        初始化模型
        
        Args:
            params: 模型参数
            lambda_v_func: 蚊虫出生率函数 Λ_v(t), 如果为None则使用常数
        """
        self.params = params
        self.lambda_v_func = lambda_v_func or (lambda t: 1000.0)  # 默认常数
        
    def _equations(self, t: float, y: np.ndarray, 
                   temperature: Optional[float] = None) -> np.ndarray:
        """
        微分方程组
        
        状态变量 y = [S_v, E_v, I_v, S_h, E_h, I_h, R_h]
        """
        S_v, E_v, I_v, S_h, E_h, I_h, R_h = y
        p = self.params
        
        # 获取时变参数
        Lambda_v = self.lambda_v_func(t)
        
        # 温度依赖参数（如果提供温度）
        if temperature is not None:
            mu_v = p.get_mu_v_temp_dependent(temperature)
            sigma_v = 1.0 / p.get_EIP_temp_dependent(temperature)
        else:
            mu_v = p.mu_v
            sigma_v = p.sigma_v
        
        N_v = S_v + E_v + I_v
        
        # 蚊媒方程 (SEI)
        dS_v = Lambda_v - mu_v * S_v - p.beta_v * p.b * (I_h / p.N_h) * S_v
        dE_v = p.beta_v * p.b * (I_h / p.N_h) * S_v - (mu_v + sigma_v) * E_v
        dI_v = sigma_v * E_v - mu_v * I_v
        
        # 人群方程 (SEIR)
        dS_h = -p.beta_h * p.b * (I_v / p.N_h) * S_h
        dE_h = p.beta_h * p.b * (I_v / p.N_h) * S_h - p.sigma_h * E_h
        dI_h = p.sigma_h * E_h - p.gamma * I_h
        dR_h = p.gamma * I_h
        
        return np.array([dS_v, dE_v, dI_v, dS_h, dE_h, dI_h, dR_h])
    
    def solve(self, y0: np.ndarray, t_span: Tuple[float, float], 
              t_eval: Optional[np.ndarray] = None,
              temperature_series: Optional[np.ndarray] = None) -> dict:
        """
        求解微分方程组
        
        Args:
            y0: 初始状态 [S_v0, E_v0, I_v0, S_h0, E_h0, I_h0, R_h0]
            t_span: 时间范围 (t_start, t_end)
            t_eval: 输出时间点
            temperature_series: 温度序列（可选）
            
        Returns:
            包含时间和状态变量的字典
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        # 如果有温度序列，创建插值函数
        if temperature_series is not None:
            from scipy.interpolate import interp1d
            temp_func = interp1d(t_eval, temperature_series, 
                                bounds_error=False, fill_value="extrapolate")
            
            def equations_with_temp(t, y):
                temp = temp_func(t)
                return self._equations(t, y, temperature=temp)
            
            sol = solve_ivp(equations_with_temp, t_span, y0, 
                           t_eval=t_eval, method='RK45')
        else:
            sol = solve_ivp(lambda t, y: self._equations(t, y), 
                           t_span, y0, t_eval=t_eval, method='RK45')
        
        return {
            't': sol.t,
            'S_v': sol.y[0],
            'E_v': sol.y[1],
            'I_v': sol.y[2],
            'S_h': sol.y[3],
            'E_h': sol.y[4],
            'I_h': sol.y[5],
            'R_h': sol.y[6],
            'N_v': sol.y[0] + sol.y[1] + sol.y[2],  # 蚊虫总数
            'new_cases': np.diff(sol.y[5] + sol.y[6], prepend=sol.y[5][0] + sol.y[6][0])
        }
    
    def get_initial_conditions(self, N_v0: float = 10000, I_h0: float = 10) -> np.ndarray:
        """
        获取合理的初始条件
        
        Args:
            N_v0: 初始蚊虫总数
            I_h0: 初始感染人数
        """
        # 蚊虫初始状态
        S_v0 = N_v0 * 0.95
        E_v0 = N_v0 * 0.04
        I_v0 = N_v0 * 0.01
        
        # 人群初始状态
        S_h0 = self.params.N_h - I_h0
        E_h0 = 0
        R_h0 = 0
        
        return np.array([S_v0, E_v0, I_v0, S_h0, E_h0, I_h0, R_h0])
    
    def compute_R0(self, N_v: float, temperature: float = 25.0) -> float:
        """
        计算基本再生数 R0
        
        R0 = sqrt(R0_vh * R0_hv)
        其中:
        R0_vh = (β_h * b * σ_v * N_v) / (μ_v * (μ_v + σ_v) * N_h)  # 蚊→人
        R0_hv = (β_v * b) / γ  # 人→蚊
        """
        p = self.params
        mu_v = p.get_mu_v_temp_dependent(temperature)
        sigma_v = 1.0 / p.get_EIP_temp_dependent(temperature)
        
        # 蚊→人的传播潜力
        R0_vh = (p.beta_h * p.b * sigma_v * N_v) / (mu_v * (mu_v + sigma_v) * p.N_h)
        
        # 人→蚊的传播潜力  
        R0_hv = (p.beta_v * p.b) / p.gamma
        
        # 综合R0
        R0 = np.sqrt(R0_vh * R0_hv)
        
        return R0


def bi_to_mosquito_density(bi: float, base_density: float = 1000) -> float:
    """
    将布雷图指数(BI)转换为蚊虫密度估计
    
    BI定义：平均每百户住宅中有伊蚊幼虫孳生的积水容器数
    
    Args:
        bi: 布雷图指数
        base_density: 基础蚊虫密度参数
        
    Returns:
        估计的蚊虫承载力 Λ_v
    """
    # 假设BI与蚊虫承载力呈近似线性关系
    # 但有饱和效应
    lambda_v = base_density * (1 - np.exp(-0.05 * bi))
    return lambda_v


if __name__ == "__main__":
    # 测试模型
    import matplotlib.pyplot as plt
    
    # 创建参数
    params = SEISEIRParameters(N_h=1e6)
    
    # 创建时变蚊虫出生率函数（模拟季节性）
    def seasonal_lambda_v(t):
        # 模拟广州的季节性（夏季高、冬季低）
        base = 500
        amplitude = 400
        # 假设t=0是1月1日
        return base + amplitude * np.sin(2 * np.pi * (t - 90) / 365)
    
    # 创建模型
    model = SEISEIRModel(params, lambda_v_func=seasonal_lambda_v)
    
    # 初始条件
    y0 = model.get_initial_conditions(N_v0=10000, I_h0=10)
    
    # 模拟365天
    t_span = (0, 365)
    t_eval = np.linspace(0, 365, 365)
    
    # 求解
    result = model.solve(y0, t_span, t_eval)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 蚊虫动态
    axes[0, 0].plot(result['t'], result['S_v'], label='S_v (易感)')
    axes[0, 0].plot(result['t'], result['E_v'], label='E_v (暴露)')
    axes[0, 0].plot(result['t'], result['I_v'], label='I_v (感染)')
    axes[0, 0].set_xlabel('时间 (天)')
    axes[0, 0].set_ylabel('蚊虫数量')
    axes[0, 0].set_title('蚊媒动力学 (SEI)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 人群动态
    axes[0, 1].plot(result['t'], result['E_h'], label='E_h (暴露)')
    axes[0, 1].plot(result['t'], result['I_h'], label='I_h (感染)')
    axes[0, 1].plot(result['t'], result['R_h'], label='R_h (康复)')
    axes[0, 1].set_xlabel('时间 (天)')
    axes[0, 1].set_ylabel('人数')
    axes[0, 1].set_title('人群动力学 (SEIR)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 蚊虫总数
    axes[1, 0].plot(result['t'], result['N_v'])
    axes[1, 0].set_xlabel('时间 (天)')
    axes[1, 0].set_ylabel('蚊虫总数')
    axes[1, 0].set_title('蚊虫种群动态')
    axes[1, 0].grid(True)
    
    # Lambda_v 函数
    lambda_values = [seasonal_lambda_v(t) for t in result['t']]
    axes[1, 1].plot(result['t'], lambda_values)
    axes[1, 1].set_xlabel('时间 (天)')
    axes[1, 1].set_ylabel('Λ_v')
    axes[1, 1].set_title('蚊虫出生率 (季节性)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/sei_seir_test.png', dpi=150)
    plt.close()
    
    print("SEI-SEIR模型测试完成!")
    print(f"R0 (25°C): {model.compute_R0(N_v=10000, temperature=25):.2f}")
