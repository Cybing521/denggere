# SEI-SEIR 蚊媒-人群动力学模型

基于TCN神经网络和SEI-SEIR微分方程的登革热传播动力学模型。

## 模型概述

本项目实现了一个**蚊媒-人群耦合动力学模型**，用于模拟和预测登革热的传播动态：

- **蚊媒部分 (SEI)**：易感(S) → 暴露(E) → 感染(I)
- **人群部分 (SEIR)**：易感(S) → 暴露(E) → 感染(I) → 康复(R)

### 核心创新

使用 **TCN (Temporal Convolutional Network)** 从气象数据预测蚊虫承载力 Λ_v(t)，实现气象驱动的动力学模型。

## 微分方程组

### 蚊媒动力学 (SEI)

$$\frac{dS_v}{dt} = \Lambda_v(t) - \mu_v S_v - \beta_v b \frac{I_h}{N_h} S_v$$

$$\frac{dE_v}{dt} = \beta_v b \frac{I_h}{N_h} S_v - (\mu_v + \sigma_v) E_v$$

$$\frac{dI_v}{dt} = \sigma_v E_v - \mu_v I_v$$

### 人群动力学 (SEIR)

$$\frac{dS_h}{dt} = -\beta_h b \frac{I_v}{N_h} S_h$$

$$\frac{dE_h}{dt} = \beta_h b \frac{I_v}{N_h} S_h - \sigma_h E_h$$

$$\frac{dI_h}{dt} = \sigma_h E_h - \gamma I_h$$

$$\frac{dR_h}{dt} = \gamma I_h$$

## 项目结构

```
wenmei/
├── data/
│   ├── BI.csv                         # CCM14布雷图指数数据集
│   └── real/                          # 真实数据
│       ├── guangzhou_weather_real.csv # 广州气象数据
│       ├── guangzhou_bi.csv           # 广州BI数据
│       └── guangzhou_merged_real.csv  # 合并数据
├── src/
│   ├── sei_seir_model.py              # SEI-SEIR微分方程模型
│   ├── tcn_model.py                   # TCN神经网络
│   ├── fetch_real_data.py             # 真实数据获取脚本
│   └── train_with_real_data.py        # 训练脚本
├── results/
│   ├── real_data_results.png          # 可视化结果
│   ├── real_data_predictions.csv      # 预测结果
│   └── tcn_model_real_data.pt         # 训练好的模型
├── paper/                             # 参考文献
├── report/
│   ├── sei_seir_report.tex            # LaTeX源文件
│   └── sei_seir_report.pdf            # PDF报告
├── requirements.txt                   # 依赖包
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 获取真实数据

```bash
python src/fetch_real_data.py
```

从 Open-Meteo API 获取广州历史气象数据，并与CCM14数据集的布雷图指数数据合并。

### 2. 训练模型

```bash
python src/train_with_real_data.py
```

使用TCN网络从气象数据预测BI，并运行SEI-SEIR模拟。

### 3. 在代码中使用

```python
from src.sei_seir_model import SEISEIRModel, SEISEIRParameters
from src.tcn_model import MosquitoTCNPredictor

# 创建模型参数
params = SEISEIRParameters(
    N_h=14_000_000,    # 广州人口
    mu_v=0.05,         # 蚊虫死亡率
    beta_v=0.5,        # 人→蚊传播概率
    beta_h=0.75,       # 蚊→人传播概率
    sigma_v=0.1,       # 蚊虫潜伏期转化率
    sigma_h=0.2,       # 人潜伏期转化率
    gamma=0.143        # 康复率
)

# 创建模型
model = SEISEIRModel(params, lambda_v_func=lambda t: 1000)

# 运行模拟
y0 = model.get_initial_conditions(N_v0=10000, I_h0=10)
result = model.solve(y0, t_span=(0, 365))
```

## 模型参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 蚊虫出生率 | Λ_v(t) | TCN预测 | 气象驱动，由TCN网络预测 |
| 蚊虫死亡率 | μ_v | 0.05 day⁻¹ | 约20天寿命 |
| 人→蚊传播概率 | β_v | 0.5 | 文献值 |
| 蚊→人传播概率 | β_h | 0.75 | 文献值 |
| 叮咬率 | b | 0.5 day⁻¹ | 文献值 |
| 蚊虫潜伏期转化率 | σ_v | 0.1 day⁻¹ | EIP约10天 |
| 人潜伏期转化率 | σ_h | 0.2 day⁻¹ | 约5天 |
| 康复率 | γ | 0.143 day⁻¹ | 约7天 |

## 数据来源

- **气象数据**: [Open-Meteo API](https://open-meteo.com/) - 免费历史气象数据
- **布雷图指数**: [CCM14 Dataset](https://github.com/xyyu001/CCM14) - 中国蚊媒监测数据
- **标准规范**: WS/T 784—2021《登革热病媒生物应急监测与控制标准》

## 模型性能

使用广州2006-2014年真实数据训练：

| 指标 | 值 |
|------|-----|
| R² | 0.988 |
| 相关系数 | 0.997 |
| MAE | 0.53 |
| RMSE | 0.80 |

## 基本再生数 R₀

$$R_0 = \sqrt{R_{0,vh} \times R_{0,hv}}$$

其中：
- $R_{0,vh}$ = 蚊→人传播潜力
- $R_{0,hv}$ = 人→蚊传播潜力

模型计算的广州R₀：
- 均值: 0.45
- 最大: 0.76
- 最小: 0.09

## 参考文献

1. Mordecai EA, et al. (2017). Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models. *PLOS Neglected Tropical Diseases*.

2. Yang HM, et al. (2009). Assessing the effects of temperature on the population of Aedes aegypti, the vector of dengue. *Epidemiology & Infection*.

3. WS/T 784—2021. 登革热病媒生物应急监测与控制标准. 中华人民共和国卫生行业标准.

## License

MIT License
