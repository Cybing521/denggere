# SEI-SEIR 蚊媒-人群动力学模型

基于TCN神经网络和SEI-SEIR微分方程的登革热传播动力学模型，使用广东省真实数据进行训练和验证。

## 模型概述

本项目实现了一个**蚊媒-人群耦合动力学模型**，用于模拟和预测登革热的传播动态：

- **蚊媒部分 (SEI)**：易感(S) → 暴露(E) → 感染(I)
- **人群部分 (SEIR)**：易感(S) → 暴露(E) → 感染(I) → 康复(R)

### 核心创新

1. 使用 **TCN (Temporal Convolutional Network)** 从气象数据预测蚊虫承载力 Λ_v(t)
2. **两阶段建模**：TCN预测蚊虫动态 → SEI-SEIR预测病例
3. 使用**广东省真实病例数据**进行模型验证

## 数据来源

| 数据类型 | 来源 | 时间范围 | 时间尺度 |
|---------|------|---------|---------|
| 登革热病例 | [CCM14 Dataset](https://github.com/xyyu001/CCM14) | 2005-2019年 | 月度 |
| 布雷图指数(BI) | CCM14 Dataset | 2006-2014年 | 月度 |
| 气象数据 | [Open-Meteo API](https://open-meteo.com/) | 2006-2014年 | 月度 |

### 广东省病例数据概览

- **总病例数**: 63,510例 (2005-2019年)
- **分析时段**: 2006-2014年, 50,258例
- **2014年暴发**: 45,189例 (占9年总数的90%)

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
│   ├── BI.csv                           # CCM14布雷图指数数据集
│   ├── data_Guangdong.xlsx              # 广东省病例+气象原始数据
│   ├── guangdong_dengue_cases.csv       # 处理后的病例数据
│   └── real/                            # 气象数据
│       ├── guangzhou_weather_real.csv
│       ├── guangzhou_bi.csv
│       └── guangzhou_merged_real.csv
├── src/
│   ├── sei_seir_model.py                # SEI-SEIR微分方程模型
│   ├── tcn_model.py                     # TCN神经网络
│   ├── fetch_real_data.py               # 气象数据获取
│   ├── train_with_real_data.py          # BI预测训练
│   ├── validate_with_cases.py           # 病例数据验证
│   ├── two_stage_model.py               # 两阶段模型(TCN+SEIR)
│   ├── two_stage_model_v2.py            # 改进版两阶段模型
│   └── hybrid_model.py                  # 混合模型
├── results/
│   ├── figures/                         # 可视化图片
│   │   ├── real_data_results.png        # BI预测结果
│   │   ├── xgboost_case_prediction.png  # 病例预测结果 ⭐
│   │   └── comprehensive_case_validation.png
│   └── data/                            # 预测数据
│       ├── real_data_predictions.csv    # BI预测数据
│       ├── xgboost_predictions.csv      # 病例预测数据
│       └── tcn_model_real_data.pt       # TCN模型权重
├── paper/                               # 参考文献
├── report/
│   ├── sei_seir_report.tex              # LaTeX源文件
│   └── sei_seir_report.pdf              # PDF报告
├── requirements.txt
└── README.md
```

## 模型性能

### 1. TCN预测布雷图指数 (BI)

使用广州2006-2014年气象数据预测BI：

| 指标 | 值 |
|------|-----|
| R² | **0.988** |
| 相关系数 | 0.997 |
| MAE | 0.53 |
| RMSE | 0.80 |

### 2. 病例数预测 (XGBoost + SEI-SEIR特征)

使用混合模型预测月度病例数：

| 指标 | 训练集 | 测试集 | 全量 |
|------|--------|--------|------|
| R² (线性) | 0.994 | -0.115 | -0.001 |
| R² (对数) | - | - | **0.821** |
| 相关系数 | 0.998 | 0.746 | 0.468 |

### 3. 关键发现

1. **环境适宜性领先病例2个月** - `env_suit_lag2`是最重要特征 (43.9%)
2. **季节性明显** - 高风险期为6-10月
3. **2014年暴发极端** - 单年病例占9年的90%，难以用常规模型预测

## 特征重要性 (病例预测)

| 排名 | 特征 | 重要性 | 含义 |
|------|------|--------|------|
| 1 | env_suit_lag2 | 43.9% | 2个月前环境适宜性 |
| 2 | month_sin | 18.8% | 季节周期 |
| 3 | cases_lag1 | 11.6% | 上月病例数 |
| 4 | cases_ma3 | 10.9% | 3月移动平均 |
| 5 | bi_ma3 | 3.8% | BI移动平均 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 获取气象数据

```bash
python src/fetch_real_data.py
```

### 2. 训练BI预测模型

```bash
python src/train_with_real_data.py
```

### 3. 训练病例预测模型

```bash
cd src && python hybrid_model.py
```

或使用XGBoost：

```python
import xgboost as xgb
# 详见 src/hybrid_model.py
```

### 4. 在代码中使用

```python
from src.sei_seir_model import SEISEIRModel, SEISEIRParameters

# 创建模型参数
params = SEISEIRParameters(
    N_h=110_000_000,   # 广东人口
    mu_v=1/14,         # 蚊虫死亡率 (寿命14天)
    beta_v=0.5,        # 人→蚊传播概率
    beta_h=0.5,        # 蚊→人传播概率
    sigma_v=0.1,       # 蚊虫EIP转化率 (10天)
    sigma_h=0.2,       # 人潜伏期转化率 (5天)
    gamma=0.143        # 康复率 (7天)
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
| 蚊虫出生率 | Λ_v(t) | TCN预测 | 气象驱动 |
| 蚊虫死亡率 | μ_v | 1/14 day⁻¹ | 约14天寿命 |
| 人→蚊传播概率 | β_v | 0.5 | 文献值 |
| 蚊→人传播概率 | β_h | 0.5 | 文献值 |
| 叮咬率 | b | 0.5 day⁻¹ | 文献值 |
| 蚊虫EIP | 1/σ_v | 10天 | 外潜伏期 |
| 人潜伏期 | 1/σ_h | 5天 | 内潜伏期 |
| 康复期 | 1/γ | 7天 | 感染期 |

## 基本再生数 R₀

$$R_0 = \sqrt{\frac{b^2 \beta_h \beta_v m \sigma_v}{(\sigma_v + \mu_v) \mu_v \gamma}}$$

广州地区R₀统计：
- 均值: 0.45
- 最大: 0.76 (2008年7月)
- R₀ > 1 比例: 0%

## 模型局限性

1. **2014年暴发难预测** - 极端事件占比过大
2. **R₀始终<1** - 可能低估传播风险
3. **月度数据粒度粗** - 无法捕捉周内变化
4. **未考虑输入性病例** - 广东作为口岸省份

## 下一步改进方向

1. 获取**周度/日度**病例数据
2. 构建**暴发风险预警系统**
3. 引入**输入性病例**参数
4. 使用**PINN**嵌入物理约束

## 参考文献

1. Mordecai EA, et al. (2017). Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya. *PLOS NTD*.

2. Yang HM, et al. (2009). Assessing the effects of temperature on Aedes aegypti. *Epidemiology & Infection*.

3. CCM14 Dataset: https://github.com/xyyu001/CCM14

4. WS/T 784—2021. 登革热病媒生物应急监测与控制标准.

## License

MIT License
