# 毕业论文交付物清单

**论文题目**：基于神经网络耦合动力学模型的登革热传播效率发现与多城市验证

---

## 目录结构

```
deliverables/
├── thesis/                          # 论文
│   ├── thesis_draft.tex             # LaTeX 源文件
│   └── thesis_draft.pdf             # 编译后 PDF（36页）
│
├── figures/                         # 图片（每张均有 PNG + PDF 两个版本）
│   ├── main/                        # 正文引用的 6 张主图
│   │   ├── fig1_phase1_guangzhou     Phase1 广州耦合模型月度预测 vs 观测
│   │   ├── fig2_phase2_formula_fit   Phase2 符号回归公式 vs 神经网络输出
│   │   ├── fig3_outbreak_2014_beta   2014年 vs 其他年份月度 β' 对比
│   │   ├── fig4_transfer_2014_bars   2014年16城年度病例：观测 vs 预测
│   │   ├── fig5_all_cities_fit_grid  16城月度预测与观测曲线网格图
│   │   └── fig6_appendix_guangzhou   附录：广州市月度拟合曲线
│   │
│   ├── appendix_cities/             # 16城逐城市拟合曲线（附录用）
│   │   └── fit_{CityName}.png/pdf   每个城市的双面板图
│   │
│   └── paper_extra/                 # 补充图（描述性分析、敏感性等）
│       ├── A1–A12                   分析图（病例预测、β'对比、公式参数等）
│       └── D1–D14                   描述性图（疾病分布、气象时序、相关热图等）
│
├── scripts/                         # 核心 Python 脚本
│   ├── run_1plus3_on_data2.py       ★ 主流程：β'估计 → NN训练 → 符号回归 → 跨城市迁移
│   ├── prepare_data2_pipeline.py    数据预处理：原始数据 → 建模用月度/双周CSV
│   ├── fetch_real_data.py           从 Open-Meteo API 获取气象数据
│   ├── plot_per_city_fit_curves.py  绘制逐城市拟合曲线
│   ├── plot_other_cities_descriptive.py  绘制多城市描述性分析图
│   └── redraw_multicity_appendix_figs.py 重绘多城市附录图
│
├── data/
│   ├── input/                       # 原始输入数据
│   │   ├── guangdong_dengue_cases.csv    广东省登革热病例数据
│   │   ├── guangzhou_biweekly_cases.csv  广州双周病例数据
│   │   ├── BI.csv                        布雷图指数（蚊媒监测）
│   │   ├── data2_raw.csv                 第二数据集原始数据
│   │   ├── data2_BI.csv                  第二数据集 BI 数据
│   │   └── cities/                       13城月度数据
│   │       └── {CityName}_monthly.csv
│   │
│   ├── processed/                   # 预处理后数据
│   │   ├── cases_weather_monthly_utf8.csv   月度病例+气象合并数据
│   │   ├── cases_weather_biweekly_utf8.csv  双周病例+气象合并数据
│   │   ├── cases_weather_weekly_utf8.csv    周度病例+气象合并数据
│   │   ├── bi_guangdong_monthly_proxy.csv   广东月度 BI 代理指标
│   │   ├── bi_guangdong_monthly_by_method.csv  按方法分的 BI 数据
│   │   ├── bi_proxy_method_selection.csv    BI 代理方法选择
│   │   └── city_coverage_summary.csv        城市数据覆盖汇总
│   │
│   └── model_outputs/               # 模型输出结果
│       ├── phase1_guangzhou_predictions_data2.csv  Phase1 广州预测值
│       ├── phase1_metrics_data2.csv                Phase1 评估指标
│       ├── phase2_formula_params_data2.csv         Phase2 公式系数
│       ├── phase2_formula_fit_metrics_data2.csv    Phase2 拟合指标
│       ├── transfer_annual2014_data2.csv           16城2014年度迁移结果
│       ├── transfer_monthly_all_cities_data2.csv   16城月度迁移预测
│       ├── transfer_metrics_data2.csv              迁移评估指标
│       └── city_monthly_metrics_data2.csv          逐城市月度指标
│
└── models/                          # 训练好的模型权重
    ├── phase1_model.pt              Phase1 神经网络权重（PyTorch）
    └── phase1_norm_params.npz       Phase1 归一化参数（NumPy）
```

---

## 复现说明

### 环境要求
- Python 3.8+
- PyTorch, NumPy, Pandas, Matplotlib, SciPy
- PySR（符号回归库）
- LaTeX: XeLaTeX + ctex 宏包 + TeX Gyre Termes 字体

### 运行顺序

```bash
# 1. 数据预处理
python src/prepare_data2_pipeline.py

# 2. 主流程（Phase1 NN训练 + Phase2 符号回归 + 跨城市迁移）
python src/run_1plus3_on_data2.py

# 3. 绘图
python src/plot_per_city_fit_curves.py
python src/plot_other_cities_descriptive.py

# 4. 编译论文
cd paper_biye && xelatex thesis_draft.tex && xelatex thesis_draft.tex
```

---

## 文件统计

| 类别 | 文件数 | 大小 |
|------|--------|------|
| 论文 | 2 | 3.7 MB |
| 图片 | 98 (49 PNG + 49 PDF) | 22 MB |
| 脚本 | 6 | 88 KB |
| 数据 | 34 | 11 MB |
| 模型 | 2 | 8 KB |
| **合计** | **142** | **~36 MB** |
