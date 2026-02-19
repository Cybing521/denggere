#!/usr/bin/env python3
"""Generate descriptive figures for other Guangdong cities.

Outputs:
- Cross-city 2014 case distribution
- Cross-city weather distributions and climatology
- Case-weather descriptive scatter (2014 city-level)
- Appendix-ready LaTeX snippet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr


ROOT = Path("/root/wenmei")
DATA = ROOT / "data"
OUT = ROOT / "results" / "paper_extra"
YEAR_START = 2005
YEAR_END = 2019


def load_monthly_weather() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fp in sorted((DATA / "cities").glob("*_monthly.csv")):
        city = fp.stem.replace("_monthly", "").title()
        df = pd.read_csv(fp)
        need = {"year", "month", "temperature", "humidity", "precipitation"}
        if not need.issubset(df.columns):
            continue
        df = df[list(need)].copy()
        df["city"] = city
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No city monthly weather files found in data/cities.")
    w = pd.concat(rows, ignore_index=True)
    w["year"] = w["year"].astype(int)
    w["month"] = w["month"].astype(int)
    w = w[(w["year"] >= YEAR_START) & (w["year"] <= YEAR_END)].copy()
    return w


def load_city_cases_2014() -> pd.DataFrame:
    path = DATA / "public_sources" / "opendengue" / "spatial_v1.1.csv"
    df = pd.read_csv(path)
    gd = df[
        (df["adm_0_name"] == "CHINA")
        & (df["adm_1_name"].str.contains("GUANGDONG", case=False, na=False))
        & (df["adm_2_name"].notna())
        & (df["Year"] == 2014)
        & (df["T_res"] == "Year")
    ].copy()
    out = (
        gd.groupby("adm_2_name")["dengue_total"]
        .max()
        .reset_index()
        .rename(columns={"adm_2_name": "city", "dengue_total": "cases_2014"})
    )
    out["city"] = out["city"].str.title()
    out["cases_2014"] = out["cases_2014"].astype(float)
    return out


def plot_cases_2014(cases: pd.DataFrame) -> Path:
    d = cases.sort_values("cases_2014", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(d["city"], d["cases_2014"], color="tab:orange", alpha=0.85)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("2014 cases (symlog)")
    ax.set_title("Other-city dengue case distribution in Guangdong (2014)")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    fig.tight_layout()
    out = OUT / "D8_other_cities_cases_2014_distribution.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_weather_city_boxes(weather: pd.DataFrame) -> Path:
    annual = (
        weather.groupby(["city", "year"], as_index=False)
        .agg(
            temp_mean=("temperature", "mean"),
            humidity_mean=("humidity", "mean"),
            precip_mean=("precipitation", "mean"),
        )
        .sort_values(["city", "year"])
    )
    cities = sorted(annual["city"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cols = [
        ("temp_mean", "Temperature (C)"),
        ("humidity_mean", "Humidity (%)"),
        ("precip_mean", "Precipitation (daily mean)"),
    ]
    for ax, (col, title) in zip(axes, cols):
        data = [annual.loc[annual["city"] == c, col].values for c in cities]
        ax.boxplot(data, labels=cities, patch_artist=True, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Cross-city weather distribution (annual means from monthly series)", y=1.03)
    fig.tight_layout()
    out = OUT / "D9_other_cities_weather_boxplots.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_weather_climatology(weather: pd.DataFrame) -> Path:
    months = np.arange(1, 13)
    cities = sorted(weather["city"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    specs = [
        ("temperature", "Temperature (C)"),
        ("humidity", "Humidity (%)"),
        ("precipitation", "Precipitation (daily mean)"),
    ]
    for ax, (col, title) in zip(axes, specs):
        clim = weather.groupby(["city", "month"], as_index=False)[col].mean()
        for c in cities:
            cdf = clim[clim["city"] == c].sort_values("month")
            ax.plot(cdf["month"], cdf[col], color="0.7", lw=0.8, alpha=0.6)
        med = clim.groupby("month")[col].median().reindex(months)
        q25 = clim.groupby("month")[col].quantile(0.25).reindex(months)
        q75 = clim.groupby("month")[col].quantile(0.75).reindex(months)
        ax.plot(months, med.values, color="tab:red", lw=2.2, label="Cross-city median")
        ax.fill_between(months, q25.values, q75.values, color="tab:red", alpha=0.18, label="IQR")
        ax.set_xticks(months)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Monthly climatology across Guangdong cities ({YEAR_START}-{YEAR_END})", y=1.03)
    fig.tight_layout()
    out = OUT / "D10_other_cities_weather_climatology.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_cases_vs_weather_2014(cases: pd.DataFrame, weather: pd.DataFrame) -> Path:
    w14 = (
        weather[weather["year"] == 2014]
        .groupby("city", as_index=False)
        .agg(
            temp_2014=("temperature", "mean"),
            humidity_2014=("humidity", "mean"),
            precip_2014=("precipitation", "sum"),
        )
    )
    m = cases.merge(w14, on="city", how="inner")
    m = m.sort_values("cases_2014", ascending=False).copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    pairs = [
        ("temp_2014", "2014 mean temperature (C)"),
        ("humidity_2014", "2014 mean humidity (%)"),
        ("precip_2014", "2014 accumulated precipitation"),
    ]
    y = np.log1p(m["cases_2014"].values.astype(float))
    for ax, (xcol, xlabel) in zip(axes, pairs):
        x = m[xcol].values.astype(float)
        ax.scatter(x, y, s=55, alpha=0.85, c="tab:blue")
        for _, r in m.iterrows():
            ax.text(r[xcol], np.log1p(r["cases_2014"]) + 0.02, r["city"], fontsize=7)
        if len(np.unique(x)) > 1:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 120)
            ax.plot(xx, coef[0] * xx + coef[1], "r--", lw=1.3)
            rho, p = spearmanr(x, y)
            ax.text(
                0.03,
                0.97,
                f"Spearman rho={rho:.2f}\np={p:.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("log(1 + 2014 cases)")
        ax.grid(alpha=0.25)

    fig.suptitle("Other-city cases vs weather descriptors (2014)", y=1.04)
    fig.tight_layout()
    out = OUT / "D11_other_cities_cases_vs_weather_2014.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out, m


def write_appendix_snippet() -> Path:
    snippet = r"""
\subsection{其他城市病例与气象描述图（附录新增）}

\noindent 为补充第二部分外推验证的数据背景，本附录加入广东其他城市的病例与气象描述图。
图D8给出2014年城市病例分布；图D9--D10给出跨城市气象统计与季节型；
图D11给出2014年城市病例与关键气象描述量（温度、湿度、降水）的散点关系。

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{../results/paper_extra/D8_other_cities_cases_2014_distribution.png}
    \caption{附录D8：广东其他城市2014年病例分布（对数坐标）。}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.99\textwidth]{../results/paper_extra/D9_other_cities_weather_boxplots.png}
    \caption{附录D9：其他城市气象分布（按城市年度均值的箱线图）。}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.99\textwidth]{../results/paper_extra/D10_other_cities_weather_climatology.png}
    \caption{附录D10：其他城市月度气候型（中位数及四分位带）。}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.99\textwidth]{../results/paper_extra/D11_other_cities_cases_vs_weather_2014.png}
    \caption{附录D11：2014年城市病例与气象描述量散点关系。}
\end{figure}
"""
    out = ROOT / "deliverables" / "appendix_other_cities_snippet.tex"
    out.write_text(snippet.strip() + "\n", encoding="utf-8")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weather = load_monthly_weather()
    cases = load_city_cases_2014()

    # Restrict to cities with both weather and 2014 cases.
    common = sorted(set(weather["city"]).intersection(set(cases["city"])))
    weather = weather[weather["city"].isin(common)].copy()
    cases = cases[cases["city"].isin(common)].copy()

    fig1 = plot_cases_2014(cases)
    fig2 = plot_weather_city_boxes(weather)
    fig3 = plot_weather_climatology(weather)
    fig4, merged = plot_cases_vs_weather_2014(cases, weather)

    summary_csv = OUT / "other_cities_2014_case_weather_summary.csv"
    merged.sort_values("cases_2014", ascending=False).to_csv(summary_csv, index=False)

    city_stats = (
        weather.groupby("city", as_index=False)
        .agg(
            years=("year", "nunique"),
            months=("month", "count"),
            temp_mean=("temperature", "mean"),
            humidity_mean=("humidity", "mean"),
            precip_mean=("precipitation", "mean"),
        )
        .sort_values("city")
    )
    city_stats_csv = OUT / "other_cities_weather_city_stats.csv"
    city_stats.to_csv(city_stats_csv, index=False)

    snippet = write_appendix_snippet()

    print("=" * 72)
    print("Other-city descriptive figures generated")
    print("=" * 72)
    print(f"Cities included: {len(common)}")
    print(f"Case figure:      {fig1}")
    print(f"Weather figure 1: {fig2}")
    print(f"Weather figure 2: {fig3}")
    print(f"Case-weather fig: {fig4}")
    print(f"Summary table:    {summary_csv}")
    print(f"City stats:       {city_stats_csv}")
    print(f"LaTeX snippet:    {snippet}")


if __name__ == "__main__":
    main()
