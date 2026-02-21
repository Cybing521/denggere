#!/usr/bin/env python3
"""Prepare data_2 as modeling-ready datasets.

Outputs are written to /root/wenmei/data_2/processed:
- cases_weather_weekly_utf8.csv
- cases_weather_biweekly_utf8.csv
- cases_weather_monthly_utf8.csv
- bi_guangdong_monthly_by_method.csv
- bi_guangdong_monthly_proxy.csv
- bi_proxy_method_selection.csv
- city_coverage_summary.csv
- data2_profile_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/root/wenmei")
DATA2 = ROOT / "data"
OUT = DATA2 / "processed"
YEAR_START = 2005
YEAR_END = 2019

CASES_FILE = DATA2 / "data.csv"
BI_FILE = DATA2 / "BI.csv"

CITY_CN_TO_EN: Dict[str, str] = {
    "东莞市": "Dongguan",
    "中山市": "Zhongshan",
    "佛山市": "Foshan",
    "广州市": "Guangzhou",
    "惠州市": "Huizhou",
    "揭阳市": "Jieyang",
    "汕头市": "Shantou",
    "江门市": "Jiangmen",
    "深圳市": "Shenzhen",
    "清远市": "Qingyuan",
    "湛江市": "Zhanjiang",
    "潮州市": "Chaozhou",
    "珠海市": "Zhuhai",
    "肇庆市": "Zhaoqing",
    "茂名市": "Maoming",
    "阳江市": "Yangjiang",
}

METHOD_PRIORITY = [
    "Breteau index",
    "Mosquito ovitrap index",
    "Light trapping",
    "Labor hour",
]


def read_csv_with_fallback(path: Path, candidates: Iterable[str] = ("utf-8", "gb18030", "gbk", "latin1")) -> Tuple[pd.DataFrame, str]:
    last_err = None
    for enc in candidates:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read {path} with fallback encodings: {last_err}")


def clean_cases_weather() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw, enc = read_csv_with_fallback(CASES_FILE, ("gb18030", "gbk", "utf-8", "latin1"))
    raw = raw.rename(columns={c: c.strip().lower() for c in raw.columns})

    need = {"city", "date", "cases", "tem", "rhu", "pre"}
    miss = need - set(raw.columns)
    if miss:
        raise ValueError(f"Missing columns in cases-weather file: {sorted(miss)}")

    raw["city_cn"] = raw["city"].astype(str).str.strip()
    raw["city_en"] = raw["city_cn"].map(CITY_CN_TO_EN)
    unknown_cn = sorted(raw.loc[raw["city_en"].isna(), "city_cn"].dropna().unique().tolist())

    raw["date"] = pd.to_datetime(raw["date"], format="%Y/%m/%d", errors="coerce")
    for c in ["cases", "tem", "rhu", "pre"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    clean = raw.dropna(subset=["date", "city_cn", "cases", "tem", "rhu", "pre"]).copy()
    clean = clean[(clean["date"].dt.year >= YEAR_START) & (clean["date"].dt.year <= YEAR_END)].copy()

    # Weekly level (deduplicated)
    weekly = (
        clean.groupby(["city_cn", "city_en", "date"], as_index=False)
        .agg(
            cases=("cases", "sum"),
            tem=("tem", "mean"),
            rhu=("rhu", "mean"),
            pre=("pre", "sum"),
        )
        .sort_values(["city_cn", "date"])
        .reset_index(drop=True)
    )
    weekly["year"] = weekly["date"].dt.year.astype(int)
    weekly["month"] = weekly["date"].dt.month.astype(int)
    weekly["week"] = weekly["date"].dt.isocalendar().week.astype(int)
    weekly["biweek"] = ((weekly["date"].dt.dayofyear - 1) // 14 + 1).astype(int)

    # Biweekly aggregation (for higher temporal resolution modeling)
    biweekly = (
        weekly.groupby(["city_cn", "city_en", "year", "biweek"], as_index=False)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_weeks=("date", "count"),
            cases=("cases", "sum"),
            tem=("tem", "mean"),
            rhu=("rhu", "mean"),
            pre=("pre", "sum"),
        )
        .sort_values(["city_cn", "year", "biweek"])
        .reset_index(drop=True)
    )

    # Monthly aggregation (compatible with existing pipeline style)
    monthly = (
        weekly.groupby(["city_cn", "city_en", "year", "month"], as_index=False)
        .agg(
            n_weeks=("date", "count"),
            cases=("cases", "sum"),
            tem=("tem", "mean"),
            rhu=("rhu", "mean"),
            pre=("pre", "sum"),
        )
        .sort_values(["city_cn", "year", "month"])
        .reset_index(drop=True)
    )

    meta = {
        "cases_encoding": enc,
        "cases_raw_rows": int(len(raw)),
        "cases_clean_rows": int(len(weekly)),
        "cases_cities": int(weekly["city_cn"].nunique()),
        "cases_date_min": str(weekly["date"].min().date()),
        "cases_date_max": str(weekly["date"].max().date()),
        "cases_positive_rows": int((weekly["cases"] > 0).sum()),
        "cases_zero_rows": int((weekly["cases"] == 0).sum()),
        "unknown_case_city_names": unknown_cn,
        "year_start": YEAR_START,
        "year_end": YEAR_END,
    }
    return weekly, biweekly, monthly, meta


def clean_bi(cities_en: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw, enc = read_csv_with_fallback(BI_FILE, ("gb18030", "gbk", "utf-8", "latin1"))
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})

    for c in ["Site_L1", "Site_L2", "Site_L3", "Site_method", "Site_month", "Site_year"]:
        if c in raw.columns:
            raw[c] = raw[c].astype(str).str.strip()

    raw["year"] = pd.to_numeric(raw.get("Site_year"), errors="coerce")
    raw["month"] = pd.to_numeric(raw.get("Site_month"), errors="coerce")
    raw["Den_admin"] = pd.to_numeric(raw.get("Den_admin"), errors="coerce")
    raw["Den_hab"] = pd.to_numeric(raw.get("Den_hab"), errors="coerce")
    raw["index_value"] = raw["Den_admin"].where(raw["Den_admin"].notna(), raw["Den_hab"])

    gd = raw[raw["Site_L1"].str.lower() == "guangdong"].copy()
    gd = gd[(gd["year"] >= YEAR_START) & (gd["year"] <= YEAR_END) & (gd["month"] >= 1) & (gd["month"] <= 12)].copy()
    gd["city_en"] = gd["Site_L2"].replace({"nan": np.nan, "NA": np.nan})
    gd["city_en"] = gd["city_en"].astype(str).str.strip()
    gd["city_en"] = gd["city_en"].replace({"nan": np.nan})

    # Keep only city names we can align with case-weather data.
    city_set = set(cities_en)
    gd = gd[gd["city_en"].isin(city_set)].copy()
    gd = gd[gd["index_value"].notna()].copy()

    by_method = (
        gd.groupby(["city_en", "year", "month", "Site_method"], as_index=False)
        .agg(
            index_value=("index_value", "mean"),
            n_records=("index_value", "count"),
        )
        .rename(columns={"Site_method": "method"})
        .sort_values(["city_en", "year", "month", "method"])
        .reset_index(drop=True)
    )

    # Select one method per city using coverage first, then method priority.
    method_stats = (
        by_method.groupby(["city_en", "method"], as_index=False)
        .agg(n_months=("index_value", "count"))
        .sort_values(["city_en", "n_months"], ascending=[True, False])
    )
    pri = {m: i for i, m in enumerate(METHOD_PRIORITY)}
    method_stats["priority"] = method_stats["method"].map(lambda x: pri.get(x, 999))
    method_stats = method_stats.sort_values(["city_en", "n_months", "priority"], ascending=[True, False, True])
    selected = method_stats.groupby("city_en", as_index=False).head(1).rename(columns={"method": "selected_method"})

    proxy = by_method.merge(selected[["city_en", "selected_method"]], left_on=["city_en", "method"], right_on=["city_en", "selected_method"], how="inner")
    proxy = proxy.drop(columns=["selected_method"]).copy()
    proxy["index_norm_city"] = proxy["index_value"] / proxy.groupby("city_en")["index_value"].transform("mean")
    proxy["index_norm_city"] = proxy["index_norm_city"].replace([np.inf, -np.inf], np.nan)

    meta = {
        "bi_encoding": enc,
        "bi_raw_rows": int(len(raw)),
        "bi_gd_rows": int(len(gd)),
        "bi_gd_cities": int(gd["city_en"].nunique()),
        "bi_year_min": int(by_method["year"].min()) if len(by_method) else None,
        "bi_year_max": int(by_method["year"].max()) if len(by_method) else None,
    }
    return by_method, proxy, selected, meta


def build_coverage(monthly_cases: pd.DataFrame, bi_proxy: pd.DataFrame) -> pd.DataFrame:
    c = (
        monthly_cases.groupby("city_en", as_index=False)
        .agg(
            cases_year_min=("year", "min"),
            cases_year_max=("year", "max"),
            n_case_months=("cases", "count"),
            n_case_months_pos=("cases", lambda s: int((s > 0).sum())),
        )
    )
    b = (
        bi_proxy.groupby("city_en", as_index=False)
        .agg(
            bi_year_min=("year", "min"),
            bi_year_max=("year", "max"),
            n_bi_months=("index_value", "count"),
            bi_mean=("index_value", "mean"),
        )
    )
    out = c.merge(b, on="city_en", how="left")
    out["bi_coverage_ratio_vs_cases"] = out["n_bi_months"] / out["n_case_months"]
    return out.sort_values("city_en").reset_index(drop=True)


def write_report(meta_cases: dict, meta_bi: dict, coverage: pd.DataFrame, method_sel: pd.DataFrame) -> None:
    lines = []
    lines.append("# data_2 profile report")
    lines.append(f"- Modeling year window: `{meta_cases['year_start']}` to `{meta_cases['year_end']}`")
    lines.append("")
    lines.append("## 1) Case-weather file (data_2/data.csv)")
    lines.append(f"- Encoding detected: `{meta_cases['cases_encoding']}`")
    lines.append(f"- Rows (raw -> clean weekly): `{meta_cases['cases_raw_rows']}` -> `{meta_cases['cases_clean_rows']}`")
    lines.append(f"- Cities: `{meta_cases['cases_cities']}`")
    lines.append(f"- Date range: `{meta_cases['cases_date_min']}` to `{meta_cases['cases_date_max']}`")
    lines.append(f"- Weekly rows with cases > 0: `{meta_cases['cases_positive_rows']}`")
    lines.append(f"- Weekly rows with cases = 0: `{meta_cases['cases_zero_rows']}`")
    if meta_cases["unknown_case_city_names"]:
        lines.append(f"- Unmapped city names: `{', '.join(meta_cases['unknown_case_city_names'])}`")
    else:
        lines.append("- Unmapped city names: `None`")

    lines.append("")
    lines.append("## 2) Mosquito index file (data_2/BI.csv)")
    lines.append(f"- Encoding detected: `{meta_bi['bi_encoding']}`")
    lines.append(f"- Rows (raw): `{meta_bi['bi_raw_rows']}`")
    lines.append(f"- Guangdong rows kept (month numeric, city aligned): `{meta_bi['bi_gd_rows']}`")
    lines.append(f"- Guangdong aligned cities: `{meta_bi['bi_gd_cities']}`")
    lines.append(f"- BI year range after filtering: `{meta_bi['bi_year_min']}` to `{meta_bi['bi_year_max']}`")
    lines.append("- Note: BI methods/units are mixed; one method is selected per city by monthly coverage.")

    lines.append("")
    lines.append("## 3) Selected mosquito method per city")
    for _, r in method_sel.sort_values("city_en").iterrows():
        lines.append(f"- {r['city_en']}: `{r['selected_method']}` (months={int(r['n_months'])})")

    lines.append("")
    lines.append("## 4) Coverage snapshot (cases monthly vs BI monthly)")
    for _, r in coverage.iterrows():
        n_bi = "NA" if pd.isna(r["n_bi_months"]) else int(r["n_bi_months"])
        ratio = "NA" if pd.isna(r["bi_coverage_ratio_vs_cases"]) else f"{r['bi_coverage_ratio_vs_cases']:.2f}"
        lines.append(
            f"- {r['city_en']}: cases_months={int(r['n_case_months'])}, "
            f"case_pos_months={int(r['n_case_months_pos'])}, bi_months={n_bi}, ratio={ratio}"
        )

    (OUT / "data2_profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    weekly, biweekly, monthly, m_cases = clean_cases_weather()
    by_method, bi_proxy, method_sel, m_bi = clean_bi(weekly["city_en"].dropna().unique().tolist())
    coverage = build_coverage(monthly, bi_proxy)

    weekly.to_csv(OUT / "cases_weather_weekly_utf8.csv", index=False, encoding="utf-8")
    biweekly.to_csv(OUT / "cases_weather_biweekly_utf8.csv", index=False, encoding="utf-8")
    monthly.to_csv(OUT / "cases_weather_monthly_utf8.csv", index=False, encoding="utf-8")
    by_method.to_csv(OUT / "bi_guangdong_monthly_by_method.csv", index=False, encoding="utf-8")
    bi_proxy.to_csv(OUT / "bi_guangdong_monthly_proxy.csv", index=False, encoding="utf-8")
    method_sel.to_csv(OUT / "bi_proxy_method_selection.csv", index=False, encoding="utf-8")
    coverage.to_csv(OUT / "city_coverage_summary.csv", index=False, encoding="utf-8")
    write_report(m_cases, m_bi, coverage, method_sel)

    print("=" * 72)
    print("Prepared data_2 modeling datasets")
    print("=" * 72)
    print(f"Cases weekly:   {OUT / 'cases_weather_weekly_utf8.csv'}")
    print(f"Cases biweekly: {OUT / 'cases_weather_biweekly_utf8.csv'}")
    print(f"Cases monthly:  {OUT / 'cases_weather_monthly_utf8.csv'}")
    print(f"BI by method:   {OUT / 'bi_guangdong_monthly_by_method.csv'}")
    print(f"BI proxy:       {OUT / 'bi_guangdong_monthly_proxy.csv'}")
    print(f"Coverage:       {OUT / 'city_coverage_summary.csv'}")
    print(f"Report:         {OUT / 'data2_profile_report.md'}")


if __name__ == "__main__":
    main()
