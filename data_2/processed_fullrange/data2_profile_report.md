# data_2 profile report
- Modeling year window: `2004` to `2023`

## 1) Case-weather file (data_2/data.csv)
- Encoding detected: `gb18030`
- Rows (raw -> clean weekly): `15888` -> `15888`
- Cities: `16`
- Date range: `2004-12-26` to `2023-12-31`
- Weekly rows with cases > 0: `2379`
- Weekly rows with cases = 0: `13509`
- Unmapped city names: `None`

## 2) Mosquito index file (data_2/BI.csv)
- Encoding detected: `gb18030`
- Rows (raw): `30172`
- Guangdong rows kept (month numeric, city aligned): `1075`
- Guangdong aligned cities: `8`
- BI year range after filtering: `2005` to `2023`
- Note: BI methods/units are mixed; one method is selected per city by monthly coverage.

## 3) Selected mosquito method per city
- Dongguan: `Breteau index` (months=24)
- Guangzhou: `Breteau index` (months=130)
- Huizhou: `Breteau index` (months=9)
- Jiangmen: `Breteau index` (months=36)
- Jieyang: `Labor hour` (months=18)
- Maoming: `Breteau index` (months=18)
- Shantou: `Breteau index` (months=44)
- Shenzhen: `Mosquito ovitrap index` (months=87)

## 4) Coverage snapshot (cases monthly vs BI monthly)
- Chaozhou: cases_months=229, case_pos_months=49, bi_months=NA, ratio=NA
- Dongguan: cases_months=229, case_pos_months=72, bi_months=24, ratio=0.10
- Foshan: cases_months=229, case_pos_months=87, bi_months=NA, ratio=NA
- Guangzhou: cases_months=229, case_pos_months=165, bi_months=130, ratio=0.57
- Huizhou: cases_months=229, case_pos_months=25, bi_months=9, ratio=0.04
- Jiangmen: cases_months=229, case_pos_months=45, bi_months=36, ratio=0.16
- Jieyang: cases_months=229, case_pos_months=30, bi_months=18, ratio=0.08
- Maoming: cases_months=229, case_pos_months=33, bi_months=18, ratio=0.08
- Qingyuan: cases_months=229, case_pos_months=34, bi_months=NA, ratio=NA
- Shantou: cases_months=229, case_pos_months=44, bi_months=44, ratio=0.19
- Shenzhen: cases_months=229, case_pos_months=119, bi_months=87, ratio=0.38
- Yangjiang: cases_months=229, case_pos_months=26, bi_months=NA, ratio=NA
- Zhanjiang: cases_months=229, case_pos_months=39, bi_months=NA, ratio=NA
- Zhaoqing: cases_months=229, case_pos_months=34, bi_months=NA, ratio=NA
- Zhongshan: cases_months=229, case_pos_months=74, bi_months=NA, ratio=NA
- Zhuhai: cases_months=229, case_pos_months=56, bi_months=NA, ratio=NA
