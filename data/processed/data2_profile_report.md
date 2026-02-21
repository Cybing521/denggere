# data_2 profile report
- Modeling year window: `2005` to `2019`

## 1) Case-weather file (data_2/data.csv)
- Encoding detected: `gb18030`
- Rows (raw -> clean weekly): `15888` -> `12528`
- Cities: `16`
- Date range: `2005-01-02` to `2019-12-29`
- Weekly rows with cases > 0: `2053`
- Weekly rows with cases = 0: `10475`
- Unmapped city names: `None`

## 2) Mosquito index file (data_2/BI.csv)
- Encoding detected: `gb18030`
- Rows (raw): `30172`
- Guangdong rows kept (month numeric, city aligned): `906`
- Guangdong aligned cities: `8`
- BI year range after filtering: `2005` to `2019`
- Note: BI methods/units are mixed; one method is selected per city by monthly coverage.

## 3) Selected mosquito method per city
- Dongguan: `Breteau index` (months=24)
- Guangzhou: `Breteau index` (months=87)
- Huizhou: `Breteau index` (months=9)
- Jiangmen: `Breteau index` (months=36)
- Jieyang: `Labor hour` (months=18)
- Maoming: `Breteau index` (months=18)
- Shantou: `Breteau index` (months=44)
- Shenzhen: `Light trapping` (months=70)

## 4) Coverage snapshot (cases monthly vs BI monthly)
- Chaozhou: cases_months=180, case_pos_months=44, bi_months=NA, ratio=NA
- Dongguan: cases_months=180, case_pos_months=65, bi_months=24, ratio=0.13
- Foshan: cases_months=180, case_pos_months=76, bi_months=NA, ratio=NA
- Guangzhou: cases_months=180, case_pos_months=135, bi_months=87, ratio=0.48
- Huizhou: cases_months=180, case_pos_months=22, bi_months=9, ratio=0.05
- Jiangmen: cases_months=180, case_pos_months=39, bi_months=36, ratio=0.20
- Jieyang: cases_months=180, case_pos_months=25, bi_months=18, ratio=0.10
- Maoming: cases_months=180, case_pos_months=27, bi_months=18, ratio=0.10
- Qingyuan: cases_months=180, case_pos_months=28, bi_months=NA, ratio=NA
- Shantou: cases_months=180, case_pos_months=39, bi_months=44, ratio=0.24
- Shenzhen: cases_months=180, case_pos_months=105, bi_months=70, ratio=0.39
- Yangjiang: cases_months=180, case_pos_months=24, bi_months=NA, ratio=NA
- Zhanjiang: cases_months=180, case_pos_months=33, bi_months=NA, ratio=NA
- Zhaoqing: cases_months=180, case_pos_months=28, bi_months=NA, ratio=NA
- Zhongshan: cases_months=180, case_pos_months=64, bi_months=NA, ratio=NA
- Zhuhai: cases_months=180, case_pos_months=50, bi_months=NA, ratio=NA
