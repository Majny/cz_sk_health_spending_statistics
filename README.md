# Liší se výdaje na zdravotnictví na obyvatele mezi ČR a SR? (párový t-test)

**Autor:** Jakub Dvořák  
**Datum:** 13. 8.

## Cíl
Porovnat výdaje na zdravotnictví na obyvatele v ČR a SR pomocí párového t-testu na ročních datech, kde jako páry budeme brát stejné roky.

## Data (zdroje)
- World Bank – **Current health expenditure per capita, PPP (current international $)** — kód `SH.XPD.CHEX.PP.CD`  
- Stránka indikátoru: https://data.worldbank.org/indicator/SH.XPD.CHEX.PP.CD  
- API CZ: https://api.worldbank.org/v2/country/CZ/indicator/SH.XPD.CHEX.PP.CD?format=json&per_page=20000  
-  API SK: https://api.worldbank.org/v2/country/SK/indicator/SH.XPD.CHEX.PP.CD?format=json&per_page=20000
-  Dokumentace API: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures

## Hypotézy
- **H₀:** průměrný rozdíl (ČR − SR) = 0  
- **H₁:** průměrný rozdíl (ČR − SR) ≠ 0  
- **α:** 0.05 (dvoustranně)

## Reprodukce
```bash
./run.sh
# výstupy:
# plots/01_trend.png
# plots/02_diff_bar.png
# plots/03_diff_hist.png
# plots/04_scatter_cz_vs_sk.png
# plots/05_qqplot_diff.png
# results/table_czsk.csv
# results/results_summary.md

