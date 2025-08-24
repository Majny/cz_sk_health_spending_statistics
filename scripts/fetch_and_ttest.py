#!/usr/bin/env python3

"""
Párový t-test výdajů na zdravotnictví na obyvatele (PPP) mezi ČR a SR.
Data: World Bank, indikátor SH.XPD.CHEX.PP.CD.

Autor: Jakub Dvořák
Datum: 13. 8.
"""

import os, math
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

AUTHOR = "Jakub Dvořák"
DATE = "13. 8."

# --- ukládat o úroveň výš (../plots, ../results) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUT_PLOTS = os.path.join(PROJ_ROOT, "plots")
OUT_RESULTS = os.path.join(PROJ_ROOT, "results")
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

WB_JSON = "https://api.worldbank.org/v2/country/{cc}/indicator/{indicator}?format=json&per_page=20000"
INDICATOR = "SH.XPD.CHEX.PP.CD"

def fetch_series(cc: str, indicator: str) -> pd.DataFrame:
    url = WB_JSON.format(cc=cc, indicator=indicator)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        raise RuntimeError(f"Error")
    rows = pd.DataFrame(data[1])[["date", "value"]].rename(columns={"date": "year"})
    rows["year"] = rows["year"].astype(int)
    rows = rows.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    return rows

# Načtení a sloučení
cz = fetch_series("CZ", INDICATOR).rename(columns={"value": "CZ"})
sk = fetch_series("SK", INDICATOR).rename(columns={"value": "SK"})
df = pd.merge(cz, sk, on="year", how="inner").sort_values("year").reset_index(drop=True)
df["diff"] = df["CZ"] - df["SK"]

# Výpočet ratio
df["ratio"] = df["CZ"] / df["SK"]
df["log_ratio"] = np.log(df["ratio"])  # pro test konstantního podílu

# Ulož CSV -> ../results/
csv_path = os.path.join(OUT_RESULTS, "table_czsk.csv")
df.to_csv(csv_path, index=False)
print(f"Uloženo: {csv_path}")

# Grafy -> ../plots/
plt.rcParams.update({"figure.dpi": 120})

# 1) Trendy + průměry
plt.figure()
plt.plot(df["year"], df["CZ"], label="ČR", color="C0")
plt.plot(df["year"], df["SK"], label="SR", color="C1")
plt.axhline(df["CZ"].mean(), linestyle="--", alpha=0.7, color="C0", label="průměr ČR")
plt.axhline(df["SK"].mean(), linestyle="--", alpha=0.7, color="C1", label="průměr SR")
plt.xlabel("Rok"); plt.ylabel("USD PPP na obyvatele")
plt.title("Výdaje na zdravotnictví na obyvatele (PPP) – trend")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "01_trend.png"), dpi=150)
plt.close()

# 2) Rozdíl po letech
colors = np.where(df["diff"] >= 0, "#1f77b4", "#d62728")
plt.figure()
plt.bar(df["year"], df["diff"], color=colors, width=0.7)
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(df["diff"].mean(), linestyle="--", alpha=0.5, label=f"průměr diff = {df['diff'].mean():.0f}")
plt.xlabel("Rok"); plt.ylabel("Rozdíl (ČR − SR)")
plt.title("Rozdíl výdajů (PPP) po letech")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "02_diff_bar.png"), dpi=150)
plt.close()

# 2R) Podíl (CZ/SK) po letech
plt.figure()
plt.bar(df["year"], df["ratio"], width=0.7)
plt.axhline(1.0, color="black", linewidth=0.8, label="= 1 (stejné)")
plt.axhline(df["ratio"].mean(), linestyle="--", alpha=0.5, label=f"průměr ratio = {df['ratio'].mean():.3f}")
plt.xlabel("Rok"); plt.ylabel("Podíl CZ/SK")
plt.title("Podíl výdajů (CZ/SK) po letech")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "02_ratio_bar.png"), dpi=150)
plt.close()

# 3) Histogram rozdílů + normála
mu, sd = df["diff"].mean(), df["diff"].std(ddof=1)
x = np.linspace(mu - 4*sd, mu + 4*sd, 200)
plt.figure()
plt.hist(df["diff"], bins=10, density=True, alpha=0.6, edgecolor="white")
plt.plot(x, norm.pdf(x, mu, sd))
plt.xlabel("Rozdíl (ČR − SR)"); plt.ylabel("Hustota")
plt.title("Rozdělení rozdílů (CZ−SK)"); plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "03_diff_hist.png"), dpi=150)
plt.close()

# 3R) Histogram podílu
plt.figure()
plt.hist(df["ratio"], bins=10, alpha=0.7, edgecolor="white")
plt.axvline(1.0, color="black", linewidth=0.8, label="= 1 (stejné)")
plt.axvline(df["ratio"].mean(), linestyle="--", alpha=0.5, label=f"průměr = {df['ratio'].mean():.3f}")
plt.xlabel("Podíl CZ/SK"); plt.ylabel("Počet let")
plt.title("Rozdělení podílu (CZ/SK)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "03_ratio_hist.png"), dpi=150)
plt.close()

# 4) Scatter CZ vs. SK
plt.figure()
plt.scatter(df["SK"], df["CZ"], s=40)
mx = max(df["CZ"].max(), df["SK"].max())
plt.plot([0, mx], [0, mx], linestyle="--")
plt.xlabel("SR (USD PPP/obyv.)"); plt.ylabel("ČR (USD PPP/obyv.)")
plt.title("ČR vs. SR – porovnání ročních hodnot")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "04_scatter_cz_vs_sk.png"), dpi=150)
plt.close()

# 5) QQ-plot rozdílů
plt.figure()
stats.probplot(df["diff"], dist="norm", plot=plt)
plt.title("QQ-plot rozdílů (ČR−SR)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "05_qqplot_diff.png"), dpi=150)
plt.close()

print(f"Uloženy PNG do: {OUT_PLOTS}")

# Párový t-test
diff = df["diff"].to_numpy(float)
n = diff.size
mean_diff = float(diff.mean())
sd_diff = float(diff.std(ddof=1))
se_diff = sd_diff / math.sqrt(n)

tt = stats.ttest_rel(df["CZ"], df["SK"])
alpha = 0.05
tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
ci_lo = mean_diff - tcrit * se_diff
ci_hi = mean_diff + tcrit * se_diff
p = float(tt.pvalue)
p_str = f"{p:.5f}" if p >= 0.001 else "< 0.001"
direction = "vyšší" if mean_diff > 0 else "nižší"

vysledky = (
    f"Párový t-test (spárované roky CZ vs. SK, n = {n}).\n"
    f"Průměrný rozdíl (CZ−SK): {mean_diff:.0f} PPP USD/obyv.\n"
    f"95% CI [{ci_lo:.0f}, {ci_hi:.0f}].\n"
    f"t({n-1}) = {tt.statistic:.3f}, p = {p_str}.\n"
)

if p < alpha:
    zaver = (
        f"Na hladině α = {alpha} zamítáme H₀. "
        f"V průměru jsou výdaje {direction} v ČR než v SR."
    )
else:
    zaver = (
        f"Na hladině α = {alpha} H₀ nezamítáme. "
        f"Rozdíl průměrných výdajů se nepotvrdil."
    )

md_path = os.path.join(OUT_RESULTS, "results_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Výsledky párového t-testu\n\n")
    f.write(f"- Autor: {AUTHOR}\n- Datum: {DATE}\n\n")
    f.write(vysledky + "\n")
    f.write("## Závěr\n" + zaver + "\n")

print(f"Uloženo: {md_path}")

print("\nShrnutí:")
print(vysledky)
print(zaver)

# Korelace CZ vs. SK
r, p_r = stats.pearsonr(df["CZ"], df["SK"])

# Test konstantního podílu
slope, intercept, r_val, p_slope, se_slope = stats.linregress(df["year"], df["log_ratio"])
t_slope = (slope / se_slope) if se_slope > 0 else float("nan")

# Geometrický průměr podílu a 95% CI
mu_lr = df["log_ratio"].mean()
sd_lr = df["log_ratio"].std(ddof=1)
se_lr = sd_lr / math.sqrt(len(df))
tcrit_ratio = stats.t.ppf(0.975, df=len(df)-1)
gm = math.exp(mu_lr)
ci_ratio_lo = math.exp(mu_lr - tcrit_ratio * se_lr)
ci_ratio_hi = math.exp(mu_lr + tcrit_ratio * se_lr)

with open(md_path, "a", encoding="utf-8") as f:
    f.write("\n## Podíl CZ/SK\n\n")
    f.write(f"- Korelace CZ vs. SK: r = {r:.3f}, p = {p_r:.3g}\n")
    f.write(f"- Geometrický průměr podílu CZ/SK: {gm:.3f} (95% CI [{ci_ratio_lo:.3f}, {ci_ratio_hi:.3f}])\n")
    f.write(f"- Test konstantního podílu: slope = {slope:.4e}, "
            f"t ≈ {t_slope:.2f}, p = {p_slope:.3g}, R² = {r_val**2:.3f}\n")

print("\nDoplňková analýza:")
print(f"Korelace CZ vs. SK: r = {r:.3f}, p = {p_r:.3g}")
print(f"Geometrický průměr podílu: {gm:.3f} (95% CI [{ci_ratio_lo:.3f}, {ci_ratio_hi:.3f}])")
print(f"log(CZ/SK) ~ rok: slope = {slope:.4e}, t ≈ {t_slope:.2f}, p = {p_slope:.3g}, R² = {r_val**2:.3f}")
