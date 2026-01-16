import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pd_distribution_utils import get_pd_distribution
from missing_data_experiments import apply_mcar, apply_mnar
from noisy_data_experiments import apply_noise

# -------------------- LOAD DATA --------------------

DATA_PATH = Path("data/raw/cs-training.csv")
df = pd.read_csv(DATA_PATH, index_col=0)

BASE_FEATURES = [
    "MonthlyIncome",
    "DebtRatio",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

TARGET = "SeriousDlqin2yrs"

# -------------------- PREP DATA --------------------

base_df = df[BASE_FEATURES + [TARGET]].copy()

# Baseline
baseline_pd = get_pd_distribution(base_df, BASE_FEATURES, TARGET)

# MCAR 30%
mcar_df = apply_mcar(base_df.copy(), 0.3)
mcar_df["MonthlyIncome"] = mcar_df["MonthlyIncome"].fillna(
    mcar_df["MonthlyIncome"].median()
)
mcar_pd = get_pd_distribution(mcar_df, BASE_FEATURES, TARGET)

# MNAR 20%
mnar_df = apply_mnar(base_df.copy(), 0.2)
mnar_df["MonthlyIncome"] = mnar_df["MonthlyIncome"].fillna(
    mnar_df["MonthlyIncome"].median()
)
mnar_pd = get_pd_distribution(mnar_df, BASE_FEATURES, TARGET)

# NOISE UP 20%
noise_df = apply_noise(base_df.copy(), 0.2, "up")
noise_pd = get_pd_distribution(noise_df, BASE_FEATURES, TARGET)

# -------------------- PLOT --------------------

plt.figure(figsize=(10, 6))

plt.hist(baseline_pd, bins=50, density=True, histtype="step", linewidth=2, label="Baseline")
plt.hist(mcar_pd, bins=50, density=True, histtype="step", linewidth=2, label="MCAR 30%")
plt.hist(mnar_pd, bins=50, density=True, histtype="step", linewidth=2, label="MNAR 20%")
plt.hist(noise_pd, bins=50, density=True, histtype="step", linewidth=2, label="Noise Up 20%")

plt.xlabel("Predicted Probability of Default")
plt.ylabel("Density")
plt.title("PD Distribution Shift under Data Quality Degradation")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/figures/pd_distribution_overlay.png", dpi=300)
plt.show()
