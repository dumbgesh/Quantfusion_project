import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- PATHS --------------------

RESULTS_PATH = Path("outputs/results")
FIG_PATH = Path("outputs/figures")
FIG_PATH.mkdir(parents=True, exist_ok=True)

# -------------------- LOAD RESULTS --------------------

baseline = pd.read_csv(
    RESULTS_PATH / "baseline_metrics.csv",
    index_col=0
)
baseline_el = baseline.loc["total_el"].values[0]

missing = pd.read_csv(
    RESULTS_PATH / "missing_data_results_strengthened.csv"
)

noisy = pd.read_csv(
    RESULTS_PATH / "noisy_data_results.csv"
)

# -------------------- COMBINE RESULTS --------------------

rows = []

rows.append({
    "scenario": "BASELINE",
    "el_index": 100.0
})

for _, r in missing.iterrows():
    rows.append({
        "scenario": r["scenario"],
        "el_index": (r["total_el"] / baseline_el) * 100
    })

for _, r in noisy.iterrows():
    rows.append({
        "scenario": r["scenario"],
        "el_index": (r["total_el"] / baseline_el) * 100
    })

df = pd.DataFrame(rows)

# -------------------- ORDER SCENARIOS --------------------

order = [
    "BASELINE",
    "MCAR_10", "MCAR_20", "MCAR_30",
    "MNAR_10", "MNAR_20", "MNAR_30",
    "NOISE_DOWN_10", "NOISE_DOWN_20",
    "NOISE_UP_10", "NOISE_UP_20",
]

df["scenario"] = pd.Categorical(
    df["scenario"],
    categories=order,
    ordered=True
)
df = df.sort_values("scenario")

# -------------------- Δ EXPECTED LOSS (BASIS POINTS) --------------------

df["delta_el_bps"] = (df["el_index"] - 100) * 100

# -------------------- PLOT --------------------

plt.figure(figsize=(13, 6))

plt.bar(
    df["scenario"],
    df["delta_el_bps"]
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.ylabel("Δ Expected Loss (basis points)")
plt.xlabel("Data Quality Scenario")
plt.title(
    "Expected Loss Sensitivity to Missing and Noisy Income Data",
    pad=15
)

plt.xticks(rotation=45, ha="right")

# -------------------- FINALIZE --------------------

plt.tight_layout()
plt.savefig(
    FIG_PATH / "expected_loss_sensitivity_minimal.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
