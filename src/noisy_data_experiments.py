import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------- CONFIG --------------------

DATA_PATH = Path("data/raw/cs-training.csv")
OUTPUT_PATH = Path("outputs/results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "MonthlyIncome",
    "DebtRatio",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

TARGET = "SeriousDlqin2yrs"

LGD = 0.45
EAD = 1.0
RANDOM_STATE = 42

# -------------------- DATA --------------------

def load_data():
    return pd.read_csv(DATA_PATH, index_col=0)

def base_dataframe(df):
    return df[BASE_FEATURES + [TARGET]].copy()

# -------------------- NOISE GENERATORS --------------------

def apply_noise(df, noise_frac, direction):
    """
    noise_frac: 0.1 or 0.2
    direction: 'up' or 'down'
    """
    df = df.copy()

    noise_multiplier = 1 + noise_frac if direction == "up" else 1 - noise_frac

    # Apply noise only to non-missing income
    mask = df["MonthlyIncome"].notna()
    df.loc[mask, "MonthlyIncome"] = (
        df.loc[mask, "MonthlyIncome"] * noise_multiplier
    )

    return df

# -------------------- EXPERIMENT --------------------

def run_experiment(df):
    df = df.copy()

    # No missing flag here — income exists but is wrong
    median_income = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(median_income)

    X = df[BASE_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    pd_test = model.predict_proba(X_test)[:, 1]
    EL_test = pd_test * LGD * EAD

    return {
        "mean_pd": pd_test.mean(),
        "p95_pd": np.percentile(pd_test, 95),
        "mean_el": EL_test.mean(),
        "total_el": EL_test.sum()
    }

# -------------------- MAIN --------------------

if __name__ == "__main__":

    np.random.seed(RANDOM_STATE)

    df = load_data()
    base_df = base_dataframe(df)

    results = []

    for noise_frac in [0.1, 0.2]:
        for direction in ["up", "down"]:

            df_noisy = apply_noise(base_df, noise_frac, direction)
            metrics = run_experiment(df_noisy)

            metrics["scenario"] = f"NOISE_{direction.upper()}_{int(noise_frac*100)}"
            results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["scenario", "mean_pd", "p95_pd", "mean_el", "total_el"]
    ]

    results_df.to_csv(
        OUTPUT_PATH / "noisy_data_results.csv",
        index=False
    )

    print("\nNOISY DATA EXPERIMENT RESULTS")
    print(results_df)
