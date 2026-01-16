import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------- CONFIG --------------------

DATA_PATH = Path("data/raw/cs-training.csv")
OUTPUT_PATH = Path("outputs/results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FEATURES_BASE = [
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
    return df[FEATURES_BASE + [TARGET]].copy()

# -------------------- MISSINGNESS --------------------

def apply_mcar(df, missing_frac):
    df = df.copy()
    n = len(df)
    idx = np.random.choice(df.index, size=int(n * missing_frac), replace=False)
    df.loc[idx, "MonthlyIncome"] = np.nan
    return df

def apply_mnar(df, missing_frac):
    df = df.copy()

    risk_score = (
        1.5 * df["DebtRatio"] +
        2.0 * df["NumberOfTime30-59DaysPastDueNotWorse"] +
        2.0 * df["NumberOfTime60-89DaysPastDueNotWorse"]
    )

    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    prob_missing = missing_frac * (0.3 + 1.7 * risk_score)

    random_draw = np.random.uniform(0, 1, size=len(df))
    df.loc[random_draw < prob_missing, "MonthlyIncome"] = np.nan

    return df

# -------------------- EXPERIMENT --------------------

def run_experiment(df):
    df = df.copy()

    df["IncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    median_income = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(median_income)

    FEATURES = FEATURES_BASE + ["IncomeMissing"]

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
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

    baseline = pd.read_csv(OUTPUT_PATH / "baseline_metrics.csv", index_col=0)

    df = load_data()
    base_df = base_dataframe(df)

    results = []

    for missing_frac in [0.1, 0.2, 0.3]:

        df_mcar = apply_mcar(base_df.copy(), missing_frac)
        mcar = run_experiment(df_mcar)
        mcar["scenario"] = f"MCAR_{int(missing_frac*100)}"

        df_mnar = apply_mnar(base_df.copy(), missing_frac)
        mnar = run_experiment(df_mnar)
        mnar["scenario"] = f"MNAR_{int(missing_frac*100)}"

        results.extend([mcar, mnar])

    results_df = pd.DataFrame(results)

    results_df["pct_change_el"] = (
        (results_df["total_el"] - baseline.loc["total_el"].values[0])
        / baseline.loc["total_el"].values[0]
    ) * 100

    results_df = results_df[
        ["scenario", "mean_pd", "p95_pd", "mean_el", "total_el", "pct_change_el"]
    ]

    results_df.to_csv(
        OUTPUT_PATH / "missing_data_results_strengthened.csv",
        index=False
    )

    print("\nSTRENGTHENED MISSING DATA RESULTS")
    print(results_df)
