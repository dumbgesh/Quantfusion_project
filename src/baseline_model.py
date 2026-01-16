import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------- CONFIG --------------------

DATA_PATH = Path("data/raw/cs-training.csv")
OUTPUT_PATH = Path("outputs/results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = [
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

def prepare_baseline(df):
    df_model = df[FEATURES + [TARGET]].copy()
    df_model["MonthlyIncome"] = df_model["MonthlyIncome"].fillna(
        df_model["MonthlyIncome"].median()
    )
    return df_model

# -------------------- MODEL --------------------

def train_model(X, y):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    model.fit(X, y)
    return model

# -------------------- MAIN --------------------

if __name__ == "__main__":

    df = load_data()
    df_model = prepare_baseline(df)

    # Save baseline dataset (NOW it exists)
    df_model.to_csv(OUTPUT_PATH / "baseline_data.csv", index=False)

    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)

    pd_test = model.predict_proba(X_test)[:, 1]
    EL_test = pd_test * LGD * EAD

    print("\nBASELINE PD SUMMARY")
    print(f"Mean PD: {pd_test.mean():.4f}")
    print(f"Median PD: {np.median(pd_test):.4f}")
    print(f"95th Percentile PD: {np.percentile(pd_test, 95):.4f}")

    print("\nBASELINE EXPECTED LOSS")
    print(f"Mean EL: {EL_test.mean():.4f}")
    print(f"Total EL (normalized): {EL_test.sum():.4f}")

    coef_df = pd.DataFrame(
        {
            "Variable": FEATURES,
            "Coefficient": model.coef_[0],
        }
    ).sort_values(by="Coefficient", ascending=False)

    print("\nMODEL COEFFICIENTS")
    print(coef_df)

    baseline_metrics = pd.Series(
        {
            "mean_pd": pd_test.mean(),
            "median_pd": np.median(pd_test),
            "p95_pd": np.percentile(pd_test, 95),
            "mean_el": EL_test.mean(),
            "total_el": EL_test.sum(),
        }
    )

    baseline_metrics.to_csv(OUTPUT_PATH / "baseline_metrics.csv")
    coef_df.to_csv(OUTPUT_PATH / "baseline_coefficients.csv", index=False)
