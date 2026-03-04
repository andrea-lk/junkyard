import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def run_logistic_regression():
    # Load trial-level data
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    processed_df = pd.read_csv(data_dir / "processed_behavioral_data.csv")

    # Build binary outcome: gamble (accept) = 1, safe (reject) = 0
    processed_df["accept"] = (processed_df["choice"] == "gamble").astype(int)

    # Previous trial choice (lagged within each subject)
    processed_df = processed_df.sort_values(["sub", "trial"])
    processed_df["prev_accept"] = processed_df.groupby("sub")["accept"].shift(1)

    # Drop first trial per subject (no previous trial available)
    processed_df = processed_df.dropna(subset=["prev_accept"])
    processed_df["prev_accept"] = processed_df["prev_accept"].astype(int)

    # Dummy-code trial_type (drop 'symmetric' as reference category)
    trial_dummies = pd.get_dummies(
        processed_df["trial_type"], prefix="trial_type", drop_first=False, dtype=int
    )
    trial_dummies = trial_dummies.drop(columns="trial_type_symmetric")

    # Build design matrix: accept ~ 1 + trial_type + prev_accept
    X = pd.concat(
        [trial_dummies, processed_df[["prev_accept"]]],
        axis=1,
    )
    X = sm.add_constant(X)
    y = processed_df["accept"]

    # Fit binary logistic regression
    model = sm.Logit(y, X)
    result = model.fit(disp=True)

    # Print full summary
    print(result.summary())

    # Print odds ratios with 95% CI
    print("\n--- Odds Ratios (with 95% CI) ---")
    ci = result.conf_int()
    ci_exp = np.exp(ci)
    ci_exp.columns = ["OR_2.5%", "OR_97.5%"]
    odds_table = pd.DataFrame({
        "coef": result.params.round(4),
        "OR": np.exp(result.params).round(4),
        "OR_2.5%": ci_exp["OR_2.5%"].round(4),
        "OR_97.5%": ci_exp["OR_97.5%"].round(4),
        "p-value": result.pvalues.round(4),
    })
    print(odds_table)

    return result


if __name__ == "__main__":
    run_logistic_regression()
