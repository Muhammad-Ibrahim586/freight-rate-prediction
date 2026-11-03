import joblib
import pandas as pd

from src.preprocessing import create_features


MODEL_PATH = "outputs/models/extra_trees.joblib"

VALIDATION_PATH = "data/validation.csv"
TEMPLATE_PATH = "data/validation-predictions-template.csv"
VALIDATION_OUTPUT = "validation_predictions.csv"

DECEMBER_PATH = "data/december-chart-inputs.csv"
DECEMBER_OUTPUT = "december_predictions.csv"


def main():
    model = joblib.load(MODEL_PATH)

    # Generate validation predictions.
    validation_df = pd.read_csv(VALIDATION_PATH)
    template_df = pd.read_csv(TEMPLATE_PATH)

    X_validation = create_features(validation_df)
    validation_predictions = model.predict(X_validation)

    submission = template_df.copy()
    submission["predicted_rate"] = validation_predictions
    submission.to_csv(VALIDATION_OUTPUT, index=False)

    print("Validation predictions saved:", VALIDATION_OUTPUT)
    print("Validation rows:", len(submission))

    # Generate December predictions.
    december_df = pd.read_csv(DECEMBER_PATH)

    X_december = create_features(december_df)
    december_predictions = model.predict(X_december)

    december_output = december_df.copy()
    december_output["predicted_rate"] = december_predictions
    december_output.to_csv(DECEMBER_OUTPUT, index=False)

    print("\nDecember predictions saved:", DECEMBER_OUTPUT)
    print("December rows:", len(december_output))
    print("December prediction range:")
    print(december_output["predicted_rate"].describe())


if __name__ == "__main__":
    main()
