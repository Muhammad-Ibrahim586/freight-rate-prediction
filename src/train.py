import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from src.evaluate import evaluate_model
from sklearn.pipeline import Pipeline

from src.preprocessing import create_features, build_preprocessor


DATA_PATH = "data/train-test.csv"
MODEL_DIR = "outputs/models"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data.
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Chronological split.
    train_df = df[df["date"] < "2025-09-01"].copy()
    val_df = df[df["date"] >= "2025-09-01"].copy()

    X_train = create_features(train_df)
    y_train = train_df["posted_rate"]

    X_val = create_features(val_df)
    y_val = val_df["posted_rate"]


    # Candidate models.
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=80,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=2,
        ),
    }

    results = {}

    for name, regressor in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_val, y_val)
        results[name] = metrics

        print(f"{name} results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save each trained model.
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"Saved: {model_path}")

    # Select best model using MAE.
    best_model_name = min(
        results,
        key=lambda name: results[name]["MAE"]
    )

    print("\nBest model:", best_model_name)
    print("Best validation MAE:", results[best_model_name]["MAE"])

    # Save model selection information.
    with open(
        os.path.join(MODEL_DIR, "best_model.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(best_model_name)


if __name__ == "__main__":
    main()
