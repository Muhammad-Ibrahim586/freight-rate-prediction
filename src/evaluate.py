
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(model, X_val, y_val):
    """
    Evaluate a trained regression model on validation data.

    Returns:
        dict: MAE, RMSE, and R-squared metrics.
    """

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = mean_squared_error(y_val, predictions) ** 0.5
    r2 = r2_score(y_val, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }
