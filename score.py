import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_ROWS = 12000

EXPECTED_IDS = {
    f"TE-{index:06d}"
    for index in range(1, EXPECTED_ROWS + 1)
}

DECEMBER_DATES = pd.date_range(
    "2025-12-01",
    "2025-12-31",
    freq="D"
)


def validate_predictions(df):
    expected_columns = ["load_id", "predicted_rate"]

    assert list(df.columns) == expected_columns
    assert len(df) == EXPECTED_ROWS
    assert df["load_id"].notna().all()
    assert df["load_id"].nunique() == EXPECTED_ROWS

    submitted_ids = set(df["load_id"].astype(str))

    assert submitted_ids == EXPECTED_IDS

    rates = pd.to_numeric(
        df["predicted_rate"],
        errors="coerce"
    )

    assert rates.notna().all()
    assert np.isfinite(rates).all()
    assert (rates > 0).all()

    print("Validation predictions: PASSED")


def validate_december(df):
    expected_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]

    assert list(df.columns) == expected_columns
    assert len(df) == 31

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    assert df["date"].notna().all()
    assert df["date"].is_unique
    assert set(df["date"]) == set(DECEMBER_DATES)

    assert df["pickup"].eq("Lexington").all()
    assert df["delivery"].eq("Fort Wayne").all()
    assert np.isclose(df["distance"], 360).all()
    assert df["equipment"].eq("Dry Van").all()
    assert np.isclose(df["weight"], 32000).all()

    rates = pd.to_numeric(
        df["predicted_rate"],
        errors="coerce"
    )

    assert rates.notna().all()
    assert np.isfinite(rates).all()
    assert (rates > 0).all()

    print("December predictions: PASSED")

    return df.sort_values("date")


def create_chart(df, output_path):
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)

    ax.plot(
        df["date"],
        df["predicted_rate"],
        linewidth=2.6,
        marker="o",
        markersize=3.2,
    )

    ax.set_title(
        "Candidate: December 2025 Predicted Load Rate",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_ylabel("Predicted rate ($)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.3)

    ax.tick_params(axis="x", rotation=35)

    ax.text(
        0,
        -0.35,
        "Lexington to Fort Wayne | 360 miles | Dry Van | "
        "32,000 lb | Only date changes",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print("December chart created:", output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True
    )

    parser.add_argument(
        "--december-predictions",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="scorer_results"
    )

    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    december = pd.read_csv(args.december_predictions)

    validate_predictions(predictions)
    december = validate_december(december)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_chart(
        december,
        output_dir / "candidate_december.png"
    )

    print("\nAll output checks completed successfully.")


if __name__ == "__main__":
    main()
