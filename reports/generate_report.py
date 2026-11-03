
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "assessment_report.pdf"
CHART = ROOT / "scorer_results" / "candidate_december.png"
TRAIN_DATA = ROOT / "data" / "train-test.csv"

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=25,
    alignment=TA_CENTER,
    spaceAfter=8,
    textColor=colors.HexColor("#17365D"),
))

styles.add(ParagraphStyle(
    name="Subtitle",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"),
    spaceAfter=20,
))

styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading2"],
    fontSize=13,
    leading=17,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=12,
    spaceAfter=6,
))

styles.add(ParagraphStyle(
    name="BodyTextCustom",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13,
    alignment=TA_LEFT,
    spaceAfter=7,
))

styles.add(ParagraphStyle(
    name="SmallNote",
    parent=styles["Normal"],
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#555555"),
))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(20 * mm, 12 * mm, "Freight Rate Prediction Assessment")
    canvas.drawRightString(
        A4[0] - 20 * mm,
        12 * mm,
        f"Page {doc.page}"
    )
    canvas.restoreState()


def paragraph(text):
    return Paragraph(text, styles["BodyTextCustom"])


def section(title):
    return Paragraph(title, styles["Section"])


def make_table(data, widths):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def build_report():
    if not CHART.exists():
        raise FileNotFoundError(f"December chart not found: {CHART}")

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_DATA}")

    df = pd.read_csv(TRAIN_DATA)

    story = []

    # Title
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "FREIGHT RATE PREDICTION",
        styles["ReportTitle"]
    ))
    story.append(Paragraph(
        "Machine Learning Assessment Report",
        styles["Subtitle"]
    ))

    story.append(paragraph(
        "<b>Project:</b> Freight Rate Prediction Challenge<br/>"
        "<b>Prediction target:</b> Posted freight rate<br/>"
        "<b>Training period:</b> January–August 2025<br/>"
        "<b>Validation period:</b> September–October 2025"
    ))

    story.append(section("Executive Summary"))
    story.append(paragraph(
        "This project develops a supervised machine learning pipeline "
        "to estimate freight load rates. The training dataset contains "
        "48,000 records covering January through October 2025. "
        "Two tree-based regression models, Random Forest and Extra Trees, "
        "were evaluated using a chronological validation strategy. "
        "Extra Trees achieved the best validation performance, with a "
        "mean absolute error (MAE) of $150.82 and an R-squared value "
        "of 0.8175. The selected model was subsequently used to generate "
        "predictions for the 12,000 validation loads and 31 December "
        "scenario records."
    ))

    story.append(section("1. Problem Definition and Dataset"))
    story.append(paragraph(
        "The objective is to predict the posted freight rate using "
        "available shipment characteristics. The training dataset "
        "contains 48,000 observations and 14 columns, including "
        "origin and destination, shipment distance, equipment type, "
        "weight, date, market indicators, and the target variable "
        "<i>posted_rate</i>."
    ))
    story.append(paragraph(
        "The target variable has a mean of $2,373.98, a median of "
        "$2,030.76, and a standard deviation of $1,486.49. Its "
        "distribution is right-skewed, with a small number of "
        "high-value observations."
    ))

    story.append(section("2. Exploratory Data Analysis"))
    story.append(paragraph(
        "Exploratory analysis identified several relevant patterns:"
    ))
    story.append(paragraph(
        "<b>Distance:</b> Distance has a strong positive correlation "
        "with posted rate (r = 0.9085), making it an important "
        "predictive feature."
    ))
    story.append(paragraph(
        "<b>Equipment:</b> Reefer loads have the highest average "
        "posted rate ($2,553.64), followed by Flatbed ($2,445.09) "
        "and Dry Van ($2,271.55)."
    ))
    story.append(paragraph(
        "<b>Weight:</b> Weight has a weak linear correlation with "
        "posted rate (r = 0.0348). It was retained as a candidate "
        "feature because tree-based models can capture nonlinear "
        "relationships."
    ))
    story.append(paragraph(
        "<b>Data quality:</b> Weight contains 300 missing values and "
        "292 negative values. Market index contains 374 missing values. "
        "The dataset contains no duplicate rows."
    ))
    story.append(paragraph(
        "The interquartile-range analysis identified 260 potential "
        "target outliers (0.54%). These observations were retained "
        "because high rates may represent legitimate long-distance "
        "shipments."
    ))

    story.append(section("3. Preprocessing and Feature Engineering"))
    story.append(paragraph(
        "Pickup, delivery, and equipment were treated as categorical "
        "features. Distance, weight, and date-derived variables were "
        "treated as numerical features. The date column was transformed "
        "into year, month, day, day of week, and day of year."
    ))
    story.append(paragraph(
        "Negative shipment weights were treated as missing values. "
        "Median imputation was applied to numerical features, while "
        "categorical missing values were imputed using the most frequent "
        "category. Standard scaling was applied to numerical features, "
        "and one-hot encoding was used for categorical variables. "
        "Unknown categories were handled using "
        "<i>handle_unknown='ignore'</i>."
    ))
    story.append(paragraph(
        "The final model uses features available in both the training "
        "and December input datasets. Market index, quote signal, and "
        "geographical coordinates were excluded because the December "
        "inputs do not provide them."
    ))

    story.append(section("4. Training and Validation Strategy"))
    story.append(paragraph(
        "A chronological split was used to approximate deployment "
        "on future freight loads. Records dated before 1 September "
        "2025 were assigned to training, while September and October "
        "records were reserved for validation."
    ))

    split_data = [
        ["Dataset", "Date range", "Records"],
        ["Training", "Jan–Aug 2025", "38,477"],
        ["Validation", "Sep–Oct 2025", "9,523"],
    ]
    story.append(make_table(split_data, [42 * mm, 65 * mm, 35 * mm]))

    story.append(Spacer(1, 3 * mm))
    story.append(paragraph(
        "Both models used 80 estimators, a maximum depth of 20, "
        "a minimum of five samples per leaf, and random_state=42. "
        "The models were trained through preprocessing and regression "
        "pipelines to keep preprocessing fitted on training data only."
    ))

    story.append(section("5. Model Evaluation and Selection"))
    story.append(paragraph(
        "The models were compared using MAE, root mean squared error "
        "(RMSE), and the coefficient of determination (R-squared). "
        "MAE was selected as the primary model-selection criterion."
    ))

    metrics_data = [
        ["Model", "MAE ($)", "RMSE ($)", "R²"],
        ["Random Forest", "172.33", "658.04", "0.8141"],
        ["Extra Trees", "150.82", "651.87", "0.8175"],
    ]
    story.append(make_table(
        metrics_data,
        [45 * mm, 35 * mm, 35 * mm, 25 * mm]
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(paragraph(
        "Extra Trees achieved the lowest MAE and RMSE and the highest "
        "R-squared value. It was therefore selected as the final model. "
        "Its validation MAE of $150.82 represents the average absolute "
        "difference between predicted and actual posted rates on the "
        "chronological validation set."
    ))

    story.append(section("6. Prediction Outputs"))
    story.append(paragraph(
        "The selected Extra Trees model generated predictions for all "
        "12,000 validation records. The predictions were written to "
        "<i>validation_predictions.csv</i> using the supplied template "
        "structure and load identifiers."
    ))
    story.append(paragraph(
        "The model also generated predictions for 31 December 2025 "
        "scenario records. These records represent the same route "
        "(Lexington to Fort Wayne), distance (360 miles), equipment "
        "(Dry Van), and weight (32,000 lb), with the date varying "
        "across the month."
    ))

    december_data = [
        ["Statistic", "Predicted rate ($)"],
        ["Mean", "859.66"],
        ["Standard deviation", "3.39"],
        ["Minimum", "854.57"],
        ["Median", "859.86"],
        ["Maximum", "866.99"],
    ]
    story.append(make_table(
        december_data,
        [65 * mm, 55 * mm]
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(paragraph(
        "The predictions range from $854.57 to $866.99. Their relatively "
        "small variation reflects the fact that shipment characteristics "
        "remain constant while the date changes."
    ))

    story.append(Spacer(1, 3 * mm))
    chart = Image(str(CHART), width=165 * mm, height=73 * mm)
    chart.hAlign = "CENTER"
    story.append(chart)
    story.append(Paragraph(
        "Figure 1. December 2025 predicted rates for the Lexington–"
        "Fort Wayne shipment scenario.",
        styles["SmallNote"]
    ))

    story.append(section("7. Limitations"))
    story.append(paragraph(
        "The validation results measure performance on September–"
        "October 2025 records and may not fully represent performance "
        "under future market conditions. Market index, quote signal, "
        "and coordinates were excluded to maintain feature consistency "
        "with the December inputs. The December scenario contains only "
        "one route and should not be interpreted as a general forecast "
        "of market-wide freight rates."
    ))
    story.append(paragraph(
        "The local output-check script verifies prediction-file "
        "structure and validity and generates the December chart. "
        "It is a reconstructed utility, not a verified official "
        "Spotter scoring implementation."
    ))

    story.append(section("Conclusion"))
    story.append(paragraph(
        "The project produced a reproducible freight-rate prediction "
        "pipeline using chronological validation and tree-based "
        "regression. Extra Trees was the strongest of the two evaluated "
        "models, achieving a validation MAE of $150.82. The final "
        "pipeline generated complete validation and December prediction "
        "files. Further improvement could involve route-level features, "
        "additional temporal data, and systematic hyperparameter tuning."
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Freight Rate Prediction Assessment Report",
        author="Assessment Candidate",
    )

    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer
    )

    print("Report generated:", OUTPUT)


if __name__ == "__main__":
    build_report()
