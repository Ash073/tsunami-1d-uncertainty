import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "uq_outputs"
CSV_FILE = OUTPUT_DIR / "uq_results_multivariate.csv"


def load_data(filename):
    deep_depth = []
    shallow_depth = []
    slope_end = []
    amplification = []

    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            deep_depth.append(float(row["deep_depth_m"]))
            shallow_depth.append(float(row["shallow_depth_m"]))
            slope_end.append(float(row["slope_end_km"]))
            amplification.append(float(row["amplification"]))

    return (
        np.array(deep_depth),
        np.array(shallow_depth),
        np.array(slope_end),
        np.array(amplification),
    )


def rank_data(values):
    """
    Compute ranks without requiring scipy.
    """
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def spearman_correlation(x, y):
    return pearson_correlation(rank_data(x), rank_data(y))


def standardize(values):
    std = np.std(values, ddof=1)

    if std == 0:
        return np.zeros_like(values)

    return (values - np.mean(values)) / std


def standardized_regression(X, y):
    """
    Multiple linear regression using standardized variables.

    Returns:
        coefficients, R^2
    """

    X_std = np.column_stack([
        standardize(X[:, 0]),
        standardize(X[:, 1]),
        standardize(X[:, 2]),
    ])

    y_std = standardize(y)

    # Add intercept
    X_design = np.column_stack([
        np.ones(len(X_std)),
        X_std
    ])

    coefficients = np.linalg.lstsq(
        X_design,
        y_std,
        rcond=None
    )[0]

    predictions = X_design @ coefficients

    ss_res = np.sum((y_std - predictions) ** 2)
    ss_tot = np.sum((y_std - np.mean(y_std)) ** 2)

    r_squared = 1.0 - ss_res / ss_tot

    return coefficients[1:], r_squared


def scatter_plot(x, y, xlabel, filename, title):
    plt.figure(figsize=(8, 5))

    plt.scatter(x, y)

    plt.xlabel(xlabel)
    plt.ylabel("Amplification at 80 km")
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
    OUTPUT_DIR / f"{filename}.png",
    dpi=300
)

    plt.close()


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n==========================================")
    print("       EXPLORATORY SENSITIVITY ANALYSIS")
    print("==========================================")

    (
        deep_depth,
        shallow_depth,
        slope_end,
        amplification
    ) = load_data(CSV_FILE)

    # --------------------------------------------------
    # Pearson correlations
    # --------------------------------------------------

    pearson_deep = pearson_correlation(
        deep_depth,
        amplification
    )

    pearson_shallow = pearson_correlation(
        shallow_depth,
        amplification
    )

    pearson_slope = pearson_correlation(
        slope_end,
        amplification
    )

    # --------------------------------------------------
    # Spearman correlations
    # --------------------------------------------------

    spearman_deep = spearman_correlation(
        deep_depth,
        amplification
    )

    spearman_shallow = spearman_correlation(
        shallow_depth,
        amplification
    )

    spearman_slope = spearman_correlation(
        slope_end,
        amplification
    )

    print("\nCorrelation analysis")
    print("------------------------------------------")

    print(
        f"Deep depth       "
        f"Pearson = {pearson_deep:+.5f}   "
        f"Spearman = {spearman_deep:+.5f}"
    )

    print(
        f"Shallow depth    "
        f"Pearson = {pearson_shallow:+.5f}   "
        f"Spearman = {spearman_shallow:+.5f}"
    )

    print(
        f"Slope end        "
        f"Pearson = {pearson_slope:+.5f}   "
        f"Spearman = {spearman_slope:+.5f}"
    )

    # --------------------------------------------------
    # Standardized regression
    # --------------------------------------------------

    X = np.column_stack([
        deep_depth,
        shallow_depth,
        slope_end
    ])

    coefficients, r_squared = standardized_regression(
        X,
        amplification
    )

    print("\nStandardized regression coefficients")
    print("------------------------------------------")

    print(
        f"Deep depth       : {coefficients[0]:+.5f}"
    )

    print(
        f"Shallow depth    : {coefficients[1]:+.5f}"
    )

    print(
        f"Slope end        : {coefficients[2]:+.5f}"
    )

    print(
        f"\nLinear-model R²  : {r_squared:.5f}"
    )

    # --------------------------------------------------
    # Preliminary ranking
    # --------------------------------------------------

    ranking = [
        ("Deep depth", abs(coefficients[0])),
        ("Shallow depth", abs(coefficients[1])),
        ("Slope end", abs(coefficients[2])),
    ]

    ranking.sort(
        key=lambda item: item[1],
        reverse=True
    )

    print("\nPreliminary parameter ranking")
    print("------------------------------------------")

    for i, (name, value) in enumerate(
        ranking,
        start=1
    ):
        print(
            f"{i}. {name:<15} "
            f"| standardized coefficient magnitude = {value:.5f}"
        )

    # --------------------------------------------------
    # Save scatter plots
    # --------------------------------------------------

    scatter_plot(
        deep_depth,
        amplification,
        "Deep-water depth (m)",
        "scatter_deep_depth",
        "Deep-Water Depth vs Wave Amplification"
    )

    scatter_plot(
        shallow_depth,
        amplification,
        "Shallow-water depth (m)",
        "scatter_shallow_depth",
        "Shallow-Water Depth vs Wave Amplification"
    )

    scatter_plot(
        slope_end,
        amplification,
        "Slope-end position (km)",
        "scatter_slope_end",
        "Slope-End Position vs Wave Amplification"
    )

    print("\nPlots saved to:")
    print("uq_outputs/scatter_deep_depth.png")
    print("uq_outputs/scatter_shallow_depth.png")
    print("uq_outputs/scatter_slope_end.png")

    print("\n==========================================")
    print("              ANALYSIS COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    main()