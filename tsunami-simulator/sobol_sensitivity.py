import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from tsunami_uncertainty_multi import run_simulation


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "uq_outputs"

PARAMETER_NAMES = [
    "deep_depth_m",
    "shallow_depth_m",
    "slope_end_km",
]

LOWER_BOUNDS = np.array([190.0, 45.0, 75.0])
UPPER_BOUNDS = np.array([210.0, 55.0, 85.0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Corrected Sobol global sensitivity analysis for the 1D tsunami model."
    )
    parser.add_argument(
        "--base-samples",
        type=int,
        default=64,
        help="Number of samples N. Must be a power of two.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for one 2D-dimensional scrambled Sobol sequence.",
    )
    return parser.parse_args()


def validate_sample_count(n):
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(
            "--base-samples must be a power of two and at least 2 "
            "(for example: 8, 16, 32, 64)."
        )


def scale_unit_samples(unit_samples):
    return LOWER_BOUNDS + unit_samples * (UPPER_BOUNDS - LOWER_BOUNDS)


def generate_saltelli_design(n, seed):
    """
    Generate a standard Saltelli-compatible A/B design from ONE
    2D-dimensional Sobol sequence.

    For first- and total-order Sobol indices we evaluate:
        A, B, and A_Bi for i=1..D

    Hence the required model evaluations are:
        N * (D + 2)
    """
    d = len(PARAMETER_NAMES)

    sampler = qmc.Sobol(
        d=2 * d,
        scramble=True,
        seed=seed,
    )

    unit = sampler.random_base2(m=int(np.log2(n)))

    A_unit = unit[:, :d]
    B_unit = unit[:, d:]

    A = scale_unit_samples(A_unit)
    B = scale_unit_samples(B_unit)

    hybrids = []
    for i in range(d):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        hybrids.append(A_Bi)

    return A, B, hybrids


def model(point):
    return run_simulation(
        deep_depth=point[0],
        shallow_depth=point[1],
        slope_end=point[2] * 1000.0,
    )


def evaluate_design(A, B, hybrids):
    n = len(A)
    d = len(hybrids)
    total_evaluations = n * (d + 2)

    outputs_A = []
    outputs_B = []
    outputs_hybrid = []

    completed = 0

    print("\n==========================================")
    print("      CORRECTED SOBOL MODEL EVALUATION")
    print("==========================================")
    print(f"Total model evaluations : {total_evaluations}")
    print("------------------------------------------")

    for i, point in enumerate(A, start=1):
        y = model(point)
        outputs_A.append(y)
        completed += 1
        print(
            f"A   {i:03d}/{n} -> "
            f"deep={point[0]:.3f} m, "
            f"shallow={point[1]:.3f} m, "
            f"slope_end={point[2]:.3f} km -> "
            f"amplification={y:.5f}x"
        )

    for i, point in enumerate(B, start=1):
        y = model(point)
        outputs_B.append(y)
        completed += 1
        print(
            f"B   {i:03d}/{n} -> "
            f"deep={point[0]:.3f} m, "
            f"shallow={point[1]:.3f} m, "
            f"slope_end={point[2]:.3f} km -> "
            f"amplification={y:.5f}x"
        )

    for parameter_index, matrix in enumerate(hybrids):
        name = PARAMETER_NAMES[parameter_index]
        current = []

        for i, point in enumerate(matrix, start=1):
            y = model(point)
            current.append(y)
            completed += 1
            print(
                f"AB({name}) {i:03d}/{n} -> "
                f"deep={point[0]:.3f} m, "
                f"shallow={point[1]:.3f} m, "
                f"slope_end={point[2]:.3f} km -> "
                f"amplification={y:.5f}x"
            )

        outputs_hybrid.append(np.asarray(current, dtype=float))

    print("------------------------------------------")
    print(f"Completed evaluations: {completed}")

    return (
        np.asarray(outputs_A, dtype=float),
        np.asarray(outputs_B, dtype=float),
        outputs_hybrid,
    )


def calculate_sobol_indices(outputs_A, outputs_B, outputs_hybrid):
    """
    First-order: Saltelli covariance estimator.
    Total-order: Jansen estimator.

    The two A/B matrices are generated from one common 2D Sobol design.
    """
    y_a = np.asarray(outputs_A, dtype=float)
    y_b = np.asarray(outputs_B, dtype=float)

    # Use both independent base matrices to estimate output variance.
    variance = np.var(np.concatenate([y_a, y_b]), ddof=1)

    if variance <= 0.0:
        raise RuntimeError("Model-output variance is non-positive.")

    first_order = []
    total_order = []

    for y_ab_i in outputs_hybrid:
        y_ab_i = np.asarray(y_ab_i, dtype=float)

        # Saltelli first-order estimator.
        s_i = np.mean(y_b * (y_ab_i - y_a)) / variance

        # Jansen total-order estimator.
        st_i = 0.5 * np.mean((y_a - y_ab_i) ** 2) / variance

        first_order.append(s_i)
        total_order.append(st_i)

    return np.asarray(first_order), np.asarray(total_order), variance


def save_model_runs(A, B, hybrids, outputs_A, outputs_B, outputs_hybrid):
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / "sobol_corrected_model_runs.csv"

    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_type",
            "parameter_replaced",
            "sample_index",
            *PARAMETER_NAMES,
            "amplification",
        ])

        for i, (point, output) in enumerate(zip(A, outputs_A), start=1):
            writer.writerow(["A", "", i, *point, output])

        for i, (point, output) in enumerate(zip(B, outputs_B), start=1):
            writer.writerow(["B", "", i, *point, output])

        for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
            for i, (point, output) in enumerate(
                zip(hybrids[parameter_index], outputs_hybrid[parameter_index]),
                start=1,
            ):
                writer.writerow([
                    "AB",
                    parameter_name,
                    i,
                    *point,
                    output,
                ])

    return filename


def save_summary(first_order, total_order, variance):
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / "sobol_corrected_sensitivity_summary.csv"

    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "parameter",
            "first_order_Si",
            "total_order_STi",
            "interaction_gap_ST_minus_Si",
        ])

        for name, s_i, st_i in zip(
            PARAMETER_NAMES,
            first_order,
            total_order,
        ):
            writer.writerow([
                name,
                s_i,
                st_i,
                st_i - s_i,
            ])

        writer.writerow([])
        writer.writerow(["output_variance", variance])

    return filename


def save_plot(first_order, total_order):
    labels = ["Deep depth", "Shallow depth", "Slope end"]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, first_order, width, label="First-order $S_i$")
    plt.bar(x + width / 2, total_order, width, label="Total-order $S_{T_i}$")
    plt.xticks(x, labels)
    plt.ylabel("Sobol sensitivity index")
    plt.title("Global Sensitivity of Tsunami Amplification")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filename = OUTPUT_DIR / "sobol_corrected_sensitivity.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename


def print_results(first_order, total_order, variance):
    print("\n==========================================")
    print("        CORRECTED SOBOL RESULTS")
    print("==========================================")
    print("Parameter                 S_i        S_Ti")
    print("------------------------------------------")

    for name, s_i, st_i in zip(PARAMETER_NAMES, first_order, total_order):
        print(f"{name:<24} {s_i:>8.5f}   {st_i:>8.5f}")

    print("------------------------------------------")
    print(f"Output variance          : {variance:.8f}")

    ranking = sorted(
        zip(PARAMETER_NAMES, total_order),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nRanking by total-order sensitivity")
    print("------------------------------------------")
    for i, (name, value) in enumerate(ranking, start=1):
        print(f"{i}. {name:<20} ST = {value:.5f}")

    print("\nSanity checks")
    print("------------------------------------------")
    first_valid = np.all((first_order >= -0.1) & (first_order <= 1.1))
    total_valid = np.all((total_order >= -0.1) & (total_order <= 1.1))
    total_sum = np.sum(first_order)

    print(f"First-order indices in practical range: {first_valid}")
    print(f"Total-order indices in practical range : {total_valid}")
    print(f"Sum of first-order indices             : {total_sum:.5f}")

    if np.any(first_order < 0) or np.any(first_order > 1) or np.any(total_order < 0) or np.any(total_order > 1):
        print("WARNING: some indices are outside [0, 1].")
        print("Increase the Sobol sample count before treating the estimates as final.")
    print("==========================================")


def main():
    args = parse_args()
    validate_sample_count(args.base_samples)
    OUTPUT_DIR.mkdir(exist_ok=True)

    d = len(PARAMETER_NAMES)
    total_evaluations = args.base_samples * (d + 2)

    print("\n==========================================")
    print("       CORRECTED SOBOL SENSITIVITY")
    print("==========================================")
    print(f"Base samples         : {args.base_samples}")
    print(f"Parameters           : {d}")
    print(f"Model evaluations    : {total_evaluations}")
    print(f"Random seed          : {args.seed}")
    print("------------------------------------------")
    print("Parameter ranges:")
    print("  Deep depth         : 190–210 m")
    print("  Shallow depth      : 45–55 m")
    print("  Slope end          : 75–85 km")

    A, B, hybrids = generate_saltelli_design(args.base_samples, args.seed)

    outputs_A, outputs_B, outputs_hybrid = evaluate_design(A, B, hybrids)

    first_order, total_order, variance = calculate_sobol_indices(
        outputs_A,
        outputs_B,
        outputs_hybrid,
    )

    model_runs_file = save_model_runs(
        A, B, hybrids, outputs_A, outputs_B, outputs_hybrid
    )
    summary_file = save_summary(first_order, total_order, variance)
    plot_file = save_plot(first_order, total_order)

    print_results(first_order, total_order, variance)

    print("\nOutputs saved to:")
    print(model_runs_file)
    print(summary_file)
    print(plot_file)


if __name__ == "__main__":
    main()
