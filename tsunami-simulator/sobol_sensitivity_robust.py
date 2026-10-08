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
        description=(
            "Robust global Sobol sensitivity analysis using multiple "
            "independent scrambled Sobol replicates."
        )
    )
    parser.add_argument(
        "--base-samples",
        type=int,
        default=64,
        help="Samples per independent replicate; must be a power of two.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=4,
        help="Number of independent scrambled Sobol replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed; replicate r uses seed + r.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the estimator self-test on a known analytic model and exit.",
    )
    return parser.parse_args()


def validate_sample_count(n):
    if n < 8 or (n & (n - 1)) != 0:
        raise ValueError(
            "--base-samples must be a power of two and at least 8 "
            "(for example: 8, 16, 32, 64)."
        )


def scale_unit_samples(unit_samples):
    return LOWER_BOUNDS + unit_samples * (UPPER_BOUNDS - LOWER_BOUNDS)


def generate_design(n, seed):
    """Generate A, B, and A_Bi matrices from one scrambled 2D Sobol design."""
    d = len(PARAMETER_NAMES)

    sampler = qmc.Sobol(
        d=2 * d,
        scramble=True,
        seed=seed,
    )

    unit = sampler.random_base2(m=int(np.log2(n)))

    A = scale_unit_samples(unit[:, :d])
    B = scale_unit_samples(unit[:, d:])

    hybrids = []
    for i in range(d):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        hybrids.append(A_Bi)

    return A, B, hybrids


def model(point):
    return run_simulation(
        deep_depth=float(point[0]),
        shallow_depth=float(point[1]),
        slope_end=float(point[2]) * 1000.0,
    )


def evaluate_matrix(matrix, label, replicate_index):
    outputs = []
    n = len(matrix)

    for i, point in enumerate(matrix, start=1):
        y = model(point)
        outputs.append(y)
        print(
            f"R{replicate_index:02d} {label:<16} {i:03d}/{n} -> "
            f"deep={point[0]:.3f} m, "
            f"shallow={point[1]:.3f} m, "
            f"slope_end={point[2]:.3f} km -> "
            f"amplification={y:.5f}x"
        )

    return np.asarray(outputs, dtype=float)


def calculate_indices(y_a, y_b, y_ab):
    """Saltelli first-order + Jansen total-order estimators for one replicate."""
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    y_ab = np.asarray(y_ab, dtype=float)

    # Pool A and B only for a stable estimate of total output variance.
    variance = np.var(np.concatenate([y_a, y_b]), ddof=1)

    if not np.isfinite(variance) or variance <= 0.0:
        raise RuntimeError("Non-positive or invalid output variance.")

    first_order = np.empty(y_ab.shape[0] if y_ab.ndim == 2 else 1)
    # y_ab is expected to have shape (D, N).
    if y_ab.ndim != 2:
        raise ValueError("y_ab must have shape (D, N).")

    d = y_ab.shape[0]
    first_order = np.empty(d)
    total_order = np.empty(d)

    for i in range(d):
        # Saltelli first-order estimator:
        # E[f(B) * (f(A_Bi) - f(A))] / Var(f)
        first_order[i] = (
            np.mean(y_b * (y_ab[i] - y_a)) / variance
        )

        # Jansen total-order estimator:
        # E[(f(A) - f(A_Bi))^2] / (2 Var(f))
        total_order[i] = (
            0.5 * np.mean((y_a - y_ab[i]) ** 2) / variance
        )

    return first_order, total_order, variance


def analytic_self_test(seed=12345):
    """Check the estimator against Y = X1 + 2X2 + 0.5X3 on U(0,1)^3."""
    n = 8192
    d = 3
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    unit = sampler.random_base2(m=int(np.log2(n)))
    A = unit[:, :d]
    B = unit[:, d:]

    hybrids = []
    for i in range(d):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        hybrids.append(A_Bi)

    def f(X):
        return X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 2]

    y_a = f(A)
    y_b = f(B)
    y_ab = np.vstack([f(H) for H in hybrids])

    s, st, _ = calculate_indices(y_a, y_b, y_ab)

    # For independent U(0,1), Var(aX) = a^2/12.
    weights = np.array([1.0, 4.0, 0.25])
    expected = weights / weights.sum()

    print("\n==========================================")
    print("         SOBOL ESTIMATOR SELF-TEST")
    print("==========================================")
    print("Known model: Y = X1 + 2X2 + 0.5X3")
    print("Parameter                 Expected     Estimated")
    print("------------------------------------------")

    for i, name in enumerate(PARAMETER_NAMES):
        print(
            f"{name:<24} {expected[i]:>10.5f}     {s[i]:>10.5f}"
        )

    max_error = float(np.max(np.abs(s - expected)))
    print("------------------------------------------")
    print(f"Maximum first-order error : {max_error:.6f}")
    print(f"Sum of estimated S_i     : {np.sum(s):.6f}")
    print(f"All S_i close to theory  : {max_error < 0.01}")
    print(f"All S_Ti close to theory : {np.max(np.abs(st - expected)) < 0.01}")
    print("==========================================")

    if max_error >= 0.01 or np.max(np.abs(st - expected)) >= 0.01:
        raise SystemExit("Self-test failed: estimator implementation needs review.")


def save_replicate_results(records):
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / "sobol_robust_replicates.csv"

    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "replicate",
            "seed",
            "parameter",
            "first_order_Si",
            "total_order_STi",
            "output_variance",
        ])

        for record in records:
            for name, s_i, st_i in zip(
                PARAMETER_NAMES,
                record["first_order"],
                record["total_order"],
            ):
                writer.writerow([
                    record["replicate"],
                    record["seed"],
                    name,
                    s_i,
                    st_i,
                    record["variance"],
                ])

    return filename


def save_summary(records):
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / "sobol_robust_summary.csv"

    first_matrix = np.vstack([r["first_order"] for r in records])
    total_matrix = np.vstack([r["total_order"] for r in records])

    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "parameter",
            "mean_Si",
            "std_Si",
            "ci95_low_Si",
            "ci95_high_Si",
            "mean_STi",
            "std_STi",
            "ci95_low_STi",
            "ci95_high_STi",
        ])

        for i, name in enumerate(PARAMETER_NAMES):
            s = first_matrix[:, i]
            st = total_matrix[:, i]
            s_mean = np.mean(s)
            st_mean = np.mean(st)
            s_se = np.std(s, ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
            st_se = np.std(st, ddof=1) / np.sqrt(len(st)) if len(st) > 1 else 0.0

            writer.writerow([
                name,
                s_mean,
                np.std(s, ddof=1),
                s_mean - 1.96 * s_se,
                s_mean + 1.96 * s_se,
                st_mean,
                np.std(st, ddof=1),
                st_mean - 1.96 * st_se,
                st_mean + 1.96 * st_se,
            ])

        writer.writerow([])
        writer.writerow([
            "mean_sum_first_order",
            np.sum(np.mean(first_matrix, axis=0)),
        ])

    return filename


def save_plot(records):
    first_matrix = np.vstack([r["first_order"] for r in records])
    total_matrix = np.vstack([r["total_order"] for r in records])

    labels = ["Deep depth", "Shallow depth", "Slope end"]
    x = np.arange(len(labels))

    s_mean = np.mean(first_matrix, axis=0)
    st_mean = np.mean(total_matrix, axis=0)
    s_err = 1.96 * np.std(first_matrix, axis=0, ddof=1) / np.sqrt(len(records))
    st_err = 1.96 * np.std(total_matrix, axis=0, ddof=1) / np.sqrt(len(records))

    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2,
        s_mean,
        width,
        yerr=s_err,
        capsize=4,
        label="First-order $S_i$",
    )
    plt.bar(
        x + width / 2,
        st_mean,
        width,
        yerr=st_err,
        capsize=4,
        label="Total-order $S_{T_i}$",
    )
    plt.xticks(x, labels)
    plt.ylabel("Sobol sensitivity index")
    plt.title("Robust Sobol Sensitivity of Tsunami Amplification")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filename = OUTPUT_DIR / "sobol_robust_sensitivity.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename


def print_results(records):
    first_matrix = np.vstack([r["first_order"] for r in records])
    total_matrix = np.vstack([r["total_order"] for r in records])

    s_mean = np.mean(first_matrix, axis=0)
    st_mean = np.mean(total_matrix, axis=0)
    s_std = np.std(first_matrix, axis=0, ddof=1)
    st_std = np.std(total_matrix, axis=0, ddof=1)

    s_se = s_std / np.sqrt(len(records))
    st_se = st_std / np.sqrt(len(records))

    print("\n==========================================")
    print("       ROBUST SOBOL RESULTS")
    print("==========================================")
    print("Parameter                 mean Si   mean STi")
    print("------------------------------------------")
    for i, name in enumerate(PARAMETER_NAMES):
        print(f"{name:<24} {s_mean[i]:>8.5f}   {st_mean[i]:>8.5f}")

    print("\n95% replicate-confidence intervals")
    print("------------------------------------------")
    for i, name in enumerate(PARAMETER_NAMES):
        print(
            f"{name:<24} "
            f"Si=[{s_mean[i]-1.96*s_se[i]:.5f}, {s_mean[i]+1.96*s_se[i]:.5f}]  "
            f"STi=[{st_mean[i]-1.96*st_se[i]:.5f}, {st_mean[i]+1.96*st_se[i]:.5f}]"
        )

    print("\nRanking by mean total-order sensitivity")
    print("------------------------------------------")
    ranking = sorted(
        zip(PARAMETER_NAMES, st_mean),
        key=lambda item: item[1],
        reverse=True,
    )
    for i, (name, value) in enumerate(ranking, start=1):
        print(f"{i}. {name:<20} ST = {value:.5f}")

    print("\nConvergence diagnostics")
    print("------------------------------------------")
    print(f"Sum of mean first-order indices : {np.sum(s_mean):.5f}")
    print(
        f"All mean STi >= mean Si       : "
        f"{bool(np.all(st_mean >= s_mean))}"
    )
    print(
        f"Largest mean |STi - Si|       : "
        f"{np.max(np.abs(st_mean - s_mean)):.5f}"
    )
    print("==========================================")


def run_analysis(args):
    d = len(PARAMETER_NAMES)
    per_replicate_evals = args.base_samples * (d + 2)
    total_evals = per_replicate_evals * args.replicates

    print("\n==========================================")
    print("       ROBUST SOBOL SENSITIVITY")
    print("==========================================")
    print(f"Base samples / replicate : {args.base_samples}")
    print(f"Replicates               : {args.replicates}")
    print(f"Parameters               : {d}")
    print(f"Evaluations / replicate  : {per_replicate_evals}")
    print(f"Total model evaluations  : {total_evals}")
    print(f"Base seed                : {args.seed}")
    print("------------------------------------------")
    print("Parameter ranges:")
    print("  Deep depth             : 190–210 m")
    print("  Shallow depth          : 45–55 m")
    print("  Slope end              : 75–85 km")

    records = []

    for replicate in range(1, args.replicates + 1):
        seed = args.seed + replicate - 1
        print("\n------------------------------------------")
        print(f"REPLICATE {replicate}/{args.replicates}  (seed={seed})")
        print("------------------------------------------")

        A, B, hybrids = generate_design(args.base_samples, seed)

        y_a = evaluate_matrix(A, "A", replicate)
        y_b = evaluate_matrix(B, "B", replicate)

        y_hybrid = []
        for parameter_index, matrix in enumerate(hybrids):
            label = f"AB({PARAMETER_NAMES[parameter_index]})"
            y_hybrid.append(
                evaluate_matrix(matrix, label, replicate)
            )

        y_hybrid = np.vstack(y_hybrid)

        s, st, variance = calculate_indices(y_a, y_b, y_hybrid)

        records.append({
            "replicate": replicate,
            "seed": seed,
            "first_order": s,
            "total_order": st,
            "variance": variance,
        })

        print("\nReplicate result")
        for name, s_i, st_i in zip(PARAMETER_NAMES, s, st):
            print(f"  {name:<22} Si={s_i:.5f}  STi={st_i:.5f}")

    replicate_file = save_replicate_results(records)
    summary_file = save_summary(records)
    plot_file = save_plot(records)

    print_results(records)

    print("\nOutputs saved to:")
    print(replicate_file)
    print(summary_file)
    print(plot_file)


def main():
    args = parse_args()

    if args.self_test:
        analytic_self_test(args.seed)
        return

    if args.replicates < 2:
        raise ValueError("Use at least 2 independent replicates for robust uncertainty estimates.")

    validate_sample_count(args.base_samples)
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_analysis(args)


if __name__ == "__main__":
    main()
