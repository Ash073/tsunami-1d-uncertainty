import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tsunami_uncertainty_multi import run_simulation


# ============================================================
# Morris experiment settings
# ============================================================

PARAMETER_NAMES = [
    "deep_depth_m",
    "shallow_depth_m",
    "slope_end_km",
]

LOWER_BOUNDS = np.array([
    190.0,
    45.0,
    75.0,
])

UPPER_BOUNDS = np.array([
    210.0,
    55.0,
    85.0,
])

NUM_LEVELS = 6
DEFAULT_TRAJECTORIES = 64
RANDOM_SEED = 12345

OUTPUT_DIR = (
    Path(__file__).resolve().parent / "uq_outputs"
)


# ============================================================
# Morris design construction
# ============================================================


def build_morris_trajectory(rng, num_levels, num_parameters):
    """
    Construct one Morris trajectory in normalized [0, 1]^k space.

    A trajectory contains k+1 points. Each successive point differs
    in exactly one parameter by the Morris step size.
    """

    delta = num_levels / (2.0 * (num_levels - 1.0))

    # For even p, p/(2(p-1)) is compatible with the p-level grid
    # when the starting point is restricted to valid grid locations.
    grid_step = 1.0 / (num_levels - 1.0)

    # Random direction and a valid starting grid point for each
    # parameter. The starting point is chosen so that the subsequent
    # +/- delta move always remains inside [0, 1].
    directions = rng.choice(
        np.array([-1.0, 1.0]),
        size=num_parameters,
    )

    x0 = np.zeros(num_parameters)

    for parameter_index, direction in enumerate(directions):
        if direction > 0:
            valid_starts = np.arange(
                0.0,
                1.0 - delta + 0.5 * grid_step,
                grid_step,
            )
        else:
            valid_starts = np.arange(
                delta,
                1.0 + 0.5 * grid_step,
                grid_step,
            )

        x0[parameter_index] = rng.choice(valid_starts)

    # Random order in which parameters change.
    order = rng.permutation(num_parameters)

    trajectory = [x0.copy()]
    current = x0.copy()

    for parameter_index in order:
        step = directions[parameter_index] * delta
        current = current.copy()
        current[parameter_index] += step
        trajectory.append(current)

    return np.array(trajectory), order


# ============================================================
# Parameter scaling
# ============================================================


def scale_parameters(normalized_values):
    """Map normalized [0, 1] inputs to physical parameter values."""

    return (
        LOWER_BOUNDS
        + normalized_values
        * (UPPER_BOUNDS - LOWER_BOUNDS)
    )


# ============================================================
# One model evaluation
# ============================================================


def evaluate_model(physical_parameters):
    """
    Run the tsunami solver for one Morris sample.

    Parameter order:
        deep depth [m]
        shallow depth [m]
        slope end [km]
    """

    deep_depth = float(physical_parameters[0])
    shallow_depth = float(physical_parameters[1])
    slope_end_km = float(physical_parameters[2])

    return run_simulation(
        deep_depth=deep_depth,
        shallow_depth=shallow_depth,
        slope_end=slope_end_km * 1000.0,
    )


# ============================================================
# Main Morris experiment
# ============================================================


def run_morris(num_trajectories):

    rng = np.random.default_rng(RANDOM_SEED)

    num_parameters = len(PARAMETER_NAMES)
    num_runs = num_trajectories * (num_parameters + 1)

    print("\n==========================================")
    print("          MORRIS SENSITIVITY")
    print("==========================================")
    print(f"Trajectories        : {num_trajectories}")
    print(f"Levels              : {NUM_LEVELS}")
    print(f"Model evaluations   : {num_runs}")
    print(f"Random seed         : {RANDOM_SEED}")
    print("------------------------------------------")

    elementary_effects = {
        name: []
        for name in PARAMETER_NAMES
    }

    all_runs = []

    for trajectory_number in range(num_trajectories):

        normalized_trajectory, order = build_morris_trajectory(
            rng,
            NUM_LEVELS,
            num_parameters,
        )

        physical_trajectory = scale_parameters(
            normalized_trajectory
        )

        outputs = []

        for point_number, parameters in enumerate(
            physical_trajectory,
            start=1,
        ):

            output = evaluate_model(parameters)
            outputs.append(output)

            all_runs.append({
                "trajectory": trajectory_number + 1,
                "point": point_number,
                "deep_depth_m": parameters[0],
                "shallow_depth_m": parameters[1],
                "slope_end_km": parameters[2],
                "amplification": output,
            })

            print(
                f"Trajectory {trajectory_number + 1:02d}/"
                f"{num_trajectories}, "
                f"point {point_number}/"
                f"{num_parameters + 1}: "
                f"deep={parameters[0]:7.3f} m, "
                f"shallow={parameters[1]:6.3f} m, "
                f"slope_end={parameters[2]:6.3f} km "
                f"-> amplification={output:.5f}x"
            )

        outputs = np.array(outputs)

        # Each step changes exactly one coded parameter.
        for step_index, parameter_index in enumerate(order):

            delta_y = (
                outputs[step_index + 1]
                - outputs[step_index]
            )

            delta_x = (
                normalized_trajectory[step_index + 1, parameter_index]
                - normalized_trajectory[step_index, parameter_index]
            )

            elementary_effect = delta_y / delta_x

            elementary_effects[
                PARAMETER_NAMES[parameter_index]
            ].append(elementary_effect)

    return all_runs, elementary_effects


# ============================================================
# Save outputs
# ============================================================


def save_run_data(rows):

    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = OUTPUT_DIR / "morris_model_runs.csv"

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trajectory",
                "point",
                "deep_depth_m",
                "shallow_depth_m",
                "slope_end_km",
                "amplification",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return filename


def save_summary(summary_rows):

    filename = OUTPUT_DIR / "morris_sensitivity_summary.csv"

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parameter",
                "mu",
                "mu_star",
                "sigma",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    return filename


# ============================================================
# Plot sensitivity results
# ============================================================


def plot_morris(summary_rows):

    names = [row["parameter"] for row in summary_rows]
    mu_star = np.array([
        float(row["mu_star"])
        for row in summary_rows
    ])
    sigma = np.array([
        float(row["sigma"])
        for row in summary_rows
    ])

    order = np.argsort(mu_star)[::-1]

    names = [names[i] for i in order]
    mu_star = mu_star[order]
    sigma = sigma[order]

    labels = [
        "Deep depth",
        "Shallow depth",
        "Slope end",
    ]

    names = [
        labels[PARAMETER_NAMES.index(name)]
        for name in names
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(
        np.arange(len(names)),
        mu_star,
        yerr=sigma,
        capsize=5,
    )
    plt.xticks(
        np.arange(len(names)),
        names,
    )
    plt.ylabel("Morris mean absolute elementary effect (mu*)")
    plt.title("Global Sensitivity of Tsunami Amplification")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "morris_sensitivity.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# Entry point
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Morris global sensitivity analysis "
            "for the 1D tsunami model."
        )
    )

    parser.add_argument(
        "--trajectories",
        type=int,
        default=DEFAULT_TRAJECTORIES,
        help=(
            "Number of Morris trajectories. "
            "Use 4 first for a quick test; 32 for the main experiment."
        ),
    )

    args = parser.parse_args()

    if args.trajectories < 1:
        raise ValueError("Number of trajectories must be at least 1.")

    all_runs, elementary_effects = run_morris(
        args.trajectories
    )

    summary_rows = []

    for parameter in PARAMETER_NAMES:

        effects = np.array(
            elementary_effects[parameter],
            dtype=float,
        )

        mu = np.mean(effects)
        mu_star = np.mean(np.abs(effects))
        sigma = np.std(effects, ddof=1)

        summary_rows.append({
            "parameter": parameter,
            "mu": mu,
            "mu_star": mu_star,
            "sigma": sigma,
        })

    run_file = save_run_data(all_runs)
    summary_file = save_summary(summary_rows)
    plot_morris(summary_rows)

    ranked = sorted(
        summary_rows,
        key=lambda row: float(row["mu_star"]),
        reverse=True,
    )

    print("\n==========================================")
    print("          MORRIS RESULTS")
    print("==========================================")
    print(
        f"{'Parameter':<18}"
        f"{'mu':>12}"
        f"{'mu*':>12}"
        f"{'sigma':>12}"
    )
    print("-" * 54)

    for row in ranked:
        print(
            f"{row['parameter']:<18}"
            f"{float(row['mu']):>12.5f}"
            f"{float(row['mu_star']):>12.5f}"
            f"{float(row['sigma']):>12.5f}"
        )

    print("------------------------------------------")
    print("Parameter ranking by mu*")

    for index, row in enumerate(ranked, start=1):
        print(
            f"{index}. {row['parameter']} "
            f"(mu* = {float(row['mu_star']):.5f})"
        )

    print("\nOutputs saved to:")
    print(run_file)
    print(summary_file)
    print(OUTPUT_DIR / "morris_sensitivity.png")


if __name__ == "__main__":
    main()
