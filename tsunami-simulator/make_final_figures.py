from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "uq_outputs"
FIG_DIR = OUT / "final_figures"

def rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = rows(OUT / "uq_results_multivariate.csv")
    deep = np.array([float(r["deep_depth_m"]) for r in data])
    shallow = np.array([float(r["shallow_depth_m"]) for r in data])
    slope = np.array([float(r["slope_end_km"]) for r in data])
    amp = np.array([float(r["amplification"]) for r in data])

    # 1. Baseline bathymetry
    x = np.linspace(0, 100, 2001)
    depth = np.full_like(x, 200.0)
    mask = (x >= 50) & (x < 80)
    depth[mask] = 200.0 + (50.0 - 200.0) * (x[mask] - 50.0) / 30.0
    depth[x >= 80] = 50.0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, depth, linewidth=2)
    ax.invert_yaxis()
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Water depth (m)")
    ax.set_title("Baseline Bathymetry")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / "01_bathymetry_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Amplification distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(amp, bins=12, edgecolor="black")
    ax.axvline(np.mean(amp), linestyle="--", linewidth=1.5, label="Monte Carlo mean")
    ax.axvline(1.35811, linestyle=":", linewidth=1.5, label="Baseline")
    ax.set_xlabel("Amplification at 80 km")
    ax.set_ylabel("Number of realizations")
    ax.set_title("Monte Carlo Distribution of Wave Amplification")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(FIG_DIR / "02_amplification_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3-5. Parameter scatter plots
    specs = [
        (deep, "Deep-water depth (m)", "03_deep_depth_vs_amplification.png",
         "Deep-Water Depth vs Wave Amplification"),
        (shallow, "Shallow-water depth (m)", "04_shallow_depth_vs_amplification.png",
         "Shallow-Water Depth vs Wave Amplification"),
        (slope, "Slope-end position (km)", "05_slope_end_vs_amplification.png",
         "Slope-End Position vs Wave Amplification"),
    ]
    for xx, xlabel, filename, title in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xx, amp)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplification at 80 km")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 6. Final sensitivity ranking from replicated Sobol total-order means.
    labels = ["slope_end_km", "shallow_depth_m", "deep_depth_m"]
    values = [0.72263, 0.22153, 0.06146]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Mean total-order Sobol sensitivity")
    ax.set_title("Global Sensitivity Ranking")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.savefig(FIG_DIR / "06_sensitivity_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Created 6 figures in: {FIG_DIR}")

if __name__ == "__main__":
    main()
