import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

g = 9.81
depth = 100.0

# ============================================================
# 2. COMPUTATIONAL DOMAIN
# ============================================================

L = 100_000.0
dx = 100.0

x = np.arange(0, L + dx, dx)
N = len(x)

# ============================================================
# 3. THEORETICAL WAVE SPEED
# ============================================================

c_theory = np.sqrt(g * depth)

print(f"Theoretical wave speed: {c_theory:.4f} m/s")
print(f"Theoretical wave speed: {c_theory * 3.6:.4f} km/h")

# ============================================================
# 4. TIME STEP
# ============================================================

dt = 1.0

CFL = c_theory * dt / dx

print(f"CFL number: {CFL:.3f}")

if CFL >= 1:
    raise ValueError("CFL condition violated!")

# ============================================================
# 5. INITIAL SURFACE WAVE
# ============================================================

amplitude = 1.0
x0 = 20_000.0
sigma = 2_000.0

eta = amplitude * np.exp(
    -((x - x0) ** 2) / (2 * sigma ** 2)
)

# ============================================================
# 6. PURE RIGHT-GOING INITIAL VELOCITY
# ============================================================

u = (c_theory / depth) * eta

print(f"Initial maximum velocity: {np.max(u):.4f} m/s")

# ============================================================
# 7. SIMULATION SETTINGS
# ============================================================

total_time = 300.0
num_steps = int(total_time / dt)

measurement_interval = 10

snapshot_times = [
    0,
    50,
    100,
    150,
    200,
    250,
    300
]

snapshots = {}

peak_times = []
peak_positions = []

# ============================================================
# 8. TIME INTEGRATION
# ============================================================

for step in range(num_steps + 1):

    time = step * dt

    # --------------------------------------------------------
    # Measure wave peak
    # --------------------------------------------------------

    if step % measurement_interval == 0:

        # Ignore the initial transient region
        if time >= 20.0:

            # Search for the positive wave peak
            peak_index = np.argmax(eta)

            # ------------------------------------------------
            # Sub-grid peak interpolation
            # ------------------------------------------------

            if 0 < peak_index < N - 1:

                y1 = eta[peak_index - 1]
                y2 = eta[peak_index]
                y3 = eta[peak_index + 1]

                denominator = (
                    y1 - 2 * y2 + y3
                )

                if abs(denominator) > 1e-12:

                    offset = (
                        0.5
                        * (y1 - y3)
                        / denominator
                    )

                    peak_position = (
                        x[peak_index]
                        + offset * dx
                    )

                else:

                    peak_position = x[peak_index]

            else:

                peak_position = x[peak_index]

            peak_times.append(time)
            peak_positions.append(peak_position)

    # --------------------------------------------------------
    # Save wave snapshots
    # --------------------------------------------------------

    if step in [
        int(t / dt)
        for t in snapshot_times
    ]:

        snapshots[time] = eta.copy()

    if step == num_steps:
        break

    # ========================================================
    # LAX-FRIEDRICHS UPDATE
    # ========================================================

    eta_new = eta.copy()
    u_new = u.copy()

    # Surface elevation
    eta_new[1:-1] = (
        0.5 * (eta[2:] + eta[:-2])
        - (depth * dt / (2 * dx))
        * (u[2:] - u[:-2])
    )

    # Velocity
    u_new[1:-1] = (
        0.5 * (u[2:] + u[:-2])
        - (g * dt / (2 * dx))
        * (eta[2:] - eta[:-2])
    )

    # ========================================================
    # BOUNDARY CONDITIONS
    # ========================================================

    eta_new[0] = eta_new[1]
    eta_new[-1] = eta_new[-2]

    u_new[0] = 0.0
    u_new[-1] = 0.0

    # ========================================================
    # ADVANCE SOLUTION
    # ========================================================

    eta = eta_new
    u = u_new

# ============================================================
# 9. CONVERT MEASUREMENTS
# ============================================================

peak_times = np.array(peak_times)
peak_positions = np.array(peak_positions)

peak_positions_km = peak_positions / 1000.0

# ============================================================
# 10. LINEAR REGRESSION
# ============================================================

slope, intercept = np.polyfit(
    peak_times,
    peak_positions_km,
    1
)

c_numerical = slope * 1000.0

# ============================================================
# 11. VALIDATION ERROR
# ============================================================

speed_error = (
    abs(c_numerical - c_theory)
    / c_theory
    * 100
)

# ============================================================
# 12. PRINT RESULTS
# ============================================================

print()
print("==========================================")
print("       RIGHT-GOING WAVE VALIDATION")
print("==========================================")

print(
    f"Theoretical speed : "
    f"{c_theory:.4f} m/s"
)

print(
    f"Numerical speed   : "
    f"{c_numerical:.4f} m/s"
)

print(
    f"Theoretical speed : "
    f"{c_theory * 3.6:.4f} km/h"
)

print(
    f"Numerical speed   : "
    f"{c_numerical * 3.6:.4f} km/h"
)

print(
    f"Speed error       : "
    f"{speed_error:.2f}%"
)

print("==========================================")

# ============================================================
# 13. PLOT WAVE PROPAGATION
# ============================================================

plt.figure(figsize=(12, 6))

for time, wave in snapshots.items():

    plt.plot(
        x / 1000,
        wave,
        label=f"{time:.0f} s"
    )

plt.xlabel("Distance (km)")
plt.ylabel("Surface displacement η (m)")

plt.title(
    "Pure Right-Going Shallow-Water Wave"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ============================================================
# 14. PLOT PEAK POSITION
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    peak_times,
    peak_positions_km,
    label="Measured wave peak"
)

fitted_positions = (
    slope * peak_times + intercept
)

plt.plot(
    peak_times,
    fitted_positions,
    label="Linear fit"
)

plt.xlabel("Time (s)")
plt.ylabel("Wave peak position (km)")

plt.title(
    "Right-Going Wave Speed Validation"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()