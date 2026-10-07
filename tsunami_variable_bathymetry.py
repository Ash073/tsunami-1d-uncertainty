import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

g = 9.81


# ============================================================
# 2. COMPUTATIONAL DOMAIN
# ============================================================

L = 100_000.0
dx = 100.0

x = np.arange(0, L + dx, dx)
N = len(x)


# ============================================================
# 3. BATHYMETRY
# ============================================================

depth = np.zeros_like(x)


# Deep ocean: 0 - 50 km
deep_region = x < 50_000.0
depth[deep_region] = 200.0


# Continental slope: 50 - 80 km
slope_region = (
    (x >= 50_000.0)
    & (x < 80_000.0)
)

depth[slope_region] = (
    200.0
    - 150.0
    * (
        (x[slope_region] - 50_000.0)
        / 30_000.0
    )
)


# Shallow region: 80 - 100 km
shallow_region = x >= 80_000.0
depth[shallow_region] = 50.0


# ============================================================
# 4. LOCAL WAVE SPEED
# ============================================================

wave_speed = np.sqrt(g * depth)

print(
    f"Deep-water wave speed: "
    f"{np.sqrt(g * 200):.2f} m/s"
)

print(
    f"Shallow-water wave speed: "
    f"{np.sqrt(g * 50):.2f} m/s"
)


# ============================================================
# 5. TIME STEP
# ============================================================

# Maximum wave speed occurs in deepest water
max_speed = np.max(wave_speed)

CFL_target = 0.5

dt = CFL_target * dx / max_speed

print(f"Time step: {dt:.4f} s")

CFL = max_speed * dt / dx

print(f"Maximum CFL number: {CFL:.3f}")


# ============================================================
# 6. INITIAL WAVE
# ============================================================

amplitude = 1.0

x0 = 20_000.0
sigma = 2_000.0


eta = amplitude * np.exp(
    -((x - x0) ** 2)
    / (2 * sigma ** 2)
)


# ============================================================
# 7. INITIAL VELOCITY
# ============================================================

# Local depth at the initial wave position
initial_depth = np.interp(
    x0,
    x,
    depth
)

initial_speed = np.sqrt(
    g * initial_depth
)

u = (
    initial_speed
    / initial_depth
) * eta


print(
    f"Initial depth: "
    f"{initial_depth:.2f} m"
)

print(
    f"Initial wave speed: "
    f"{initial_speed:.2f} m/s"
)


# ============================================================
# 8. SIMULATION SETTINGS
# ============================================================

total_time = 2600.0

num_steps = int(
    total_time / dt
)


# Save snapshots
snapshot_times = [
    0,
    200,
    400,
    600,
    800,
    1000,
    1200,
    1400
]

snapshots = {}


# ============================================================
# 9. VIRTUAL WAVE GAUGES
# ============================================================

# Each gauge measures the surface displacement eta(t)
# at a fixed position.

gauge_positions = [
    40_000.0,   # 40 km - before slope
    50_000.0,   # 50 km - start of slope
    60_000.0,   # 60 km - on slope
    70_000.0,   # 70 km - on slope
    80_000.0,   # 80 km - shallow region begins
    90_000.0    # 90 km - shallow region
]

# Convert physical positions into grid indices
gauge_indices = [
    np.argmin(np.abs(x - position))
    for position in gauge_positions
]

# Store eta(t) at every gauge
gauge_history = {
    position: []
    for position in gauge_positions
}

time_history = []

# ============================================================
# 10. TIME INTEGRATION
# ============================================================

for step in range(num_steps + 1):

    time = step * dt


    # --------------------------------------------------------
    # Save snapshot
    # --------------------------------------------------------

    for snapshot_time in snapshot_times:

        if (
            abs(time - snapshot_time)
            < dt / 2
            and snapshot_time not in snapshots
        ):

            snapshots[snapshot_time] = eta.copy()


    # --------------------------------------------------------
    # Record wave-gauge measurements
    # --------------------------------------------------------

    for position, index in zip(
        gauge_positions,
        gauge_indices
    ):
        gauge_history[position].append(
            eta[index]
        )

    time_history.append(time)

    # ========================================================
    # NUMERICAL UPDATE
    # ========================================================

    eta_new = eta.copy()
    u_new = u.copy()


    # --------------------------------------------------------
    # Continuity equation
    #
    # eta_t + d(hu)/dx = 0
    # --------------------------------------------------------

    hu = depth * u

    eta_new[1:-1] = (
        0.5
        * (
            eta[2:]
            + eta[:-2]
        )
        -
        (
            dt
            / (2 * dx)
        )
        * (
            hu[2:]
            - hu[:-2]
        )
    )


    # --------------------------------------------------------
    # Momentum equation
    #
    # u_t + g d(eta)/dx = 0
    # --------------------------------------------------------

    u_new[1:-1] = (
        0.5
        * (
            u[2:]
            + u[:-2]
        )
        -
        (
            g
            * dt
            / (2 * dx)
        )
        * (
            eta[2:]
            - eta[:-2]
        )
    )


    # ========================================================
    # BOUNDARY CONDITIONS
    # ========================================================

    eta_new[0] = eta_new[1]
    eta_new[-1] = eta_new[-2]

    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]


    # ========================================================
    # UPDATE
    # ========================================================

    eta = eta_new
    u = u_new


# ============================================================
# 11. GAUGE AMPLIFICATION ANALYSIS
# ============================================================

print()
print("==========================================")
print("       WAVE GAUGE ANALYSIS")
print("==========================================")

# Use the first gauge as the reference amplitude
reference_position = gauge_positions[0]

reference_signal = np.array(
    gauge_history[reference_position]
)

reference_amplitude = np.max(
    np.abs(reference_signal)
)

print(
    f"Reference gauge : "
    f"{reference_position / 1000:.0f} km"
)

print(
    f"Reference amplitude : "
    f"{reference_amplitude:.4f} m"
)

print()

print(
    "Position (km) | Depth (m) | "
    "Max |eta| (m) | Amplification"
)

print("-" * 60)

for position in gauge_positions:

    signal = np.array(
        gauge_history[position]
    )

    maximum_amplitude = np.max(
        np.abs(signal)
    )

    gauge_depth = np.interp(
        position,
        x,
        depth
    )

    amplification = (
        maximum_amplitude
        / reference_amplitude
    )

    print(
        f"{position / 1000:10.0f} | "
        f"{gauge_depth:9.2f} | "
        f"{maximum_amplitude:13.4f} | "
        f"{amplification:13.4f}x"
    )

print("==========================================")


# ============================================================
# 12. PLOT BATHYMETRY
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    x / 1000,
    depth
)

plt.gca().invert_yaxis()

plt.xlabel("Distance (km)")
plt.ylabel("Water depth (m)")

plt.title(
    "Seafloor Bathymetry"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 13. WAVE GAUGE TIME SERIES
# ============================================================

plt.figure(figsize=(12, 7))

for position in gauge_positions:

    plt.plot(
        np.array(time_history),
        np.array(gauge_history[position]),
        label=f"{position / 1000:.0f} km"
    )

plt.xlabel("Time (s)")
plt.ylabel("Surface displacement η (m)")
plt.title(
    "Wave-Gauge Measurements Over Variable Bathymetry"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ============================================================
# 14. AMPLIFICATION VS POSITION
# ============================================================

gauge_amplitudes = []
gauge_depths = []

for position in gauge_positions:

    signal = np.array(
        gauge_history[position]
    )

    gauge_amplitudes.append(
        np.max(np.abs(signal))
    )

    gauge_depths.append(
        np.interp(
            position,
            x,
            depth
        )
    )

gauge_amplitudes = np.array(
    gauge_amplitudes
)

amplification_factors = (
    gauge_amplitudes
    / reference_amplitude
)

plt.figure(figsize=(10, 5))

plt.plot(
    np.array(gauge_positions) / 1000,
    amplification_factors,
    marker="o"
)

plt.xlabel("Distance (km)")
plt.ylabel("Amplification factor")
plt.title(
    "Wave Amplification Over Variable Bathymetry"
)

plt.grid()
plt.tight_layout()
plt.show()