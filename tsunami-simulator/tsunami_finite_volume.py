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
dx = 12.5

x = np.arange(0.0, L + dx, dx)
N = len(x)


# ============================================================
# 3. VARIABLE BATHYMETRY
# ============================================================

depth = np.zeros_like(x)

# Deep ocean: 0-50 km
deep_region = x < 50_000.0
depth[deep_region] = 200.0

# Continental slope: 50-80 km
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

# Shallow region: 80-100 km
shallow_region = x >= 80_000.0
depth[shallow_region] = 50.0

# Bottom elevation.
# Free surface at rest is z = 0.
bottom = -depth


print(
    f"Deep-water depth: "
    f"{np.max(depth):.2f} m"
)

print(
    f"Shallow-water depth: "
    f"{np.min(depth):.2f} m"
)


# ============================================================
# 4. INITIAL RIGHT-GOING WAVE
# ============================================================

amplitude = 1.0
x0 = 20_000.0
sigma = 2_000.0


# Gaussian surface displacement
eta = amplitude * np.exp(
    -((x - x0) ** 2)
    / (2.0 * sigma ** 2)
)


# Depth at the initial wave location
initial_depth = np.interp(
    x0,
    x,
    depth
)


# Linear shallow-water wave speed
initial_speed = np.sqrt(
    g * initial_depth
)


# Right-going linear-wave velocity relation
u = (
    initial_speed
    / initial_depth
) * eta


# Total water depth
H = depth + eta


# Conserved momentum
q = H * u


print(
    f"Initial depth: "
    f"{initial_depth:.2f} m"
)

print(
    f"Initial wave speed: "
    f"{initial_speed:.4f} m/s"
)

print(
    f"Initial maximum elevation: "
    f"{np.max(eta):.4f} m"
)

print(
    f"Initial maximum velocity: "
    f"{np.max(np.abs(u)):.4f} m/s"
)


# ============================================================
# 5. PHYSICAL FLUX
# ============================================================

def flux(H, q):
    """
    Compute the shallow-water physical flux.

    Conservative variables:
        H = total water depth
        q = H * u

    Flux:
        F1 = q
        F2 = q^2/H + 0.5*g*H^2
    """

    velocity = np.zeros_like(H)

    wet = H > 1e-12

    velocity[wet] = (
        q[wet] / H[wet]
    )

    F1 = q

    F2 = (
        q * velocity
        + 0.5 * g * H**2
    )

    return F1, F2


# ============================================================
# 6. HYDROSTATIC RECONSTRUCTION + RUSANOV FLUX
# ============================================================

def hydrostatic_rusanov_flux(
    H_left,
    q_left,
    b_left,
    H_right,
    q_right,
    b_right
):
    """
    Compute hydrostatic-reconstructed
    Rusanov fluxes at all interfaces.
    """

    # --------------------------------------------------------
    # Free-surface elevation on each side
    # --------------------------------------------------------

    eta_left = H_left + b_left
    eta_right = H_right + b_right

    # --------------------------------------------------------
    # Interface bottom elevation
    # --------------------------------------------------------

    interface_bottom = np.maximum(
        b_left,
        b_right
    )

    # --------------------------------------------------------
    # Reconstructed interface depths
    # --------------------------------------------------------

    H_left_star = np.maximum(
        0.0,
        eta_left - interface_bottom
    )

    H_right_star = np.maximum(
        0.0,
        eta_right - interface_bottom
    )

    # --------------------------------------------------------
    # Reconstructed momenta
    # --------------------------------------------------------

    q_left_star = np.zeros_like(q_left)
    q_right_star = np.zeros_like(q_right)

    wet_left = H_left > 1e-12
    wet_right = H_right > 1e-12

    q_left_star[wet_left] = (
        q_left[wet_left]
        * H_left_star[wet_left]
        / H_left[wet_left]
    )

    q_right_star[wet_right] = (
        q_right[wet_right]
        * H_right_star[wet_right]
        / H_right[wet_right]
    )

    # --------------------------------------------------------
    # Physical fluxes
    # --------------------------------------------------------

    F1_left, F2_left = flux(
        H_left_star,
        q_left_star
    )

    F1_right, F2_right = flux(
        H_right_star,
        q_right_star
    )

    # --------------------------------------------------------
    # Velocities
    # --------------------------------------------------------

    u_left = np.zeros_like(
        H_left_star
    )

    u_right = np.zeros_like(
        H_right_star
    )

    wet_left_star = H_left_star > 1e-12
    wet_right_star = H_right_star > 1e-12

    u_left[wet_left_star] = (
        q_left_star[wet_left_star]
        / H_left_star[wet_left_star]
    )

    u_right[wet_right_star] = (
        q_right_star[wet_right_star]
        / H_right_star[wet_right_star]
    )

    # --------------------------------------------------------
    # Local characteristic speeds
    # --------------------------------------------------------

    wave_speed_left = (
        np.abs(u_left)
        + np.sqrt(g * H_left_star)
    )

    wave_speed_right = (
        np.abs(u_right)
        + np.sqrt(g * H_right_star)
    )

    interface_speed = np.maximum(
        wave_speed_left,
        wave_speed_right
    )

    # --------------------------------------------------------
    # Rusanov flux
    # --------------------------------------------------------

    numerical_F1 = 0.5 * (
        F1_left
        + F1_right
        - interface_speed
        * (
            H_right_star
            - H_left_star
        )
    )

    numerical_F2 = 0.5 * (
        F2_left
        + F2_right
        - interface_speed
        * (
            q_right_star
            - q_left_star
        )
    )

    # --------------------------------------------------------
    # Hydrostatic pressure corrections
    # --------------------------------------------------------

    F2_minus = (
        numerical_F2
        + 0.5 * g
        * (
            H_left**2
            - H_left_star**2
        )
    )

    F2_plus = (
        numerical_F2
        + 0.5 * g
        * (
            H_right**2
            - H_right_star**2
        )
    )

    F1_minus = numerical_F1
    F1_plus = numerical_F1

    return (
        F1_minus,
        F2_minus,
        F1_plus,
        F2_plus
    )


# ============================================================
# 7. FINITE-VOLUME UPDATE
# ============================================================

def finite_volume_step(
    H,
    q,
    bottom,
    dt,
    dx
):
    """
    Advance the shallow-water solution
    by one finite-volume timestep.
    """

    (
        F1_minus,
        F2_minus,
        F1_plus,
        F2_plus
    ) = hydrostatic_rusanov_flux(
        H[:-1],
        q[:-1],
        bottom[:-1],
        H[1:],
        q[1:],
        bottom[1:]
    )

    H_new = H.copy()
    q_new = q.copy()

    # --------------------------------------------------------
    # Conservative update
    # --------------------------------------------------------

    H_new[1:-1] = (
        H[1:-1]
        - (dt / dx)
        * (
            F1_minus[1:]
            - F1_plus[:-1]
        )
    )

    q_new[1:-1] = (
        q[1:-1]
        - (dt / dx)
        * (
            F2_minus[1:]
            - F2_plus[:-1]
        )
    )

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    H_new[0] = H_new[1]
    H_new[-1] = H_new[-2]

    q_new[0] = q_new[1]
    q_new[-1] = q_new[-2]

    return H_new, q_new


# ============================================================
# 8. CFL-CONTROLLED TIME STEP
# ============================================================

CFL_target = 0.5

velocity = np.zeros_like(H)

wet = H > 1e-12

velocity[wet] = (
    q[wet] / H[wet]
)

local_wave_speed = (
    np.abs(velocity)
    + np.sqrt(g * H)
)

maximum_speed = np.max(
    local_wave_speed
)

dt = (
    CFL_target
    * dx
    / maximum_speed
)

CFL = (
    maximum_speed
    * dt
    / dx
)

print(
    f"Maximum wave speed: "
    f"{maximum_speed:.4f} m/s"
)

print(
    f"Time step: "
    f"{dt:.4f} s"
)

print(
    f"Maximum CFL number: "
    f"{CFL:.3f}"
)


# ============================================================
# 9. WAVE PROPAGATION
# ============================================================

total_time = 2600.0

num_steps = int(
    total_time / dt
)


# ------------------------------------------------------------
# Virtual wave gauges
# ------------------------------------------------------------

gauge_positions = np.array([
    40_000.0,
    50_000.0,
    60_000.0,
    70_000.0,
    80_000.0,
    85_000.0
])

gauge_indices = [
    np.argmin(
        np.abs(x - position)
    )
    for position in gauge_positions
]


gauge_history = {
    position: []
    for position in gauge_positions
}

time_history = []


# ------------------------------------------------------------
# Save snapshots
# ------------------------------------------------------------

snapshot_times = np.arange(
    0.0,
    2600.0 + 50.0,
    200.0
)

snapshots = {}


# ============================================================
# 10. TIME INTEGRATION
# ============================================================

time = 0.0

for step in range(num_steps + 1):

    # --------------------------------------------------------
    # Record gauges
    # --------------------------------------------------------

    for position, index in zip(
        gauge_positions,
        gauge_indices
    ):
        eta_current = H + bottom

        gauge_history[position].append(
            eta_current[index]
        )

    time_history.append(time)


    # --------------------------------------------------------
    # Save snapshots
    # --------------------------------------------------------

    for snapshot_time in snapshot_times:

        if (
            abs(time - snapshot_time)
            < dt / 2.0
            and snapshot_time not in snapshots
        ):

            snapshots[snapshot_time] = (
                H + bottom
            ).copy()


    # --------------------------------------------------------
    # Stop after final timestep
    # --------------------------------------------------------

    if step == num_steps:
        break


    # --------------------------------------------------------
    # Advance solution
    # --------------------------------------------------------

    H, q = finite_volume_step(
        H,
        q,
        bottom,
        dt,
        dx
    )

    time += dt


# ============================================================
# 11. INCIDENT-PULSE ANALYSIS
# ============================================================

# Estimate travel time from the initial wave location
# to each gauge using the local shallow-water speed:
#
#     c(x) = sqrt(g h(x))
#
# Travel time:
#
#     T = integral(dx / c(x))

local_speed = np.sqrt(
    g * depth
)

travel_time = np.zeros_like(x)

for i in range(1, N):

    travel_time[i] = (
        travel_time[i - 1]
        + 0.5
        * (
            1.0 / local_speed[i - 1]
            + 1.0 / local_speed[i]
        )
        * dx
    )


# Interpolate expected arrival time at each gauge
expected_arrival_times = np.interp(
    gauge_positions,
    x,
    travel_time
)

# Correct travel time so that it starts at x0
initial_travel_time = np.interp(
    x0,
    x,
    travel_time
)

expected_arrival_times -= (
    initial_travel_time
)


# ------------------------------------------------------------
# Find the first incident pulse
# ------------------------------------------------------------

gauge_amplitudes = []
gauge_arrival_times = []

for position, expected_time in zip(
    gauge_positions,
    expected_arrival_times
):

    signal = np.array(
        gauge_history[position]
    )

    times = np.array(
        time_history
    )

    # Search only around the expected arrival.
    #
    # This avoids accidentally measuring a later reflected
    # wave as the "maximum tsunami amplitude".
    window_half_width = 200.0

    window = (
        (times >= expected_time - window_half_width)
        &
        (times <= expected_time + window_half_width)
    )

    window_indices = np.where(window)[0]

    if len(window_indices) == 0:
        raise RuntimeError(
            f"No arrival window found for "
            f"gauge at {position / 1000:.1f} km"
        )

    local_index = window_indices[
        np.argmax(
            signal[window_indices]
        )
    ]

    incident_amplitude = (
        signal[local_index]
    )

    incident_arrival_time = (
        times[local_index]
    )

    gauge_amplitudes.append(
        incident_amplitude
    )

    gauge_arrival_times.append(
        incident_arrival_time
    )


gauge_amplitudes = np.array(
    gauge_amplitudes
)

gauge_arrival_times = np.array(
    gauge_arrival_times
)


# ------------------------------------------------------------
# Relative amplification
# ------------------------------------------------------------

reference_amplitude = (
    gauge_amplitudes[0]
)

amplification_factors = (
    gauge_amplitudes
    / reference_amplitude
)


# ============================================================
# RESULTS
# ============================================================

print()
print("==========================================")
print("       INCIDENT-PULSE ANALYSIS")
print("==========================================")

print(
    "Position | Depth | Expected t | "
    "Peak t | Peak eta | Amplification"
)

print("-" * 80)

for position, expected_time, arrival_time, amplitude_value, amplification in zip(
    gauge_positions,
    expected_arrival_times,
    gauge_arrival_times,
    gauge_amplitudes,
    amplification_factors
):

    gauge_depth = np.interp(
        position,
        x,
        depth
    )

    print(
        f"{position / 1000:7.0f} km | "
        f"{gauge_depth:5.0f} m | "
        f"{expected_time:10.1f} s | "
        f"{arrival_time:7.1f} s | "
        f"{amplitude_value:9.5f} m | "
        f"{amplification:10.4f}x"
    )

print("==========================================")


# ============================================================
# 12. PLOT BATHYMETRY
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    x / 1000.0,
    depth
)

plt.gca().invert_yaxis()

plt.xlabel("Distance (km)")
plt.ylabel("Water depth (m)")
plt.title(
    "Variable Bathymetry"
)

plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 13. PLOT WAVE PROPAGATION
# ============================================================

plt.figure(figsize=(12, 6))

for snapshot_time, wave in snapshots.items():

    plt.plot(
        x / 1000.0,
        wave,
        label=f"{snapshot_time:.0f} s"
    )

plt.xlabel("Distance (km)")
plt.ylabel("Surface displacement η (m)")
plt.title(
    "Tsunami Wave Propagation Over Variable Bathymetry"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# 14. PLOT WAVE GAUGES
# ============================================================

plt.figure(figsize=(12, 7))

for position in gauge_positions:

    plt.plot(
        np.array(time_history),
        np.array(
            gauge_history[position]
        ),
        label=f"{position / 1000:.0f} km"
    )

plt.xlabel("Time (s)")
plt.ylabel("Surface displacement η (m)")
plt.title(
    "Wave-Gauge Measurements"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# 15. PLOT AMPLIFICATION
# ============================================================

amplification_factors = (
    np.array(gauge_amplitudes)
    / reference_amplitude
)

plt.figure(figsize=(10, 5))

plt.plot(
    gauge_positions / 1000.0,
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

# ============================================================
# 16. GREEN'S-LAW COMPARISON
# ============================================================

green_amplification = (
    depth[[
        np.argmin(np.abs(x - position))
        for position in gauge_positions
    ]]
)

green_amplification = (
    200.0 / green_amplification
) ** 0.25


print()
print("==========================================")
print("      GREEN'S-LAW COMPARISON")
print("==========================================")

print(
    "Position (km) | Depth (m) | "
    "Simulated | Green's law"
)

print("-" * 60)

for position, simulated, theoretical in zip(
    gauge_positions,
    amplification_factors,
    green_amplification
):

    gauge_depth = np.interp(
        position,
        x,
        depth
    )

    print(
        f"{position / 1000:10.0f} | "
        f"{gauge_depth:9.2f} | "
        f"{simulated:9.4f}x | "
        f"{theoretical:9.4f}x"
    )

print("==========================================")

# ============================================================
# 17. SIMULATED VS GREEN'S-LAW AMPLIFICATION
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    gauge_positions / 1000.0,
    amplification_factors,
    marker="o",
    label="Finite-volume simulation"
)

plt.plot(
    gauge_positions / 1000.0,
    green_amplification,
    marker="s",
    linestyle="--",
    label="Green's-law reference"
)

plt.xlabel("Distance (km)")
plt.ylabel("Relative amplification")
plt.title(
    "Simulated Shoaling vs Green's-Law Reference"
)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()