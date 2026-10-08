import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

g = 9.81


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

L = 100_000.0

dx = 50.0

deep_depth = 200.0

slope_start = 50_000.0

slope_end = 80_000.0

source_position = 20_000.0

wave_amplitude = 1.0

wave_sigma = 2_000.0

total_time = 2200.0

CFL_target = 0.4


# ============================================================
# 3. COMPUTATIONAL GRID
# ============================================================

x = np.arange(
    0.0,
    L + dx,
    dx
)

N = len(x)


# ============================================================
# 4. BATHYMETRY
# ============================================================

def create_bathymetry(
    x,
    shallow_depth,
    deep_depth=200.0,
    slope_start=50_000.0,
    slope_end=80_000.0
):
    """
    Create the piecewise-linear bathymetry.

    Deep ocean:
        0 -> slope_start

    Continental slope:
        slope_start -> slope_end

    Shallow region:
        slope_end -> end
    """

    depth = np.zeros_like(x)

    # --------------------------------------------------------
    # Deep region
    # --------------------------------------------------------

    deep_region = (
        x < slope_start
    )

    depth[deep_region] = (
        deep_depth
    )

    # --------------------------------------------------------
    # Continental slope
    # --------------------------------------------------------

    slope_region = (
        (x >= slope_start)
        &
        (x < slope_end)
    )

    fraction = (
        x[slope_region]
        - slope_start
    ) / (
        slope_end
        - slope_start
    )

    depth[slope_region] = (
        deep_depth
        + (
            shallow_depth
            - deep_depth
        )
        * fraction
    )

    # --------------------------------------------------------
    # Shallow region
    # --------------------------------------------------------

    shallow_region = (
        x >= slope_end
    )

    depth[shallow_region] = (
        shallow_depth
    )

    return depth


# ============================================================
# 5. MINMOD LIMITER
# ============================================================

def minmod3(a, b, c):
    """
    Three-argument minmod limiter.
    """

    same_sign = (
        (np.sign(a) == np.sign(b))
        &
        (np.sign(b) == np.sign(c))
    )

    magnitude = np.minimum(
        np.abs(a),
        np.minimum(
            np.abs(b),
            np.abs(c)
        )
    )

    result = np.zeros_like(a)

    result[same_sign] = (
        np.sign(a[same_sign])
        * magnitude[same_sign]
    )

    return result


# ============================================================
# 6. PRIMITIVE VARIABLES
# ============================================================

def primitive_variables(
    H,
    q,
    bottom
):
    """
    Convert conservative variables into:

        eta = free-surface elevation
        u   = velocity
    """

    eta = H + bottom

    u = np.zeros_like(H)

    wet = H > 1e-12

    u[wet] = (
        q[wet]
        / H[wet]
    )

    return eta, u


# ============================================================
# 7. MUSCL SLOPES
# ============================================================

def compute_primitive_slopes(
    eta,
    u,
    dx
):
    """
    Limited MUSCL slopes for eta and u.
    """

    eta_slope = np.zeros_like(eta)

    u_slope = np.zeros_like(u)

    # --------------------------------------------------------
    # Surface elevation slope
    # --------------------------------------------------------

    eta_backward = (
        eta[1:-1]
        - eta[:-2]
    ) / dx

    eta_centered = (
        eta[2:]
        - eta[:-2]
    ) / (2.0 * dx)

    eta_forward = (
        eta[2:]
        - eta[1:-1]
    ) / dx

    eta_slope[1:-1] = minmod3(
        eta_backward,
        eta_centered,
        eta_forward
    )

    # --------------------------------------------------------
    # Velocity slope
    # --------------------------------------------------------

    u_backward = (
        u[1:-1]
        - u[:-2]
    ) / dx

    u_centered = (
        u[2:]
        - u[:-2]
    ) / (2.0 * dx)

    u_forward = (
        u[2:]
        - u[1:-1]
    ) / dx

    u_slope[1:-1] = minmod3(
        u_backward,
        u_centered,
        u_forward
    )

    return (
        eta_slope,
        u_slope
    )


# ============================================================
# 8. PHYSICAL FLUX
# ============================================================

def physical_flux(
    H,
    q
):
    """
    Shallow-water physical flux.
    """

    velocity = np.zeros_like(H)

    wet = H > 1e-12

    velocity[wet] = (
        q[wet]
        / H[wet]
    )

    F1 = q

    F2 = (
        q * velocity
        + 0.5 * g * H**2
    )

    return (
        F1,
        F2
    )


# ============================================================
# 9. MUSCL + HYDROSTATIC RUSANOV FLUX
# ============================================================

def hydrostatic_muscl_flux(
    H,
    q,
    bottom,
    dx
):
    """
    MUSCL reconstruction +
    hydrostatic reconstruction +
    Rusanov flux.
    """

    # --------------------------------------------------------
    # Primitive variables
    # --------------------------------------------------------

    eta, u = primitive_variables(
        H,
        q,
        bottom
    )

    # --------------------------------------------------------
    # Slopes
    # --------------------------------------------------------

    eta_slope, u_slope = (
        compute_primitive_slopes(
            eta,
            u,
            dx
        )
    )

    # --------------------------------------------------------
    # Reconstruct interface states
    # --------------------------------------------------------

    eta_left = (
        eta[:-1]
        + 0.5 * dx * eta_slope[:-1]
    )

    eta_right = (
        eta[1:]
        - 0.5 * dx * eta_slope[1:]
    )

    u_left = (
        u[:-1]
        + 0.5 * dx * u_slope[:-1]
    )

    u_right = (
        u[1:]
        - 0.5 * dx * u_slope[1:]
    )

    # --------------------------------------------------------
    # Interface bathymetry
    # --------------------------------------------------------

    interface_bottom = np.maximum(
        bottom[:-1],
        bottom[1:]
    )

    # --------------------------------------------------------
    # Hydrostatic reconstruction
    # --------------------------------------------------------

    H_left = np.maximum(
        0.0,
        eta_left
        - interface_bottom
    )

    H_right = np.maximum(
        0.0,
        eta_right
        - interface_bottom
    )

    # Reconstructed momentum
    q_left = H_left * u_left

    q_right = H_right * u_right

    # --------------------------------------------------------
    # Physical fluxes
    # --------------------------------------------------------

    F1_left, F2_left = physical_flux(
        H_left,
        q_left
    )

    F1_right, F2_right = physical_flux(
        H_right,
        q_right
    )

    # --------------------------------------------------------
    # Characteristic speeds
    # --------------------------------------------------------

    wave_speed_left = (
        np.abs(u_left)
        + np.sqrt(g * H_left)
    )

    wave_speed_right = (
        np.abs(u_right)
        + np.sqrt(g * H_right)
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
            H_right
            - H_left
        )
    )

    numerical_F2 = 0.5 * (
        F2_left
        + F2_right
        - interface_speed
        * (
            q_right
            - q_left
        )
    )

    # --------------------------------------------------------
    # Hydrostatic corrections
    # --------------------------------------------------------

    H_left_cell = np.maximum(
        0.0,
        eta_left
        - bottom[:-1]
    )

    H_right_cell = np.maximum(
        0.0,
        eta_right
        - bottom[1:]
    )

    F2_minus = (
        numerical_F2
        + 0.5 * g
        * (
            H_left_cell**2
            - H_left**2
        )
    )

    F2_plus = (
        numerical_F2
        + 0.5 * g
        * (
            H_right_cell**2
            - H_right**2
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
# 10. SPATIAL OPERATOR
# ============================================================

def spatial_operator(
    H,
    q,
    bottom,
    dx
):
    """
    Compute the finite-volume spatial operator.
    """

    (
        F1_minus,
        F2_minus,
        F1_plus,
        F2_plus
    ) = hydrostatic_muscl_flux(
        H,
        q,
        bottom,
        dx
    )

    dHdt = np.zeros_like(H)

    dqdt = np.zeros_like(q)

    dHdt[1:-1] = (
        -(
            F1_minus[1:]
            - F1_plus[:-1]
        )
        / dx
    )

    dqdt[1:-1] = (
        -(
            F2_minus[1:]
            - F2_plus[:-1]
        )
        / dx
    )

    return (
        dHdt,
        dqdt
    )


# ============================================================
# 11. CFL TIMESTEP
# ============================================================

def compute_dt(
    H,
    q,
    dx,
    CFL
):
    """
    Compute stable timestep.
    """

    velocity = np.zeros_like(H)

    wet = H > 1e-12

    velocity[wet] = (
        q[wet]
        / H[wet]
    )

    characteristic_speed = (
        np.abs(velocity)
        + np.sqrt(g * H)
    )

    maximum_speed = np.max(
        characteristic_speed
    )

    dt = (
        CFL
        * dx
        / maximum_speed
    )

    return dt


# ============================================================
# 12. SSP-RK2
# ============================================================

def advance_rk2(
    H,
    q,
    bottom,
    dt,
    dx
):
    """
    Two-stage SSP-RK2 integrator.
    """

    # Stage 1
    dHdt1, dqdt1 = spatial_operator(
        H,
        q,
        bottom,
        dx
    )

    H_stage = (
        H
        + dt * dHdt1
    )

    q_stage = (
        q
        + dt * dqdt1
    )

    # Stage 2
    dHdt2, dqdt2 = spatial_operator(
        H_stage,
        q_stage,
        bottom,
        dx
    )

    H_new = (
        0.5 * H
        + 0.5
        * (
            H_stage
            + dt * dHdt2
        )
    )

    q_new = (
        0.5 * q
        + 0.5
        * (
            q_stage
            + dt * dqdt2
        )
    )

    # Boundary conditions
    H_new[0] = H_new[1]
    H_new[-1] = H_new[-2]

    q_new[0] = q_new[1]
    q_new[-1] = q_new[-2]

    return (
        H_new,
        q_new
    )


# ============================================================
# 13. SINGLE SIMULATION
# ============================================================

def run_simulation(
    shallow_depth,
    return_times=False
):
    """
    Run one tsunami simulation for a specified
    shallow-water depth.

    Returns:
        amplification at 80 km
        plus optional diagnostic data.
    """

    # --------------------------------------------------------
    # Bathymetry
    # --------------------------------------------------------

    depth = create_bathymetry(
        x,
        shallow_depth,
        deep_depth,
        slope_start,
        slope_end
    )

    bottom = -depth

    # --------------------------------------------------------
    # Initial wave
    # --------------------------------------------------------

    eta = wave_amplitude * np.exp(
        -(
            (x - source_position) ** 2
        )
        / (
            2.0 * wave_sigma**2
        )
    )

    source_depth = np.interp(
        source_position,
        x,
        depth
    )

    source_speed = np.sqrt(
        g * source_depth
    )

    u = (
        source_speed
        / source_depth
    ) * eta

    H = depth + eta

    q = H * u

    # --------------------------------------------------------
    # Gauge locations
    # --------------------------------------------------------

    gauge_positions = np.array([
        40_000.0,
        80_000.0
    ])

    gauge_indices = [
        np.argmin(
            np.abs(
                x - position
            )
        )
        for position in gauge_positions
    ]

    gauge_history = {
        position: []
        for position in gauge_positions
    }

    time_history = []

    # --------------------------------------------------------
    # Expected arrival times
    # --------------------------------------------------------

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

    source_travel_time = np.interp(
        source_position,
        x,
        travel_time
    )

    expected_arrival_times = np.interp(
        gauge_positions,
        x,
        travel_time
    )

    expected_arrival_times -= (
        source_travel_time
    )

    # --------------------------------------------------------
    # Time integration
    # --------------------------------------------------------

    time = 0.0

    while time < total_time:

        eta_current = (
            H + bottom
        )

        # Record gauges
        for position, index in zip(
            gauge_positions,
            gauge_indices
        ):
            gauge_history[position].append(
                eta_current[index]
            )

        time_history.append(time)

        # CFL timestep
        dt = compute_dt(
            H,
            q,
            dx,
            CFL_target
        )

        dt = min(
            dt,
            total_time - time
        )

        # Advance
        H, q = advance_rk2(
            H,
            q,
            bottom,
            dt,
            dx
        )

        time += dt

    # --------------------------------------------------------
    # Incident-pulse extraction
    # --------------------------------------------------------

    amplitudes = []

    peak_times = []

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

        # Arrival window
        window_half_width = 250.0

        mask = (
            (times >= expected_time - window_half_width)
            &
            (times <= expected_time + window_half_width)
        )

        indices = np.where(mask)[0]

        if len(indices) == 0:

            raise RuntimeError(
                "No incident-wave window "
                f"found at {position / 1000:.0f} km."
            )

        local_index = indices[
            np.argmax(
                signal[indices]
            )
        ]

        amplitudes.append(
            signal[local_index]
        )

        peak_times.append(
            times[local_index]
        )

    amplitudes = np.array(
        amplitudes
    )

    peak_times = np.array(
        peak_times
    )

    reference_amplitude = (
        amplitudes[0]
    )

    amplification = (
        amplitudes[1]
        / reference_amplitude
    )

    if return_times:

        return {
            "shallow_depth": shallow_depth,
            "reference_amplitude": reference_amplitude,
            "target_amplitude": amplitudes[1],
            "amplification": amplification,
            "reference_peak_time": peak_times[0],
            "target_peak_time": peak_times[1]
        }

    return amplification


# ============================================================
# 14. BASELINE CHECK
# ============================================================

print()
print("==========================================")
print("          BASELINE CHECK")
print("==========================================")

baseline = run_simulation(
    shallow_depth=50.0,
    return_times=True
)

print(
    f"Shallow depth       : "
    f"{baseline['shallow_depth']:.2f} m"
)

print(
    f"40 km amplitude     : "
    f"{baseline['reference_amplitude']:.5f} m"
)

print(
    f"80 km amplitude     : "
    f"{baseline['target_amplitude']:.5f} m"
)

print(
    f"Amplification       : "
    f"{baseline['amplification']:.5f}x"
)

print(
    f"40 km peak time     : "
    f"{baseline['reference_peak_time']:.1f} s"
)

print(
    f"80 km peak time     : "
    f"{baseline['target_peak_time']:.1f} s"
)

print("==========================================")


# ============================================================
# 15. MONTE CARLO UNCERTAINTY EXPERIMENT
# ============================================================

number_of_samples = 100

minimum_shallow_depth = 45.0

maximum_shallow_depth = 55.0

random_seed = 42

rng = np.random.default_rng(
    random_seed
)

shallow_depth_samples = (
    rng.uniform(
        minimum_shallow_depth,
        maximum_shallow_depth,
        number_of_samples
    )
)

amplification_samples = []

print()
print("==========================================")
print("       MONTE CARLO SIMULATION")
print("==========================================")

for sample_number, shallow_depth_sample in enumerate(
    shallow_depth_samples,
    start=1
):

    amplification = run_simulation(
        shallow_depth=shallow_depth_sample
    )

    amplification_samples.append(
        amplification
    )

    print(
        f"Run {sample_number:02d}/"
        f"{number_of_samples}: "
        f"shallow depth = "
        f"{shallow_depth_sample:7.3f} m"
        f"  ->  amplification = "
        f"{amplification:.5f}x"
    )


amplification_samples = np.array(
    amplification_samples
)


# ============================================================
# 16. UNCERTAINTY STATISTICS
# ============================================================

mean_amplification = np.mean(
    amplification_samples
)

std_amplification = np.std(
    amplification_samples,
    ddof=1
)

minimum_amplification = np.min(
    amplification_samples
)

maximum_amplification = np.max(
    amplification_samples
)

p05 = np.percentile(
    amplification_samples,
    5
)

p50 = np.percentile(
    amplification_samples,
    50
)

p95 = np.percentile(
    amplification_samples,
    95
)


print()
print("==========================================")
print("       UNCERTAINTY RESULTS")
print("==========================================")

print(
    f"Mean amplification      : "
    f"{mean_amplification:.5f}x"
)

print(
    f"Standard deviation      : "
    f"{std_amplification:.5f}x"
)

print(
    f"Minimum                 : "
    f"{minimum_amplification:.5f}x"
)

print(
    f"Maximum                 : "
    f"{maximum_amplification:.5f}x"
)

print(
    f"5th percentile          : "
    f"{p05:.5f}x"
)

print(
    f"Median                  : "
    f"{p50:.5f}x"
)

print(
    f"95th percentile         : "
    f"{p95:.5f}x"
)

print("==========================================")


# ============================================================
# 17. DEPTH-AMplIFICATION CORRELATION
# ============================================================

correlation = np.corrcoef(
    shallow_depth_samples,
    amplification_samples
)[0, 1]

print(
    f"Depth-amplification correlation: "
    f"{correlation:.5f}"
)


# ============================================================
# 18. PLOT: UNCERTAIN BATHYMETRY EFFECT
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.scatter(
    shallow_depth_samples,
    amplification_samples
)

plt.xlabel(
    "Shallow-water depth (m)"
)

plt.ylabel(
    "Amplification at 80 km"
)

plt.title(
    "Effect of Shallow-Water Depth "
    "Uncertainty on Wave Amplification"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 19. PLOT: AMPLIFICATION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    amplification_samples,
    bins=8
)

plt.xlabel(
    "Amplification at 80 km"
)

plt.ylabel(
    "Number of simulations"
)

plt.title(
    "Monte Carlo Distribution of Wave Amplification"
)

plt.grid()

plt.tight_layout()

plt.show()