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
dx = 50.0

x = np.arange(
    0.0,
    L + dx,
    dx
)

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

# Bottom elevation
#
# Free surface at rest:
#       z = 0
#
# Therefore:
#       bottom = -depth

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


# Initial surface displacement
eta = amplitude * np.exp(
    -((x - x0) ** 2)
    / (2.0 * sigma ** 2)
)


# Water depth at source location
initial_depth = np.interp(
    x0,
    x,
    depth
)


# Linear shallow-water speed
initial_speed = np.sqrt(
    g * initial_depth
)


# Right-going linear-wave velocity
u = (
    initial_speed
    / initial_depth
) * eta


# Conservative variables
#
# H = total water depth
# q = H*u

H = depth + eta
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
# 5. PRIMITIVE VARIABLES
# ============================================================

def primitive_variables(H, q, bottom):
    """
    Convert conservative variables to:

        eta = free-surface elevation
        u   = depth-averaged velocity
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
# 6. MINMOD LIMITER
# ============================================================

def minmod3(a, b, c):
    """
    Three-argument minmod limiter.

    The limiter prevents spurious oscillations near
    sharp gradients while allowing higher-order
    reconstruction in smooth regions.
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
# 7. MUSCL SLOPES
# ============================================================

def compute_primitive_slopes(
    eta,
    u,
    dx
):
    """
    Compute limited MUSCL slopes for
    free-surface elevation and velocity.
    """

    eta_slope = np.zeros_like(eta)
    u_slope = np.zeros_like(u)

    # --------------------------------------------------------
    # Free-surface slope
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

def physical_flux(H, q):
    """
    Shallow-water physical flux.

    F1 = q

    F2 = q^2/H + 0.5*g*H^2
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
    Compute interface fluxes using:

        1. primitive-variable MUSCL reconstruction
        2. hydrostatic reconstruction
        3. Rusanov flux

    Returns:

        F1_minus
        F2_minus
        F1_plus
        F2_plus
    """

    # --------------------------------------------------------
    # Convert to primitive variables
    # --------------------------------------------------------

    eta, u = primitive_variables(
        H,
        q,
        bottom
    )

    # --------------------------------------------------------
    # Compute limited slopes
    # --------------------------------------------------------

    eta_slope, u_slope = (
        compute_primitive_slopes(
            eta,
            u,
            dx
        )
    )

    # --------------------------------------------------------
    # Reconstruct left and right states
    # at every interface
    # --------------------------------------------------------

    eta_left = (
        eta[:-1]
        + 0.5
        * dx
        * eta_slope[:-1]
    )

    eta_right = (
        eta[1:]
        - 0.5
        * dx
        * eta_slope[1:]
    )

    u_left = (
        u[:-1]
        + 0.5
        * dx
        * u_slope[:-1]
    )

    u_right = (
        u[1:]
        - 0.5
        * dx
        * u_slope[1:]
    )

    # --------------------------------------------------------
    # Interface bottom
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

    # Reconstruct momentum
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
    # Local characteristic speeds
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
    # Rusanov numerical flux
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
    # Hydrostatic pressure corrections
    #
    # Use the reconstructed cell-side depths, not the
    # original cell-center depths.
    # --------------------------------------------------------

    H_left_cell = np.maximum(
        0.0,
        eta_left - bottom[:-1]
    )

    H_right_cell = np.maximum(
        0.0,
        eta_right - bottom[1:]
    )

    F2_minus = (
        numerical_F2
        + 0.5
        * g
        * (
            H_left_cell ** 2
            - H_left ** 2
        )
    )

    F2_plus = (
        numerical_F2
        + 0.5
        * g
        * (
            H_right_cell ** 2
            - H_right ** 2
        )
    )

    # Mass flux has no hydrostatic correction
    F1_minus = numerical_F1
    F1_plus = numerical_F1

    # Return all four interface fluxes
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
    Compute:

        dU/dt = L(U)

    using MUSCL + hydrostatic reconstruction.
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

    # For cell i:
    #
    # right interface  -> F_minus[i]
    # left interface   -> F_plus[i-1]

    dHdt[1:-1] = (
        -
        (
            F1_minus[1:]
            -
            F1_plus[:-1]
        )
        / dx
    )

    dqdt[1:-1] = (
        -
        (
            F2_minus[1:]
            -
            F2_plus[:-1]
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
    CFL_target,
    dx
):
    """
    CFL-controlled timestep.
    """

    u = np.zeros_like(H)

    wet = H > 1e-12

    u[wet] = (
        q[wet]
        / H[wet]
    )

    characteristic_speed = (
        np.abs(u)
        + np.sqrt(g * H)
    )

    maximum_speed = np.max(
        characteristic_speed
    )

    dt = (
        CFL_target
        * dx
        / maximum_speed
    )

    return (
        dt,
        maximum_speed
    )


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
    Two-stage SSP Runge-Kutta method.
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
        + 0.5 * (
            H_stage
            + dt * dHdt2
        )
    )

    q_new = (
        0.5 * q
        + 0.5 * (
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
# 13. STILL-WATER BALANCE TEST
# ============================================================

print()
print("==========================================")
print("       STILL-WATER BALANCE TEST")
print("==========================================")


H_rest = depth.copy()
q_rest = np.zeros_like(x)

CFL_target = 0.4

rest_test_time = 100.0
rest_time = 0.0

initial_rest_eta = (
    H_rest + bottom
).copy()

while rest_time < rest_test_time:

    dt_rest, _ = compute_dt(
        H_rest,
        q_rest,
        CFL_target,
        dx
    )

    dt_rest = min(
        dt_rest,
        rest_test_time - rest_time
    )

    H_rest, q_rest = advance_rk2(
        H_rest,
        q_rest,
        bottom,
        dt_rest,
        dx
    )

    rest_time += dt_rest


rest_eta = H_rest + bottom

rest_surface_error = np.max(
    np.abs(
        rest_eta
        - initial_rest_eta
    )
)

rest_velocity = np.divide(
    q_rest,
    H_rest,
    out=np.zeros_like(q_rest),
    where=H_rest > 1e-12
)

rest_velocity_error = np.max(
    np.abs(rest_velocity)
)


print(
    f"Maximum surface change : "
    f"{rest_surface_error:.6e} m"
)

print(
    f"Maximum velocity       : "
    f"{rest_velocity_error:.6e} m/s"
)

print("==========================================")


# ============================================================
# 14. RESET INITIAL TSUNAMI WAVE
# ============================================================

eta = amplitude * np.exp(
    -((x - x0) ** 2)
    / (2.0 * sigma ** 2)
)

u = (
    initial_speed
    / initial_depth
) * eta

H = depth + eta
q = H * u


# ============================================================
# 15. SIMULATION SETTINGS
# ============================================================

CFL_target = 0.4

total_time = 2600.0

time = 0.0


# Virtual wave gauges
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


# Snapshots for visualization
snapshot_times = np.arange(
    0.0,
    2600.0 + 200.0,
    200.0
)

snapshots = {}


# ============================================================
# 16. EXPECTED ARRIVAL TIMES
# ============================================================

local_speed = np.sqrt(
    g * depth
)

travel_time = np.zeros_like(x)

for i in range(1, N):

    travel_time[i] = (
        travel_time[i - 1]
        + 0.5
        * (
            1.0
            / local_speed[i - 1]
            +
            1.0
            / local_speed[i]
        )
        * dx
    )

source_travel_time = np.interp(
    x0,
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


# ============================================================
# 17. TIME INTEGRATION
# ============================================================

while time < total_time:

    # --------------------------------------------------------
    # Current free surface
    # --------------------------------------------------------

    eta_current = H + bottom

    # --------------------------------------------------------
    # Record gauges
    # --------------------------------------------------------

    for position, index in zip(
        gauge_positions,
        gauge_indices
    ):

        gauge_history[position].append(
            eta_current[index]
        )

    time_history.append(time)

    # --------------------------------------------------------
    # Save snapshots
    # --------------------------------------------------------

    for snapshot_time in snapshot_times:

        if (
            snapshot_time not in snapshots
            and time >= snapshot_time
        ):

            snapshots[snapshot_time] = (
                eta_current.copy()
            )

    # --------------------------------------------------------
    # CFL timestep
    # --------------------------------------------------------

    dt, maximum_speed = compute_dt(
        H,
        q,
        CFL_target,
        dx
    )

    dt = min(
        dt,
        total_time - time
    )

    # --------------------------------------------------------
    # Advance one timestep
    # --------------------------------------------------------

    H, q = advance_rk2(
        H,
        q,
        bottom,
        dt,
        dx
    )

    time += dt


# ============================================================
# 18. INCIDENT-PULSE ANALYSIS
# ============================================================

gauge_amplitudes = []
gauge_peak_times = []


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

    # Search around expected incident arrival
    window_half_width = 200.0

    window = (
        (times >= expected_time - window_half_width)
        &
        (times <= expected_time + window_half_width)
    )

    window_indices = np.where(
        window
    )[0]

    if len(window_indices) == 0:

        raise RuntimeError(
            f"No arrival window for "
            f"{position / 1000:.1f} km gauge."
        )

    # Search for the positive incident crest
    local_index = window_indices[
        np.argmax(
            signal[window_indices]
        )
    ]

    peak_amplitude = (
        signal[local_index]
    )

    peak_time = (
        times[local_index]
    )

    gauge_amplitudes.append(
        peak_amplitude
    )

    gauge_peak_times.append(
        peak_time
    )


gauge_amplitudes = np.array(
    gauge_amplitudes
)

gauge_peak_times = np.array(
    gauge_peak_times
)


# Relative amplification
reference_amplitude = (
    gauge_amplitudes[0]
)

amplification_factors = (
    gauge_amplitudes
    / reference_amplitude
)


# ============================================================
# 19. GREEN'S-LAW REFERENCE
# ============================================================

reference_depth = np.interp(
    gauge_positions[0],
    x,
    depth
)

gauge_depths = np.interp(
    gauge_positions,
    x,
    depth
)

green_amplification = (
    reference_depth
    / gauge_depths
) ** 0.25


# ============================================================
# 20. RESULTS
# ============================================================

print()
print("==========================================")
print("      MUSCL BATHYMETRY EXPERIMENT")
print("==========================================")

print(
    "Position | Depth | Expected t | "
    "Peak t | Peak eta | Amplification"
)

print("-" * 85)

for (
    position,
    gauge_depth,
    expected_time,
    peak_time,
    peak_amplitude,
    amplification
) in zip(
    gauge_positions,
    gauge_depths,
    expected_arrival_times,
    gauge_peak_times,
    gauge_amplitudes,
    amplification_factors
):

    print(
        f"{position / 1000:7.0f} km | "
        f"{gauge_depth:5.0f} m | "
        f"{expected_time:10.1f} s | "
        f"{peak_time:7.1f} s | "
        f"{peak_amplitude:9.5f} m | "
        f"{amplification:10.4f}x"
    )

print("==========================================")


print()
print("==========================================")
print("       GREEN'S-LAW COMPARISON")
print("==========================================")

print(
    "Position (km) | Simulated | Green's law"
)

print("-" * 50)

for position, simulated, theoretical in zip(
    gauge_positions,
    amplification_factors,
    green_amplification
):

    print(
        f"{position / 1000:13.0f} | "
        f"{simulated:9.4f}x | "
        f"{theoretical:9.4f}x"
    )

print("==========================================")


# ============================================================
# 21. BATHYMETRY PLOT
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    x / 1000.0,
    depth
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Distance (km)"
)

plt.ylabel(
    "Water depth (m)"
)

plt.title(
    "Variable Bathymetry"
)

plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 22. WAVE PROPAGATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)

for snapshot_time, wave in snapshots.items():

    plt.plot(
        x / 1000.0,
        wave,
        label=f"{snapshot_time:.0f} s"
    )

plt.xlabel(
    "Distance (km)"
)

plt.ylabel(
    "Surface displacement η (m)"
)

plt.title(
    "MUSCL Wave Propagation Over Variable Bathymetry"
)

plt.legend(
    ncol=2
)

plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 23. WAVE-GAUGE TIME SERIES
# ============================================================

plt.figure(
    figsize=(12, 7)
)

for position in gauge_positions:

    plt.plot(
        np.array(time_history),
        np.array(
            gauge_history[position]
        ),
        label=f"{position / 1000:.0f} km"
    )

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Surface displacement η (m)"
)

plt.title(
    "MUSCL Wave-Gauge Measurements"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# 24. AMPLIFICATION COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    gauge_positions / 1000.0,
    amplification_factors,
    marker="o",
    label="MUSCL simulation"
)

plt.plot(
    gauge_positions / 1000.0,
    green_amplification,
    marker="s",
    linestyle="--",
    label="Green's-law reference"
)

plt.xlabel(
    "Distance (km)"
)

plt.ylabel(
    "Relative amplification"
)

plt.title(
    "MUSCL Shoaling vs Green's-Law Reference"
)

plt.legend()

plt.grid()
plt.tight_layout()
plt.show()