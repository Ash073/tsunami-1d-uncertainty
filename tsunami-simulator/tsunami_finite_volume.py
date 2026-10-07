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
    90_000.0
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
# 11. GAUGE AMPLITUDE ANALYSIS
# ============================================================

reference_position = gauge_positions[0]

reference_signal = np.array(
    gauge_history[reference_position]
)

reference_amplitude = np.max(
    np.abs(reference_signal)
)


print()
print("==========================================")
print("       WAVE GAUGE ANALYSIS")
print("==========================================")

print(
    f"Reference gauge: "
    f"{reference_position / 1000:.0f} km"
)

print(
    f"Reference amplitude: "
    f"{reference_amplitude:.6f} m"
)

print()

print(
    "Position (km) | Depth (m) | "
    "Max |eta| (m) | Amplification"
)

print("-" * 60)


gauge_amplitudes = []
gauge_depths = []

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

    gauge_amplitudes.append(
        maximum_amplitude
    )

    gauge_depths.append(
        gauge_depth
    )

    print(
        f"{position / 1000:10.0f} | "
        f"{gauge_depth:9.2f} | "
        f"{maximum_amplitude:13.6f} | "
        f"{amplification:13.4f}x"
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