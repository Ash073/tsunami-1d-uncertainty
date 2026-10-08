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
# 3. CONSTANT WATER DEPTH
# ============================================================

h0 = 50.0


# ============================================================
# 4. INITIAL RIGHT-GOING WAVE
# ============================================================

# Start with a small amplitude so that we are testing
# the nearly-linear limit.

amplitude = 1.0

x0 = 20_000.0
sigma = 2_000.0


eta = amplitude * np.exp(
    -((x - x0) ** 2)
    / (2.0 * sigma ** 2)
)


# Linear shallow-water wave speed
c_theory = np.sqrt(
    g * h0
)


# Right-going linear-wave velocity
u = (
    c_theory / h0
) * eta


# Total water depth
H = h0 + eta


# Conserved momentum
q = H * u


# Conservative state:
#
# U[0] = H
# U[1] = q

U = np.vstack(
    (
        H,
        q
    )
)


print(
    f"Water depth: "
    f"{h0:.2f} m"
)

print(
    f"Theoretical wave speed: "
    f"{c_theory:.4f} m/s"
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
# 5. MINMOD LIMITER
# ============================================================

def minmod3(a, b, c):
    """
    Three-argument minmod slope limiter.

    The limiter suppresses new oscillations near
    steep gradients while retaining higher-order
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
# 6. MUSCL SLOPE RECONSTRUCTION
# ============================================================

def compute_slopes(U, dx):
    """
    Compute limited slopes for all conservative variables.
    """

    slopes = np.zeros_like(U)

    for k in range(U.shape[0]):

        backward = (
            U[k, 1:-1]
            - U[k, :-2]
        ) / dx

        centered = (
            U[k, 2:]
            - U[k, :-2]
        ) / (2.0 * dx)

        forward = (
            U[k, 2:]
            - U[k, 1:-1]
        ) / dx

        slopes[k, 1:-1] = minmod3(
            backward,
            centered,
            forward
        )

    return slopes


# ============================================================
# 7. PHYSICAL FLUX
# ============================================================

def physical_flux(U):
    """
    Compute the physical shallow-water flux.

    U = [H, q]

    F = [
        q,
        q^2/H + 0.5*g*H^2
    ]
    """

    H = U[0]
    q = U[1]

    velocity = np.zeros_like(H)

    wet = H > 1e-12

    velocity[wet] = (
        q[wet] / H[wet]
    )

    F = np.zeros_like(U)

    F[0] = q

    F[1] = (
        q * velocity
        + 0.5 * g * H**2
    )

    return F


# ============================================================
# 8. RUSANOV FLUX
# ============================================================

def rusanov_flux(U_left, U_right):
    """
    Rusanov numerical flux between reconstructed
    left and right interface states.
    """

    H_left = U_left[0]
    q_left = U_left[1]

    H_right = U_right[0]
    q_right = U_right[1]

    F_left = physical_flux(
        U_left
    )

    F_right = physical_flux(
        U_right
    )

    u_left = np.zeros_like(
        H_left
    )

    u_right = np.zeros_like(
        H_right
    )

    wet_left = H_left > 1e-12
    wet_right = H_right > 1e-12

    u_left[wet_left] = (
        q_left[wet_left]
        / H_left[wet_left]
    )

    u_right[wet_right] = (
        q_right[wet_right]
        / H_right[wet_right]
    )

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

    numerical_flux = 0.5 * (
        F_left
        + F_right
        -
        interface_speed
        * (
            U_right
            - U_left
        )
    )

    return numerical_flux


# ============================================================
# 9. SPATIAL OPERATOR
# ============================================================

def spatial_operator(U, dx):
    """
    Calculate the finite-volume spatial operator

        dU/dt = L(U)

    using MUSCL reconstruction + Rusanov flux.
    """

    slopes = compute_slopes(
        U,
        dx
    )

    # --------------------------------------------------------
    # Reconstruct left and right states at interfaces
    # --------------------------------------------------------

    U_left = (
        U[:, :-1]
        + 0.5 * dx * slopes[:, :-1]
    )

    U_right = (
        U[:, 1:]
        - 0.5 * dx * slopes[:, 1:]
    )

    # --------------------------------------------------------
    # Numerical fluxes
    # --------------------------------------------------------

    F = rusanov_flux(
        U_left,
        U_right
    )

    # --------------------------------------------------------
    # Flux divergence
    # --------------------------------------------------------

    L_operator = np.zeros_like(U)

    L_operator[:, 1:-1] = (
        -
        (
            F[:, 1:]
            -
            F[:, :-1]
        )
        / dx
    )

    return L_operator


# ============================================================
# 10. CFL TIME STEP
# ============================================================

def compute_dt(U, dx, CFL_target):
    """
    Compute a stable timestep from the maximum
    local characteristic speed.
    """

    H = U[0]
    q = U[1]

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
        CFL_target
        * dx
        / maximum_speed
    )

    return dt, maximum_speed


# ============================================================
# 11. SSP-RK2 TIME INTEGRATOR
# ============================================================

def advance_rk2(U, dt, dx):
    """
    Two-stage strong-stability-preserving
    Runge-Kutta method.
    """

    # Stage 1
    k1 = spatial_operator(
        U,
        dx
    )

    U_stage = U + dt * k1

    # Stage 2
    k2 = spatial_operator(
        U_stage,
        dx
    )

    U_new = (
        0.5 * U
        +
        0.5 * (
            U_stage
            + dt * k2
        )
    )

    # Boundary conditions
    U_new[:, 0] = U_new[:, 1]
    U_new[:, -1] = U_new[:, -2]

    return U_new


# ============================================================
# 12. SIMULATION SETTINGS
# ============================================================

CFL_target = 0.4

total_time = 3400.0

time = 0.0

snapshot_times = np.arange(
    0.0,
    1000.0 + 50.0,
    50.0
)

snapshots = {}

# ============================================================
# WAVE GAUGES
# ============================================================

gauge_positions = np.array([
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

peak_time_history = []
peak_position_history = []

# ============================================================
# 13. TIME INTEGRATION
# ============================================================

while time <= total_time:

    # --------------------------------------------------------
    # Save snapshot
    # --------------------------------------------------------

    for snapshot_time in snapshot_times:

        if (
            snapshot_time not in snapshots
            and time >= snapshot_time
        ):
            snapshots[snapshot_time] = (
                U[0] - h0
            ).copy()

    # --------------------------------------------------------
    # Stop condition
    # --------------------------------------------------------

    if time >= total_time:
        break

    # --------------------------------------------------------
    # CFL timestep
    # --------------------------------------------------------

    dt, maximum_speed = compute_dt(
        U,
        dx,
        CFL_target
    )

    # Prevent overshooting final time
    dt = min(
        dt,
        total_time - time
    )

        # --------------------------------------------------------
    # Advance solution
    # --------------------------------------------------------

    U = advance_rk2(
        U,
        dt,
        dx
    )

    time += dt

    # --------------------------------------------------------
    # Record wave-gauge measurements
    # --------------------------------------------------------

    eta_current = U[0] - h0

    for position, index in zip(
        gauge_positions,
        gauge_indices
    ):

        gauge_history[position].append(
            eta_current[index]
        )

    time_history.append(time)

    # --------------------------------------------------------
    # Track the main right-going wave
    # --------------------------------------------------------

    eta_current = U[0] - h0

    valid_indices = np.where(
        x > x0
    )[0]

    local_index = valid_indices[
        np.argmax(
            eta_current[valid_indices]
        )
    ]

    peak_position = x[local_index]

    # --------------------------------------------------------
    # Sub-grid quadratic interpolation
    # --------------------------------------------------------

    if 0 < local_index < N - 1:

        y1 = eta_current[local_index - 1]
        y2 = eta_current[local_index]
        y3 = eta_current[local_index + 1]

        denominator = (
            y1
            - 2.0 * y2
            + y3
        )

        if abs(denominator) > 1e-14:

            offset = 0.5 * (
                (y1 - y3)
                / denominator
            )

            peak_position = (
                x[local_index]
                + offset * dx
            )

    peak_time_history.append(time)
    peak_position_history.append(
        peak_position
    )

# ============================================================
# 14. CONTINUOUS WAVE-SPEED MEASUREMENT
# ============================================================

peak_time_history = np.array(
    peak_time_history
)

peak_position_history = np.array(
    peak_position_history
)


# Fit only after the wave has clearly
# moved away from its initial position.

fit_mask = (
    (peak_time_history >= 100.0)
    &
    (peak_time_history <= 900.0)
)


if np.count_nonzero(fit_mask) < 2:

    raise RuntimeError(
        "Not enough wave-peak measurements "
        "for wave-speed fitting."
    )


wave_speed_numerical, intercept = (
    np.polyfit(
        peak_time_history[fit_mask],
        peak_position_history[fit_mask],
        1
    )
)


speed_error = (
    abs(
        wave_speed_numerical
        - c_theory
    )
    / c_theory
) * 100.0


# ============================================================
# 15. VALIDATION RESULTS
# ============================================================

print()
print("==========================================")
print("       MUSCL FINITE-VOLUME VALIDATION")
print("==========================================")

print(
    f"Theoretical wave speed : "
    f"{c_theory:.4f} m/s"
)

print(
    f"Numerical wave speed   : "
    f"{wave_speed_numerical:.4f} m/s"
)

print(
    f"Wave-speed error       : "
    f"{speed_error:.4f}%"
)

print(
    f"Peak measurements used : "
    f"{np.count_nonzero(fit_mask)}"
)

print(
    f"Final simulation time  : "
    f"{time:.2f} s"
)

print("==========================================")

# ============================================================
# 16. WAVE PROPAGATION PLOT
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
    "MUSCL Finite-Volume Wave Propagation"
)

plt.legend(
    ncol=2
)

plt.grid()

plt.tight_layout()

plt.show()

# ============================================================
# CONSTANT-DEPTH MUSCL CONTROL
# ============================================================

gauge_amplitudes = []

for position in gauge_positions:

    signal = np.array(
        gauge_history[position]
    )

    maximum_amplitude = np.max(
        np.abs(signal)
    )

    gauge_amplitudes.append(
        maximum_amplitude
    )

gauge_amplitudes = np.array(
    gauge_amplitudes
)

relative_amplitudes = (
    gauge_amplitudes
    / gauge_amplitudes[0]
)


print()
print("==========================================")
print("        MUSCL CONSTANT-DEPTH CONTROL")
print("==========================================")

print(
    "Position (km) | "
    "Max |eta| (m) | "
    "Relative amplitude"
)

print("-" * 60)

for position, amplitude_value, relative in zip(
    gauge_positions,
    gauge_amplitudes,
    relative_amplitudes
):

    print(
        f"{position / 1000:13.0f} | "
        f"{amplitude_value:13.6f} | "
        f"{relative:18.4f}x"
    )

print("==========================================")

# ============================================================
# GAUGE TIME SERIES
# ============================================================

plt.figure(figsize=(12, 6))

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
    "MUSCL Constant-Depth Control"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()