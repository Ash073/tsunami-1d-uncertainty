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
dx = 25.0

x = np.arange(
    0.0,
    L + dx,
    dx
)

N = len(x)


# ============================================================
# 3. CONSTANT WATER DEPTH
# ============================================================

depth = 50.0


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


# Shallow-water wave speed
c = np.sqrt(
    g * depth
)


# Right-going linear-wave velocity
u = (
    c / depth
) * eta


# Total water depth
H = depth + eta


# Conserved momentum
q = H * u


print(
    f"Water depth: "
    f"{depth:.2f} m"
)

print(
    f"Theoretical wave speed: "
    f"{c:.4f} m/s"
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
# 6. RUSANOV FLUX
# ============================================================

def rusanov_flux(
    H_left,
    q_left,
    H_right,
    q_right
):
    """
    Compute the Rusanov numerical flux
    at all cell interfaces.
    """

    F1_left, F2_left = flux(
        H_left,
        q_left
    )

    F1_right, F2_right = flux(
        H_right,
        q_right
    )

    velocity_left = np.zeros_like(
        H_left
    )

    velocity_right = np.zeros_like(
        H_right
    )

    wet_left = H_left > 1e-12
    wet_right = H_right > 1e-12

    velocity_left[wet_left] = (
        q_left[wet_left]
        / H_left[wet_left]
    )

    velocity_right[wet_right] = (
        q_right[wet_right]
        / H_right[wet_right]
    )

    wave_speed_left = (
        np.abs(velocity_left)
        + np.sqrt(g * H_left)
    )

    wave_speed_right = (
        np.abs(velocity_right)
        + np.sqrt(g * H_right)
    )

    interface_speed = np.maximum(
        wave_speed_left,
        wave_speed_right
    )

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

    return (
        numerical_F1,
        numerical_F2
    )


# ============================================================
# 7. FINITE-VOLUME UPDATE
# ============================================================

def finite_volume_step(
    H,
    q,
    dt,
    dx
):
    """
    Advance the constant-depth
    shallow-water solution by one timestep.
    """

    F1, F2 = rusanov_flux(
        H[:-1],
        q[:-1],
        H[1:],
        q[1:]
    )

    H_new = H.copy()
    q_new = q.copy()

    # Conservative update
    H_new[1:-1] = (
        H[1:-1]
        -
        (dt / dx)
        * (
            F1[1:]
            -
            F1[:-1]
        )
    )

    q_new[1:-1] = (
        q[1:-1]
        -
        (dt / dx)
        * (
            F2[1:]
            -
            F2[:-1]
        )
    )

    # Boundary conditions
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
# 9. WAVE GAUGES
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


# ============================================================
# 10. SIMULATION
# ============================================================

total_time = 3400.0

num_steps = int(
    total_time / dt
)

time = 0.0


for step in range(num_steps + 1):

    # --------------------------------------------------------
    # Record gauge measurements
    # --------------------------------------------------------

    eta_current = H - depth

    for position, index in zip(
        gauge_positions,
        gauge_indices
    ):
        gauge_history[position].append(
            eta_current[index]
        )

    time_history.append(time)

    # --------------------------------------------------------
    # Stop at final step
    # --------------------------------------------------------

    if step == num_steps:
        break

    # --------------------------------------------------------
    # Advance solution
    # --------------------------------------------------------

    H, q = finite_volume_step(
        H,
        q,
        dt,
        dx
    )

    time += dt


# ============================================================
# 11. NUMERICAL-DAMPING ANALYSIS
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


reference_amplitude = (
    gauge_amplitudes[0]
)


relative_amplitudes = (
    gauge_amplitudes
    / reference_amplitude
)


print()
print("==========================================")
print("      CONSTANT-DEPTH CONTROL")
print("==========================================")

print(
    "Position (km) | Max |eta| (m) | "
    "Relative amplitude"
)

print("-" * 55)

for position, amplitude_value, relative in zip(
    gauge_positions,
    gauge_amplitudes,
    relative_amplitudes
):

    print(
        f"{position / 1000:10.0f} | "
        f"{amplitude_value:13.6f} | "
        f"{relative:17.4f}x"
    )

print("==========================================")


# ============================================================
# 12. GAUGE TIME SERIES
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
plt.ylabel(
    "Surface displacement η (m)"
)

plt.title(
    "Constant-Depth Control: Wave Gauges"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# 13. NUMERICAL DAMPING PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    gauge_positions / 1000.0,
    relative_amplitudes,
    marker="o"
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Distance (km)")
plt.ylabel(
    "Relative amplitude"
)

plt.title(
    "Numerical Amplitude Change "
    "in Constant 50 m Depth"
)

plt.grid()
plt.tight_layout()
plt.show()