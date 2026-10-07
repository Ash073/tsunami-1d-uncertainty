import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

g = 9.81


# ============================================================
# 2. COMPUTATIONAL DOMAIN
# ============================================================

L = 100_000.0      # Domain length (m)
dx = 50.0         # Grid spacing (m)

x = np.arange(0.0, L + dx, dx)
N = len(x)


# ============================================================
# 3. CONSTANT WATER DEPTH
# ============================================================

h0 = 100.0         # Undisturbed water depth (m)

print(f"Water depth: {h0:.2f} m")


# ============================================================
# 4. INITIAL WAVE
# ============================================================

amplitude = 0.01
x0 = 20_000.0
sigma = 2_000.0

eta = amplitude * np.exp(
    -((x - x0) ** 2) / (2.0 * sigma ** 2)
)


# ============================================================
# 5. CONSERVATIVE VARIABLES
# ============================================================

# Total water depth
H = h0 + eta

# Linear right-going shallow-water wave speed
c = np.sqrt(g * h0)

# Velocity corresponding to a right-going linear wave
u = (c / h0) * eta

# Conserved momentum
q = H * u


print(f"Theoretical wave speed: {c:.4f} m/s")
print(f"Initial maximum elevation: {np.max(eta):.4f} m")
print(f"Initial maximum velocity: {np.max(u):.4f} m/s")


# ============================================================
# 6. FLUX FUNCTION
# ============================================================

def flux(H, q):
    """
    Compute the physical shallow-water flux.

    Conservative variables:
        H = total water depth
        q = H * u = water-column momentum

    Flux:
        F1 = q
        F2 = q^2 / H + 0.5 * g * H^2
    """

    u = q / H

    F1 = q

    F2 = (
        q * u
        + 0.5 * g * H**2
    )

    return F1, F2


# ============================================================
# 7. RUSANOV NUMERICAL FLUX
# ============================================================

def rusanov_flux(H_left, q_left, H_right, q_right):
    """
    Compute the Rusanov (local Lax-Friedrichs) flux
    at an interface between two cells.
    """

    # Physical flux on the left side
    F1_left, F2_left = flux(
        H_left,
        q_left
    )

    # Physical flux on the right side
    F1_right, F2_right = flux(
        H_right,
        q_right
    )

    # Velocities on both sides
    u_left = q_left / H_left
    u_right = q_right / H_right

    # Local wave speeds
    a_left = (
        np.abs(u_left)
        + np.sqrt(g * H_left)
    )

    a_right = (
        np.abs(u_right)
        + np.sqrt(g * H_right)
    )

    # Maximum propagation speed at each interface
    a = np.maximum(
        a_left,
        a_right
    )

    # Rusanov flux
    numerical_F1 = 0.5 * (
        F1_left
        + F1_right
        - a * (H_right - H_left)
    )

    numerical_F2 = 0.5 * (
        F2_left
        + F2_right
        - a * (q_right - q_left)
    )

    return numerical_F1, numerical_F2

# ============================================================
# 8. TIME STEP
# ============================================================

CFL_target = 0.5

# Maximum local wave speed:
# |u| + sqrt(gH)
maximum_speed = (
    np.max(np.abs(u))
    + np.sqrt(g * np.max(H))
)

dt = CFL_target * dx / maximum_speed

CFL = maximum_speed * dt / dx

print(f"Maximum wave speed: {maximum_speed:.4f} m/s")
print(f"Time step: {dt:.4f} s")
print(f"Maximum CFL number: {CFL:.3f}")

# ============================================================
# 9. FINITE-VOLUME UPDATE
# ============================================================

def finite_volume_step(H, q, dt, dx):
    """
    Advance the shallow-water solution by one time step
    using the Rusanov finite-volume flux.
    """

    # --------------------------------------------------------
    # 1. Compute fluxes at cell interfaces
    # --------------------------------------------------------

    F1, F2 = rusanov_flux(
        H[:-1],
        q[:-1],
        H[1:],
        q[1:]
    )

    # --------------------------------------------------------
    # 2. Copy current state
    # --------------------------------------------------------

    H_new = H.copy()
    q_new = q.copy()

    # --------------------------------------------------------
    # 3. Conservative finite-volume update
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. Boundary conditions
    # --------------------------------------------------------

    H_new[0] = H_new[1]
    H_new[-1] = H_new[-2]

    q_new[0] = q_new[1]
    q_new[-1] = q_new[-2]

    return H_new, q_new
# ============================================================
# 10. SIMULATION SETTINGS
# ============================================================

total_time = 1000.0

num_steps = int(
    total_time / dt
)

snapshot_times = [
    0,
    200,
    400,
    600,
    800,
    1000
]

snapshots = {}


# ============================================================
# 11. TIME INTEGRATION
# ============================================================

time = 0.0

for step in range(num_steps + 1):

    # --------------------------------------------------------
    # Save snapshots
    # --------------------------------------------------------

    for snapshot_time in snapshot_times:

        if (
            abs(time - snapshot_time)
            < dt / 2
            and snapshot_time not in snapshots
        ):

            eta_snapshot = H - h0

            snapshots[snapshot_time] = (
                eta_snapshot.copy()
            )

    # --------------------------------------------------------
    # Stop after final snapshot
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

    # --------------------------------------------------------
    # Update time
    # --------------------------------------------------------

    time += dt
# ============================================================
# 12. NUMERICAL WAVE-SPEED MEASUREMENT
# ============================================================

snapshot_times_sorted = sorted(snapshots.keys())

peak_positions = []

for snapshot_time in snapshot_times_sorted:

    wave = snapshots[snapshot_time]

    # Search for the main right-going wave
    # to the right of the initial wave position.
    valid_region = x > x0

    valid_indices = np.where(valid_region)[0]

    local_peak_index = valid_indices[
        np.argmax(
            wave[valid_indices]
        )
    ]

   # ============================================================
# SUB-GRID PEAK LOCATION
# ============================================================

peak_positions = []

for snapshot_time in snapshot_times_sorted:

    wave = snapshots[snapshot_time]

    valid_region = x > x0
    valid_indices = np.where(valid_region)[0]

    local_peak_index = valid_indices[
        np.argmax(wave[valid_indices])
    ]

    # Default: grid-point location
    peak_position = x[local_peak_index]

    # Quadratic interpolation around the peak
    if 0 < local_peak_index < N - 1:

        y1 = wave[local_peak_index - 1]
        y2 = wave[local_peak_index]
        y3 = wave[local_peak_index + 1]

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
                x[local_peak_index]
                + offset * dx
            )

    peak_positions.append(peak_position)


# Convert to arrays
snapshot_times_array = np.array(
    snapshot_times_sorted
)

peak_positions_array = np.array(
    peak_positions
)


# Fit a straight line:
# position = speed * time + constant

wave_speed_numerical, intercept = np.polyfit(
    snapshot_times_array,
    peak_positions_array,
    1
)


speed_error = (
    abs(
        wave_speed_numerical - c
    )
    / c
) * 100.0


print()
print("==========================================")
print("       FINITE-VOLUME VALIDATION")
print("==========================================")

print(
    f"Theoretical wave speed : "
    f"{c:.4f} m/s"
)

print(
    f"Numerical wave speed   : "
    f"{wave_speed_numerical:.4f} m/s"
)

print(
    f"Wave-speed error       : "
    f"{speed_error:.2f}%"
)

print("==========================================")