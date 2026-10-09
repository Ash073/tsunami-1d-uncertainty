"""
Virtual visualization layer for the existing 1D nonlinear shallow-water solver.

This runs one baseline simulation, records actual free-surface snapshots, and
creates:
  - a 3D-style animated ribbon of the 1D wave over its bathymetric profile,
  - a space-time wave-elevation plot,
  - gauge time-series plots,
  - a compressed archive of the actual numerical snapshots.

Important:
  This is a 3D visualization of a 1D model, not a 3D fluid simulation.
  The display-width axis is an illustration-only extrusion. The free-surface
  anomaly is vertically exaggerated for visibility and that factor is stated
  in the plot labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# Physical/model settings
g = 9.81
L = 100_000.0
dx = 50.0
slope_start = 50_000.0
source_position = 20_000.0
wave_amplitude = 1.0
wave_sigma = 2_000.0
CFL_target = 0.4

x = np.arange(0.0, L + dx, dx)
N = len(x)


def create_bathymetry(deep_depth, shallow_depth, slope_end):
    """
    Create a piecewise-linear bathymetry.

    Returns positive water depth. The numerical solver later
    converts it to the bed elevation using bottom = -depth.

    Parameters
    ----------
    deep_depth : float
        Constant deep-water depth before the slope.
    shallow_depth : float
        Constant shallow-water depth after the slope.
    slope_end : float
        x-location where the slope reaches shallow_depth.
    """

    depth = np.full_like(x, deep_depth, dtype=float)

    slope_mask = (x >= slope_start) & (x < slope_end)

    depth[slope_mask] = (
        deep_depth
        + (shallow_depth - deep_depth)
        * (x[slope_mask] - slope_start)
        / (slope_end - slope_start)
    )

    shallow_mask = x >= slope_end
    depth[shallow_mask] = shallow_depth

    return depth


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

def run_visualization_simulation(
    deep_depth: float,
    shallow_depth: float,
    slope_end: float,
    total_time: float,
    number_of_snapshots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one real model simulation and capture free-surface snapshots."""
    if deep_depth <= shallow_depth:
        raise ValueError("deep depth must be greater than shallow depth.")
    if shallow_depth <= 0:
        raise ValueError("shallow depth must be positive.")
    if not (slope_start < slope_end < L):
        raise ValueError("slope-end must lie between slope start and domain length.")
    if total_time <= 0:
        raise ValueError("total time must be positive.")
    if number_of_snapshots < 2:
        raise ValueError("At least two snapshots are required.")

    depth = create_bathymetry(deep_depth, shallow_depth, slope_end)
    bottom = -depth

    eta = wave_amplitude * np.exp(
        -((x - source_position) ** 2) / (2.0 * wave_sigma**2)
    )
    source_depth = np.interp(source_position, x, depth)
    source_speed = np.sqrt(g * source_depth)
    u = (source_speed / source_depth) * eta

    H = depth + eta
    q = H * u

    target_times = np.linspace(0.0, total_time, number_of_snapshots)
    captured_times = [0.0]
    captured_eta = [eta.copy()]
    next_target = 1
    time = 0.0

    while time < total_time:
        dt = compute_dt(H, q, dx, CFL_target)
        dt = min(dt, total_time - time)

        H, q = advance_rk2(H, q, bottom, dt, dx)
        time += dt
        eta_current = H + bottom

        # Save the current model state when it passes the next target time.
        while next_target < len(target_times) and time >= target_times[next_target]:
            captured_times.append(float(time))
            captured_eta.append(eta_current.copy())
            next_target += 1

    if captured_times[-1] < time:
        captured_times.append(float(time))
        captured_eta.append((H + bottom).copy())

    return x.copy(), depth.copy(), np.asarray(captured_times), np.asarray(captured_eta)


def make_3d_axes(ax, x_km, depth, eta, vertical_exaggeration, title):
    """Render a 1D domain as a 3D-style ribbon; y is only a display width."""
    # Downsample only for plotting; the saved snapshots retain the full grid.
    stride = max(1, len(x_km) // 500)
    xp = x_km[::stride]
    dp = depth[::stride]
    ep = eta[::stride]

    X = np.broadcast_to(xp, (2, len(xp)))
    Y = np.broadcast_to(np.array([[0.0], [1.0]]), (2, len(xp)))
    bed_z = np.broadcast_to(-dp, (2, len(xp)))
    wave_z = np.broadcast_to(ep * vertical_exaggeration, (2, len(xp)))

    ax.plot_surface(X, Y, bed_z, alpha=0.55, linewidth=0, antialiased=True)
    ax.plot_surface(X, Y, wave_z, alpha=0.95, linewidth=0, antialiased=True)

    for gauge_km in (40.0, 80.0):
        idx = int(np.argmin(np.abs(x_km - gauge_km)))
        ax.plot(
            [gauge_km, gauge_km],
            [0.5, 0.5],
            [-float(depth[idx]), 0.0],
            linestyle="--",
            linewidth=1.0,
        )
        ax.text(gauge_km, 0.5, 3.0, f"{gauge_km:.0f} km gauge")

    ax.scatter([source_position / 1000.0], [0.5], [0.0], marker="*", s=65)
    ax.set_xlim(float(x_km[0]), float(x_km[-1]))
    ax.set_ylim(-0.15, 1.15)
    ax.set_zlim(-float(np.max(depth)) - 15.0, 30.0)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Display width (illustration only)")
    ax.set_zlabel(f"Elevation (m; wave anomaly ×{vertical_exaggeration:g})")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-62)
    try:
        ax.set_box_aspect((100, 9, 35))
    except AttributeError:
        pass
    ax.grid(True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one 1D tsunami simulation and create a 3D-style visualization."
    )
    parser.add_argument("--deep-depth", type=float, default=200.0, help="Deep depth in metres.")
    parser.add_argument("--shallow-depth", type=float, default=50.0, help="Shallow depth in metres.")
    parser.add_argument("--slope-end-km", type=float, default=80.0, help="End of slope in km.")
    parser.add_argument("--total-time", type=float, default=2200.0, help="Simulation duration in seconds.")
    parser.add_argument("--frames", type=int, default=91, help="Approximate number of snapshots.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate.")
    parser.add_argument(
        "--vertical-exaggeration",
        type=float,
        default=20.0,
        help="Visual multiplier applied to free-surface anomaly only.",
    )
    parser.add_argument("--no-gif", action="store_true", help="Create static plots/data without GIF animation.")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "uq_outputs" / "virtual_lab"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running one simulation and recording wave snapshots...")
    x_m, depth_m, times_s, eta_m = run_visualization_simulation(
        deep_depth=args.deep_depth,
        shallow_depth=args.shallow_depth,
        slope_end=args.slope_end_km * 1000.0,
        total_time=args.total_time,
        number_of_snapshots=args.frames,
    )

    np.savez_compressed(
        output_dir / "virtual_tsunami_snapshots.npz",
        x_m=x_m,
        depth_m=depth_m,
        times_s=times_s,
        eta_m=eta_m,
        deep_depth_m=np.asarray(args.deep_depth),
        shallow_depth_m=np.asarray(args.shallow_depth),
        slope_end_km=np.asarray(args.slope_end_km),
        vertical_exaggeration=np.asarray(args.vertical_exaggeration),
    )

    # Space-time diagram from actual model snapshots.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    mesh = ax.pcolormesh(x_m / 1000.0, times_s, eta_m, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Free-surface elevation η (m)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Tsunami Wave Propagation: Space-Time Diagram")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_dir / "space_time_wave.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Time histories at the two gauges used in the UQ experiment.
    fig, ax = plt.subplots(figsize=(9, 5))
    for gauge_km in (40.0, 80.0):
        idx = int(np.argmin(np.abs(x_m / 1000.0 - gauge_km)))
        ax.plot(times_s, eta_m[:, idx], label=f"{gauge_km:.0f} km gauge")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Free-surface elevation η (m)")
    ax.set_title("Gauge Time Series")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gauge_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # One still frame from the actual computed fields.
    frame_idx = max(0, min(len(times_s) - 1, len(times_s) // 2))
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    make_3d_axes(
        ax,
        x_m / 1000.0,
        depth_m,
        eta_m[frame_idx],
        args.vertical_exaggeration,
        f"3D-Style View of the 1D Model (t = {times_s[frame_idx]:.1f} s)",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "virtual_tsunami_3d_snapshot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    gif_path = output_dir / "virtual_tsunami_3d.gif"
    if not args.no_gif:
        print(f"Rendering animation ({len(times_s)} saved states)...")
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        stride = max(1, len(x_m) // 500)
        xp = x_m[::stride] / 1000.0
        dp = depth_m[::stride]
        X = np.broadcast_to(xp, (2, len(xp)))
        Y = np.broadcast_to(np.array([[0.0], [1.0]]), (2, len(xp)))
        bed_z = np.broadcast_to(-dp, (2, len(xp)))
        ax.plot_surface(X, Y, bed_z, alpha=0.55, linewidth=0, antialiased=True)

        for gauge_km in (40.0, 80.0):
            idx = int(np.argmin(np.abs(x_m / 1000.0 - gauge_km)))
            ax.plot(
                [gauge_km, gauge_km],
                [0.5, 0.5],
                [-float(depth_m[idx]), 0.0],
                linestyle="--",
                linewidth=1.0,
            )
            ax.text(gauge_km, 0.5, 3.0, f"{gauge_km:.0f} km gauge")

        ax.scatter([source_position / 1000.0], [0.5], [0.0], marker="*", s=65)
        ax.set_xlim(float(x_m[0] / 1000.0), float(x_m[-1] / 1000.0))
        ax.set_ylim(-0.15, 1.15)
        ax.set_zlim(-float(np.max(depth_m)) - 15.0, 30.0)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Display width (illustration only)")
        ax.set_zlabel(f"Elevation (m; wave anomaly ×{args.vertical_exaggeration:g})")
        ax.view_init(elev=24, azim=-62)
        try:
            ax.set_box_aspect((100, 9, 35))
        except AttributeError:
            pass

        wave_surface = [None]

        def update(frame: int):
            if wave_surface[0] is not None:
                wave_surface[0].remove()
            ep = eta_m[frame, ::stride] * args.vertical_exaggeration
            wave_z = np.broadcast_to(ep, (2, len(xp)))
            wave_surface[0] = ax.plot_surface(
                X, Y, wave_z, alpha=0.95, linewidth=0, antialiased=True
            )
            ax.set_title(
                f"Virtual Tsunami Lab — 1D physics, 3D-style view | t = {times_s[frame]:.1f} s"
            )
            return (wave_surface[0],)

        animation = FuncAnimation(
            fig,
            update,
            frames=len(times_s),
            interval=1000 / max(1, args.fps),
            blit=False,
        )
        try:
            animation.save(gif_path, writer=PillowWriter(fps=max(1, args.fps)))
            print(f"Animation saved: {gif_path}")
        except (RuntimeError, ImportError) as exc:
            print(f"GIF export failed ({exc}). Static figures and snapshots were still saved.")
            print("To enable GIF export, install Pillow: python -m pip install pillow")
        plt.close(fig)

    print("\nVirtual lab outputs:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")
    print(f"\nSaved {len(times_s)} actual wave snapshots on {len(x_m)} spatial grid points.")
    print("Note: the 3D view is a visualization of a 1D simulation, not a 3D fluid solver.")


if __name__ == "__main__":
    main()
