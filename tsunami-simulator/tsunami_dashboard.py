"""Smooth, client-side playback dashboard for the 1D tsunami model.

Place this file beside tsunami_virtual_lab.py and launch it with:
    python -m streamlit run tsunami_dashboard_smooth.py

The solver remains 1D. The 3D-style scene extrudes the wave profile for display only.
Playback interpolates between saved numerical snapshots in the browser, avoiding a
full Streamlit rerun for every animation frame.
"""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import tsunami_virtual_lab as model


st.set_page_config(page_title="Virtual Tsunami Lab", page_icon="🌊", layout="wide")
st.title("🌊 Virtual Tsunami Lab")
st.caption(
    "Interactive visualization of a 1D nonlinear shallow-water simulation. "
    "The 3D scene is a display extrusion, not a 3D fluid solver. Playback uses "
    "smooth client-side interpolation between saved model snapshots."
)

with st.sidebar:
    st.header("Simulation parameters")
    with st.form("simulation_parameters"):
        deep_depth = st.slider("Deep-water depth (m)", 190.0, 210.0, 200.0, 1.0)
        shallow_depth = st.slider("Shallow-water depth (m)", 45.0, 55.0, 50.0, 0.5)
        slope_end_km = st.slider("Slope-end position (km)", 75.0, 85.0, 80.0, 0.5)
        source_amplitude = st.slider("Initial wave amplitude (m)", 0.25, 3.0, 1.0, 0.25)
        total_time = st.slider("Simulation duration (s)", 1200, 2400, 2200, 100)
        snapshot_count = st.select_slider("Saved snapshots", options=[41, 61, 91], value=91)
        vertical_exaggeration = st.slider(
            "Wave-height visual exaggeration", min_value=5, max_value=50, value=20, step=5
        )
        run_clicked = st.form_submit_button("▶ Run simulation", type="primary")
    st.caption("Parameter ranges are synthetic development assumptions.")


@st.cache_data(show_spinner=False)
def run_cached(
    deep: float,
    shallow: float,
    slope_end_km_arg: float,
    amplitude: float,
    duration: float,
    snapshots: int,
):
    model.wave_amplitude = amplitude
    return model.run_visualization_simulation(
        deep_depth=deep,
        shallow_depth=shallow,
        slope_end=slope_end_km_arg * 1000.0,
        total_time=duration,
        number_of_snapshots=snapshots,
    )


if "simulation" not in st.session_state:
    st.session_state["simulation"] = None

if run_clicked:
    if deep_depth <= shallow_depth:
        st.error("Deep-water depth must be greater than shallow-water depth.")
        st.stop()
    with st.spinner("Solving the shallow-water equations and saving wave states..."):
        try:
            result = run_cached(
                deep_depth, shallow_depth, slope_end_km, source_amplitude, total_time, snapshot_count
            )
            st.session_state["simulation"] = {
                "x_m": result[0],
                "depth_m": result[1],
                "times_s": result[2],
                "eta_m": result[3],
                "deep_depth_m": deep_depth,
                "shallow_depth_m": shallow_depth,
                "slope_end_km": slope_end_km,
                "source_amplitude_m": source_amplitude,
                "total_time_s": total_time,
                "frames": snapshot_count,
                "vertical_exaggeration": vertical_exaggeration,
            }
        except Exception as exc:
            st.exception(exc)
            st.stop()

sim = st.session_state["simulation"]
if sim is None:
    st.info("Choose the parameters in the sidebar and click **Run simulation** to create wave snapshots.")
    st.stop()

x_m = np.asarray(sim["x_m"])
depth_m = np.asarray(sim["depth_m"])
times_s = np.asarray(sim["times_s"])
eta_m = np.asarray(sim["eta_m"])
x_km_full = x_m / 1000.0
z_exag = int(sim["vertical_exaggeration"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Deep depth", f'{sim["deep_depth_m"]:.1f} m')
c2.metric("Shallow depth", f'{sim["shallow_depth_m"]:.1f} m')
c3.metric("Slope end", f'{sim["slope_end_km"]:.1f} km')
c4.metric("Saved wave states", f"{len(times_s)}")

gauge_indices = {
    40: int(np.argmin(np.abs(x_km_full - 40.0))),
    80: int(np.argmin(np.abs(x_km_full - 80.0))),
}
peak_40 = float(np.max(eta_m[:, gauge_indices[40]]))
peak_80 = float(np.max(eta_m[:, gauge_indices[80]]))
amp_ratio = peak_80 / peak_40 if peak_40 > 1e-12 else float("nan")
st.caption(
    f"Snapshot-based peak estimates (approximate): 40 km = {peak_40:.4f} m; "
    f"80 km = {peak_80:.4f} m; ratio = {amp_ratio:.3f}×. These displayed-snapshot "
    "diagnostics do not replace the higher-resolution gauge measurements used in the UQ experiments."
)

st.subheader("Smooth wave playback")
st.write(
    "Use the in-panel Play/Pause control and scrubber. Playback runs in your browser, "
    "interpolates between stored snapshots, and does not rerun Streamlit on every frame."
)

# Downsample the displayed spatial grid only; the downloaded simulation arrays stay untouched.
stride = max(1, len(x_km_full) // 320)
x_km = x_km_full[::stride]
depth_view = depth_m[::stride]
eta_view = eta_m[:, ::stride]
eta_display = eta_view * z_exag
n_frames, n_points = eta_view.shape

# Keep the bathymetry and surface arrays small enough for responsive client-side WebGL redraws.
X = np.broadcast_to(x_km, (2, len(x_km)))
Y = np.broadcast_to(np.array([[0.0], [1.0]]), (2, len(x_km)))
bed_z = np.broadcast_to(-depth_view, (2, len(x_km)))
wave_z0 = np.broadcast_to(eta_display[0], (2, len(x_km)))
source_idx = int(np.argmin(np.abs(x_km - 20.0)))

# Build every visualization as its own full-width Plotly figure. Each card has
# an independent fullscreen button; the wave/profile/time cursor remain linked
# by the browser-side player without rerunning the Streamlit app.

fig3d = go.Figure()
fig3d.add_trace(
    go.Surface(
        x=X, y=Y, z=bed_z, name="Seabed", opacity=0.72, showscale=False,
        hovertemplate="Distance: %{x:.1f} km<br>Bed elevation: %{z:.1f} m<extra>Seabed</extra>",
    )
)
fig3d.add_trace(
    go.Surface(
        x=X, y=Y, z=wave_z0, name="Wave surface (display scale)", opacity=0.95,
        showscale=False,
        hovertemplate=f"Distance: %{{x:.1f}} km<br>η × {z_exag} = %{{z:.2f}} m<extra>Display scale</extra>",
    )
)
for gauge_km in (40.0, 80.0):
    idx = int(np.argmin(np.abs(x_km_full - gauge_km)))
    fig3d.add_trace(
        go.Scatter3d(
            x=[gauge_km, gauge_km], y=[0.5, 0.5], z=[-float(depth_m[idx]), 0.0],
            mode="lines+text", text=[f"{gauge_km:.0f} km gauge", ""],
            textposition="top center", name=f"{gauge_km:.0f} km gauge",
            line={"dash": "dash", "width": 4},
        )
    )
fig3d.add_trace(
    go.Scatter3d(
        x=[20.0], y=[0.5], z=[float(eta_display[0, source_idx])],
        mode="markers+text", text=["Source"], textposition="top center",
        marker={"symbol": "diamond", "size": 6}, name="Initial source location",
    )
)
fig3d.update_layout(
    title="Interactive 3D-style wave surface (illustrative extrusion)",
    height=700, autosize=True,
    margin={"l": 10, "r": 10, "t": 65, "b": 10},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    scene={
        "xaxis_title": "Distance (km)",
        "yaxis_title": "Display width (illustration only)",
        "zaxis_title": f"Elevation (m; η displayed ×{z_exag})",
        "xaxis": {"range": [float(np.min(x_km)), float(np.max(x_km))]},
        "yaxis": {"range": [-0.1, 1.1]},
        "zaxis": {"range": [float(-np.max(depth_view) - 15), float(max(25.0, np.max(np.abs(eta_display)) + 5))]},
        "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 0.8}},
        "aspectratio": {"x": 2.5, "y": 0.3, "z": 1.0},
    },
)

# A dedicated bathymetry plot makes depth transitions easier to inspect.
fig_bathy = go.Figure()
fig_bathy.add_trace(
    go.Scatter(
        x=x_km, y=-depth_view, mode="lines", fill="tozeroy",
        name="Seabed elevation", line={"width": 2.5},
    )
)
fig_bathy.update_layout(
    title="Seafloor bathymetry profile", height=520, autosize=True,
    margin={"l": 55, "r": 25, "t": 65, "b": 55},
    xaxis_title="Distance along model (km)", yaxis_title="Seabed elevation (m)",
)

# The wave profile has its own scale because η is much smaller than ocean depth.
profile_pad = max(0.1, float(np.max(np.abs(eta_display))) * 0.15)
fig_profile = go.Figure()
fig_profile.add_trace(
    go.Scatter(
        x=x_km, y=eta_display[0], mode="lines", name=f"Surface anomaly ×{z_exag}",
        line={"width": 2.5},
        hovertemplate="Distance: %{x:.2f} km<br>Displayed η: %{y:.3f} m<extra></extra>",
    )
)
fig_profile.update_layout(
    title="Current wave profile", height=520, autosize=True,
    margin={"l": 55, "r": 25, "t": 65, "b": 55},
    xaxis_title="Distance (km)",
    yaxis_title=f"Displayed free-surface anomaly (η × {z_exag}) in m",
    yaxis={"range": [float(np.min(eta_display) - profile_pad), float(np.max(eta_display) + profile_pad)]},
)

fig_gauge = go.Figure()
for gauge_km, idx in gauge_indices.items():
    fig_gauge.add_trace(
        go.Scatter(
            x=times_s, y=eta_m[:, idx], mode="lines", name=f"{gauge_km} km gauge",
            line={"width": 2.2},
        )
    )
eta_min = float(np.min(eta_m))
eta_max = float(np.max(eta_m))
if eta_max <= eta_min:
    eta_max = eta_min + 1.0
fig_gauge.add_trace(
    go.Scatter(
        x=[times_s[0], times_s[0]], y=[eta_min, eta_max], mode="lines",
        name="Current simulation time", line={"dash": "dash", "width": 2},
    )
)
fig_gauge.update_layout(
    title="Gauge time series", height=520, autosize=True,
    margin={"l": 55, "r": 25, "t": 65, "b": 55},
    xaxis_title="Simulation time (s)", yaxis_title="Free-surface elevation η (m)",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        x=x_km, y=times_s, z=eta_view, colorscale="RdBu", zmid=0.0,
        colorbar={"title": "η (m)"},
        hovertemplate="Distance: %{x:.1f} km<br>Time: %{y:.1f} s<br>η: %{z:.4f} m<extra></extra>",
    )
)
fig_heatmap.update_layout(
    title="Space-time diagram — actual saved model snapshots",
    height=620, autosize=True,
    margin={"l": 55, "r": 25, "t": 65, "b": 55},
    xaxis_title="Distance (km)", yaxis_title="Simulation time (s)",
)

config = {"responsive": True, "displaylogo": False, "scrollZoom": True}
plot3d_html = pio.to_html(fig3d, full_html=False, include_plotlyjs="cdn", div_id="tsunami3d", config=config)
bathy_html = pio.to_html(fig_bathy, full_html=False, include_plotlyjs=False, div_id="tsunamiBathymetry", config=config)
profile_html = pio.to_html(fig_profile, full_html=False, include_plotlyjs=False, div_id="tsunamiProfile", config=config)
gauge_html = pio.to_html(fig_gauge, full_html=False, include_plotlyjs=False, div_id="tsunamiGauge", config=config)
heatmap_html = pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False, div_id="tsunamiHeatmap", config=config)

# Only the spatially downsampled arrays are sent to the browser for playback.
payload = {
    "times": times_s.astype(float).tolist(),
    "eta_display": eta_display.astype(float).tolist(),
    "source_idx": source_idx,
}
payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:#172033; background:#f1f5f9; }}
  .player {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:12px 14px; margin:0 0 14px 0;
             border:1px solid #d9dee8; border-radius:12px; background:#fff; position:sticky; top:0; z-index:20; box-shadow:0 2px 8px #0f172a10; }}
  .player button, .fullscreen-btn {{ border:0; border-radius:8px; padding:9px 15px; background:#1f2937; color:white; font-size:14px; cursor:pointer; }}
  .player button:hover, .fullscreen-btn:hover {{ opacity:0.88; }}
  .player label {{ font-size:13px; color:#475569; }}
  #timeSlider {{ flex:1 1 220px; min-width:150px; accent-color:#2563eb; }}
  #speed {{ border:1px solid #cbd5e1; border-radius:7px; padding:7px; background:white; }}
  #timeLabel {{ min-width:125px; font-variant-numeric:tabular-nums; font-weight:600; }}
  .tip {{ font-size:12px; color:#64748b; padding:0 3px 10px; line-height:1.5; }}
  .plot-card {{ background:white; border:1px solid #d9dee8; border-radius:14px; padding:10px 12px 4px; margin:0 0 18px 0; box-shadow:0 3px 12px #0f172a0a; overflow:hidden; }}
  .plot-header {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:4px 8px 0; }}
  .plot-header h2 {{ margin:5px 0; font-size:20px; font-weight:650; }}
  .plot-header p {{ margin:3px 0 8px; font-size:12px; color:#64748b; }}
  .fullscreen-btn {{ background:#334155; white-space:nowrap; }}
  .js-plotly-plot {{ width:100% !important; }}
  .plot-card:fullscreen, .plot-card.focus-mode {{ background:#fff; width:100vw; height:100vh; overflow:auto; padding:12px 18px; border:0; border-radius:0; }}
  .plot-card:fullscreen .js-plotly-plot, .plot-card.focus-mode .js-plotly-plot {{ height:calc(100vh - 95px) !important; }}
  .plot-card:fullscreen .plot-header h2, .plot-card.focus-mode .plot-header h2 {{ font-size:24px; }}
  .research-note {{ font-size:12px; color:#64748b; background:white; padding:12px; border-radius:10px; border:1px solid #d9dee8; line-height:1.6; }}
  @media (max-width: 650px) {{ .plot-header {{ align-items:flex-start; }} .plot-header h2 {{ font-size:16px; }} .fullscreen-btn {{ padding:8px 10px; }} }}
</style>
</head>
<body>
<div class="player">
  <button id="playButton" type="button">▶ Play</button>
  <button id="restartButton" type="button">↺ Restart</button>
  <label for="timeSlider">Simulation time</label>
  <input id="timeSlider" type="range" min="0" max="{n_frames - 1}" step="0.01" value="0">
  <span id="timeLabel">t = {times_s[0]:.1f} s</span>
  <label for="speed">Seconds / saved interval</label>
  <select id="speed">
    <option value="0.15">Very fast</option>
    <option value="0.25">Fast</option>
    <option value="0.35" selected>Normal</option>
    <option value="0.6">Slow</option>
    <option value="1.0">Very slow</option>
  </select>
</div>
<div class="tip">Playback interpolates between saved numerical snapshots to make movement smoother; intermediate display frames are visualization values, not additional solver time steps. Each chart has its own Fullscreen button and Plotly toolbar.</div>

<section class="plot-card" id="card-3d">
  <div class="plot-header"><div><h2>1. Interactive 3D-style wave view</h2><p>Rotate and zoom to inspect the surface and bathymetry.</p></div><button class="fullscreen-btn" data-target="card-3d">⛶ Fullscreen</button></div>
  {plot3d_html}
</section>
<section class="plot-card" id="card-bathymetry">
  <div class="plot-header"><div><h2>2. Seafloor bathymetry</h2><p>Ocean-floor depth profile along the 1D model.</p></div><button class="fullscreen-btn" data-target="card-bathymetry">⛶ Fullscreen</button></div>
  {bathy_html}
</section>
<section class="plot-card" id="card-profile">
  <div class="plot-header"><div><h2>3. Current wave profile</h2><p>Updates with playback. Wave elevation is visually exaggerated by ×{z_exag} for readability.</p></div><button class="fullscreen-btn" data-target="card-profile">⛶ Fullscreen</button></div>
  {profile_html}
</section>
<section class="plot-card" id="card-gauge">
  <div class="plot-header"><div><h2>4. Gauge time series</h2><p>Compare free-surface elevation at 40 km and 80 km; the dashed line follows playback.</p></div><button class="fullscreen-btn" data-target="card-gauge">⛶ Fullscreen</button></div>
  {gauge_html}
</section>
<section class="plot-card" id="card-heatmap">
  <div class="plot-header"><div><h2>5. Space-time wave diagram</h2><p>Shows the actual saved model states across distance and time.</p></div><button class="fullscreen-btn" data-target="card-heatmap">⛶ Fullscreen</button></div>
  {heatmap_html}
</section>
<div class="research-note">Research note: the vertical exaggeration affects visualization only. This remains a 1D nonlinear shallow-water model, not a 3D fluid solver or an operational tsunami forecast. Parameter ranges are synthetic development assumptions.</div>

<script>
(() => {{
  const state = {payload_json};
  const n = state.times.length;
  const plot3d = document.getElementById('tsunami3d');
  const profile = document.getElementById('tsunamiProfile');
  const gauge = document.getElementById('tsunamiGauge');
  const playButton = document.getElementById('playButton');
  const restartButton = document.getElementById('restartButton');
  const slider = document.getElementById('timeSlider');
  const timeLabel = document.getElementById('timeLabel');
  const speed = document.getElementById('speed');
  let position = 0;
  let playing = false;
  let lastTick = null;
  let lastRenderTick = 0;
  let renderBusy = false;
  let queuedPosition = null;

  function resizeCharts() {{
    document.querySelectorAll('.js-plotly-plot').forEach((chart) => {{
      if (window.Plotly && chart && chart.data) Plotly.Plots.resize(chart);
    }});
  }}

  document.querySelectorAll('.fullscreen-btn').forEach((button) => {{
    button.addEventListener('click', async () => {{
      const card = document.getElementById(button.dataset.target);
      try {{
        if (document.fullscreenElement === card) {{
          await document.exitFullscreen();
        }} else if (document.fullscreenElement) {{
          await document.exitFullscreen();
          await card.requestFullscreen();
        }} else {{
          await card.requestFullscreen();
        }}
      }} catch (error) {{
        // If browser fullscreen is blocked by the host, provide an in-page focus view.
        card.classList.toggle('focus-mode');
        button.textContent = card.classList.contains('focus-mode') ? '✕ Exit fullscreen view' : '⛶ Fullscreen';
      }}
      window.setTimeout(resizeCharts, 120);
    }});
  }});
  document.addEventListener('fullscreenchange', () => {{
    document.querySelectorAll('.fullscreen-btn').forEach((button) => {{
      const card = document.getElementById(button.dataset.target);
      button.textContent = document.fullscreenElement === card ? '✕ Exit fullscreen' : '⛶ Fullscreen';
    }});
    window.setTimeout(resizeCharts, 120);
  }});

  function setPlaying(value) {{
    playing = value;
    playButton.textContent = playing ? '⏸ Pause' : '▶ Play';
    lastTick = null;
  }}

  function renderAt(pos) {{
    position = Math.max(0, Math.min(n - 1, Number(pos)));
    queuedPosition = position;
    if (renderBusy) return;
    renderBusy = true;
    (async () => {{
      while (queuedPosition !== null) {{
        const p = queuedPosition;
        queuedPosition = null;
        const lo = Math.floor(p);
        const hi = Math.min(n - 1, lo + 1);
        const alpha = p - lo;
        const loWave = state.eta_display[lo];
        const hiWave = state.eta_display[hi];
        const wave = new Array(loWave.length);
        for (let k = 0; k < wave.length; k++) {{
          wave[k] = loWave[k] + (hiWave[k] - loWave[k]) * alpha;
        }}
        const currentTime = state.times[lo] + (state.times[hi] - state.times[lo]) * alpha;
        const updates = [
          Plotly.restyle(plot3d, {{z: [[wave, wave]]}}, [1]),
          Plotly.restyle(plot3d, {{z: [[wave[state.source_idx]]]}}, [4]),
          Plotly.restyle(profile, {{y: [wave]}}, [0]),
          Plotly.restyle(gauge, {{x: [[currentTime, currentTime]]}}, [2]),
        ];
        await Promise.all(updates);
        slider.value = String(p);
        timeLabel.textContent = 't = ' + currentTime.toFixed(1) + ' s';
      }}
      renderBusy = false;
    }})().catch(err => {{
      renderBusy = false;
      timeLabel.textContent = 'Playback rendering error';
      console.error(err);
    }});
  }}

  playButton.addEventListener('click', () => {{
    if (playing) {{
      setPlaying(false);
    }} else {{
      if (position >= n - 1) renderAt(0);
      setPlaying(true);
    }}
  }});
  restartButton.addEventListener('click', () => {{
    setPlaying(false);
    renderAt(0);
  }});
  slider.addEventListener('input', () => {{
    setPlaying(false);
    renderAt(Number(slider.value));
  }});

  function tick(timestamp) {{
    window.requestAnimationFrame(tick);
    if (!playing) {{ lastTick = timestamp; return; }}
    if (lastTick === null) {{ lastTick = timestamp; return; }}
    const elapsed = timestamp - lastTick;
    lastTick = timestamp;
    position += elapsed / (Number(speed.value) * 1000);
    if (position >= n - 1) {{
      position = n - 1;
      setPlaying(false);
    }}
    if (timestamp - lastRenderTick >= 40 || !playing) {{
      lastRenderTick = timestamp;
      renderAt(position);
    }}
  }}

  function waitForPlots() {{
    const allReady = plot3d && profile && gauge && plot3d.data && profile.data && gauge.data && window.Plotly;
    if (!allReady) {{
      window.setTimeout(waitForPlots, 50);
      return;
    }}
    renderAt(0);
    window.requestAnimationFrame(tick);
  }}
  waitForPlots();
}})();
</script>
</body>
</html>
"""

# Taller scrollable component so all standalone plots are accessible in sequence.
st.components.v1.html(html, height=3100, scrolling=True)
st.caption(
    "The plot cards are separate full-width figures with fullscreen controls. Wave-height exaggeration affects "
    "visualization only; intermediate animation frames are interpolated from saved numerical snapshots."
)
