"""
Gaussian Plume App.

Capabilities:
- Interactive visualization of Gaussian plume concentration maps at ground level or user-defined receptor heights.
- Time-evolution advection animation with optional MP4 export of rendered frames.
- Point-concentration solver with support for Pasquill–Gifford curves or user-supplied σ_y/σ_z values.
- Cross-sectional views (X vs Z and Y vs Z) and summary metrics for quick assessment of peak ground impacts.

Author: Subodh Purohit
Motivation: Dr. Abhradeep Majumder, Dr. Krishna C. Etika
Purpose: Educational use only.
"""

# -------------------------
# Imports
# -------------------------
import io
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

APP_VERSION = "2026-05-03"

try:
    import imageio
except ImportError:
    imageio = None

# -------------------------
# Configuration / Constants
# -------------------------
SIGMA_COEFFS = {
    'y': {
        'A': [0.22, 0.16, 0.11, 0.08, 0.06, 0.04],
        'B': [0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
    },
    'z': {
        'A': [0.20, 0.12, 0.08, 0.06, 0.03, 0.016],
        'B': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    },
    'index': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
}

WIND_PROFILE_EXPONENTS = {
    'Rough Surface (urban)': {'A': 0.15, 'B': 0.15, 'C': 0.20, 'D': 0.25, 'E': 0.30, 'F': 0.30},
    'Smooth Surface (rural)': {'A': 0.07, 'B': 0.07, 'C': 0.10, 'D': 0.15, 'E': 0.35, 'F': 0.35},
}

# -------------------------
# MP4 Export Helpers (kept for other places that may use it)
# -------------------------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def export_animation_to_mp4(frames_rgb, fps: int, out_path: str):
    """
    Write frames (H,W,3) uint8 RGB numpy arrays to an MP4 using imageio.
    Returns the path to the written file.
    """
    if imageio is None:
        raise RuntimeError("MP4 export requires the optional 'imageio' package. Install it to enable animation export.")

    _ensure_dir(out_path)
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec='libx264', format='ffmpeg', macro_block_size=None)
    except Exception:
        writer = imageio.get_writer(out_path, fps=fps, format='ffmpeg')

    for im in frames_rgb:
        writer.append_data(im)
    writer.close()
    return out_path

def _render_frame_with_top_info(array2d, t_seconds: float, vmin=None, vmax=None,
                                x_extent: Tuple[float, float] = None, y_extent: Tuple[float, float] = None,
                                figsize=(10, 4)):
    """
    Render a 2D array with matplotlib including axes, colorbar and a top info line.
    Returns an RGB uint8 numpy array (H, W, 3).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if x_extent is None:
        x_extent = (0.0, array2d.shape[1])
    if y_extent is None:
        y_extent = (0.0, array2d.shape[0])

    im = ax.imshow(array2d, origin='lower', aspect='auto',
                   extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('Downwind distance x (m)')
    ax.set_ylabel('Crosswind distance y (m)')
    ax.set_title('Concentration (µg m⁻³)')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Concentration (µg m⁻³)')

    max_val = np.nanmax(array2d) if array2d.size else 0.0
    info_text = f"t = {t_seconds:.1f} s   |   conc. = {max_val:,.2f} µg/m³"
    fig.text(0.5, 0.96, info_text, ha='center', va='center', fontsize=11, weight='bold')

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8)
    arr = arr.reshape((h, w, 4))
    rgb = arr[..., :3]
    plt.close(fig)
    return rgb

# -------------------------
# Model Functions
# -------------------------
@st.cache_data
def get_dispersion_coefficients(x, stability_class):
    """Calculates sigma_y and sigma_z based on downwind distance and stability class."""
    try:
        idx = SIGMA_COEFFS['index'][stability_class]
    except KeyError:
        return 0, 0

    Ay = SIGMA_COEFFS['y']['A'][idx]
    By = SIGMA_COEFFS['y']['B'][idx]
    Az = SIGMA_COEFFS['z']['A'][idx]
    Bz = SIGMA_COEFFS['z']['B'][idx]

    x_arr = np.asarray(x, dtype=float)
    x_for_power = np.where(x_arr <= 1e-6, 1e-6, x_arr)

    sigma_y = Ay * (x_for_power ** By)
    sigma_z = Az * (x_for_power ** Bz)

    if stability_class == 'F':
        sigma_z = np.where(np.asarray(x) < 100, np.maximum(sigma_z, 1.0), sigma_z)

    sigma_y = np.maximum(sigma_y, 1e-3)
    sigma_z = np.maximum(sigma_z, 1e-3)

    return sigma_y, sigma_z

@st.cache_data
def gaussian_plume_model(x_m, y_m, z, H, Q, U, stability_class):
    """
    Vectorized Gaussian plume concentration for arrays X,Y at a fixed height z.
    Returns concentration in g/m^3.
    """
    X = np.asarray(x_m, dtype=float)
    Y = np.asarray(y_m, dtype=float)

    C = np.zeros_like(X, dtype=float)
    positive_mask = X > 0.0
    if not np.any(positive_mask):
        return C

    sigma_y, sigma_z = get_dispersion_coefficients(X, stability_class)
    denom = 2 * np.pi * U * sigma_y * sigma_z

    exp_y = np.exp(-Y ** 2 / (2 * sigma_y ** 2))
    exp_z_real = np.exp(-(z - H) ** 2 / (2 * sigma_z ** 2))
    exp_z_image = np.exp(-(z + H) ** 2 / (2 * sigma_z ** 2))
    vertical_term = exp_z_real + exp_z_image

    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(positive_mask, Q / denom, 0.0)
        C = scaling_factor * exp_y * vertical_term

    C = np.where(positive_mask, C, 0.0)
    return C

# Compatibility wrapper - kept for backward compatibility with earlier code references
def gaussian_plume_point_source(Q_g_s, H_m, U_m_s, x, y, z, stability_class):
    """
    Wrapper matching older call signatures: forwards to gaussian_plume_model.
    """
    return gaussian_plume_model(x, y, z, H_m, Q_g_s, U_m_s, stability_class)

def find_max_concentration(H, Q, U, stability_class):
    """Finds maximum ground concentration (z=0) on the centerline y=0 over a search range."""
    x_range = np.linspace(10, 5000, 500)
    max_C = 0.0
    max_x = 0.0
    z = 0.0

    for x in x_range:
        sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)
        if sigma_y == 0 or sigma_z == 0:
            continue
        vertical_term = 2.0 * np.exp(-H ** 2 / (2 * sigma_z ** 2))
        scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
        C_centerline = scaling_factor * vertical_term
        if C_centerline > max_C:
            max_C = C_centerline
            max_x = x
    return max_C, max_x

def calculate_single_point_concentration(x, y, z, H, Q, U, stability_class):
    """Concentration at a single point (scalars)."""
    if x <= 0:
        return 0.0

    sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)
    if sigma_y == 0 or sigma_z == 0:
        return 0.0

    exp_y = np.exp(-y ** 2 / (2 * sigma_y ** 2))
    exp_z_real = np.exp(-(z - H) ** 2 / (2 * sigma_z ** 2))
    exp_z_image = np.exp(-(z + H) ** 2 / (2 * sigma_z ** 2))
    vertical_term = exp_z_real + exp_z_image
    scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
    C = scaling_factor * exp_y * vertical_term
    return C

# Custom-sigma helper for solver
def calculate_point_concentration_custom_sigma(x, y, z, H, Q, U, sigma_y, sigma_z):
    if x <= 0 or sigma_y <= 0 or sigma_z <= 0:
        return 0.0
    exp_y = np.exp(-y ** 2 / (2 * sigma_y ** 2))
    exp_z_real = np.exp(-(z - H) ** 2 / (2 * sigma_z ** 2))
    exp_z_image = np.exp(-(z + H) ** 2 / (2 * sigma_z ** 2))
    vertical_term = exp_z_real + exp_z_image
    scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
    C = scaling_factor * exp_y * vertical_term
    return C

def find_max_concentration_custom_sigma_fixed_ratio(H, Q, U, stability_class):
    # Behavior preserved: fallback to find_max_concentration
    return find_max_concentration(H, Q, U, stability_class)

def calculate_wind_speed_at_height(u_ref, z_ref, z_target, stability_class, surface_type):
    """Power-law wind profile from slide 38: u/u1 = (z/z1)^p."""
    p = WIND_PROFILE_EXPONENTS[surface_type][stability_class]
    z_ref_safe = max(float(z_ref), 1e-6)
    z_target_safe = max(float(z_target), 1e-6)
    u_at_target = float(u_ref) * (z_target_safe / z_ref_safe) ** p
    return u_at_target, p

def correct_wind_to_surface(u_H: float, H: float, p: float = 0.25) -> float:
    """
    Convert wind speed measured at stack height H to the 10 m surface reference
    using the power-law profile:  u_0 = u_H * (10 / H)^p

    Parameters
    ----------
    u_H : wind speed at stack altitude H (m/s)
    H   : stack height above ground (m)
    p   : wind shear exponent (default 0.25, typical neutral/standard)

    Returns
    -------
    u_0 : equivalent 10 m surface wind speed (m/s)
    """
    if H <= 0:
        return float(u_H)
    return float(u_H) * (10.0 / float(H)) ** float(p)

# -------------------------
# Pasquill–Gifford Stability Table
# -------------------------
# Source: Pasquill (1961) / Gifford (1961), as tabulated in standard references.
# Rows = surface wind speed bins; columns = daytime solar radiation or
# night-time cloud cover.  Mixed entries (e.g. "A-B") are resolved to the more
# stable single class for computation (conservative choice).

_PG_DAY = {
    # (wind_bin, solar_radiation) -> (table_label, resolved_class)
    #   solar_radiation: 'strong' | 'moderate' | 'slight'
    (0, 'strong'):   ('A',   'A'),
    (0, 'moderate'): ('A-B', 'B'),
    (0, 'slight'):   ('B',   'B'),
    (1, 'strong'):   ('A-B', 'B'),
    (1, 'moderate'): ('B',   'B'),
    (1, 'slight'):   ('C',   'C'),
    (2, 'strong'):   ('B',   'B'),
    (2, 'moderate'): ('B-C', 'C'),
    (2, 'slight'):   ('C',   'C'),
    (3, 'strong'):   ('C',   'C'),
    (3, 'moderate'): ('C-D', 'D'),
    (3, 'slight'):   ('D',   'D'),
    (4, 'strong'):   ('C',   'C'),
    (4, 'moderate'): ('D',   'D'),
    (4, 'slight'):   ('D',   'D'),
}

_PG_NIGHT = {
    # (wind_bin, cloudiness) -> (table_label, resolved_class)
    #   cloudiness: 'cloudy' (≥4/8) | 'clear' (<3/8)
    (0, 'cloudy'): ('E', 'E'),
    (0, 'clear'):  ('F', 'F'),
    (1, 'cloudy'): ('E', 'E'),
    (1, 'clear'):  ('F', 'F'),
    (2, 'cloudy'): ('D', 'D'),
    (2, 'clear'):  ('E', 'E'),
    (3, 'cloudy'): ('D', 'D'),
    (3, 'clear'):  ('D', 'D'),
    (4, 'cloudy'): ('D', 'D'),
    (4, 'clear'):  ('D', 'D'),
}

def _wind_bin(u_surface: float) -> int:
    """Map surface wind speed (m/s) to PG table row index (0–4)."""
    if u_surface < 2:   return 0
    if u_surface < 3:   return 1
    if u_surface < 5:   return 2
    if u_surface <= 6:  return 3
    return 4

def determine_stability_class(u_surface: float, time_of_day: str, condition: str):
    """
    Look up Pasquill-Gifford stability class from surface wind speed and
    atmospheric conditions.

    Parameters
    ----------
    u_surface   : surface (10 m) wind speed (m/s)
    time_of_day : 'day' or 'night'
    condition   : for day   → 'strong' | 'moderate' | 'slight'
                  for night → 'cloudy' | 'clear'

    Returns
    -------
    (table_label, resolved_class)
        table_label    : raw entry, e.g. "A-B"
        resolved_class : single letter used for σ calculations, e.g. "B"
    """
    wb = _wind_bin(float(u_surface))
    if time_of_day == 'day':
        return _PG_DAY.get((wb, condition), ('D', 'D'))
    return _PG_NIGHT.get((wb, condition), ('D', 'D'))

# -------------------------
# UI Building Functions
# -------------------------
def _configure_page():
    st.set_page_config(layout="wide", page_title="Gaussian Plume Visualizer")

def _sidebar_inputs():
    st.sidebar.header("Source & Environment Parameters")
    st.sidebar.subheader("1. Source Strength")
    st.sidebar.markdown(r"Emission Rate ($Q$, $\text{g/s}$)")
    Q_g_s = st.sidebar.slider('Hidden label', 10.0, 500.0, 100.0, step=10.0, key='Q_slider', label_visibility='collapsed')
    st.sidebar.markdown(r"Effective Stack Height ($H$, $\text{m}$)")
    H_m = st.sidebar.slider('Hidden label', 10.0, 200.0, 100.0, step=5.0, key='H_slider', label_visibility='collapsed')

    st.sidebar.subheader("2. Atmospheric Conditions")
    st.sidebar.markdown(r"Wind Speed ($U$, $\text{m/s}$)")
    U_m_s_input = st.sidebar.slider('Hidden label', 1.0, 20.0, 5.0, step=0.5, key='U_slider', label_visibility='collapsed')

    stability_options = {'A': 'A - Extremely Unstable', 'B': 'B - Moderately Unstable', 'C': 'C - Slightly Unstable',
                         'D': 'D - Neutral (Overcast/High Wind)', 'E': 'E - Slightly Stable', 'F': 'F - Moderately Stable'}
    stability_class = st.sidebar.selectbox("Atmospheric Stability Class", options=list(stability_options.keys()),
                                         format_func=lambda x: stability_options[x], index=3, key='stability_slider')

    use_wind_profile = st.sidebar.checkbox(
        "Estimate stack-height wind speed using power law",
        value=False,
        help="Use slide 38: u/u1 = (z/z1)^p. When enabled, the wind speed above is treated as the reference wind speed."
    )

    if use_wind_profile:
        wind_surface = st.sidebar.selectbox(
            "Surface roughness",
            options=list(WIND_PROFILE_EXPONENTS.keys()),
            index=0,
            key="wind_profile_surface"
        )
        wind_ref_height = st.sidebar.number_input(
            "Reference wind height z1 (m)",
            min_value=0.1,
            value=10.0,
            step=1.0,
            key="wind_profile_ref_height"
        )
        U_m_s, wind_exponent = calculate_wind_speed_at_height(
            U_m_s_input, wind_ref_height, H_m, stability_class, wind_surface
        )
        st.sidebar.info(
            f"Using stack-height wind speed: U({H_m:.0f} m) = {U_m_s:.2f} m/s "
            f"from U({wind_ref_height:.0f} m) = {U_m_s_input:.2f} m/s, p = {wind_exponent:.2f}."
        )
    else:
        U_m_s = U_m_s_input
        st.sidebar.caption("Using entered wind speed directly as the model wind speed.")

    st.sidebar.subheader("3. Visualization Plane (z)")
    use_custom_z = st.sidebar.checkbox(r"Visualize at a Custom Height ($z$)", value=False,
                                       help="Toggle to view concentration on a horizontal plane above the ground.")

    if use_custom_z:
        max_z_limit = max(150.0, H_m * 1.5)
        Z_RECEPTOR_M = st.sidebar.slider(r"Receptor Plane Height ($z$, $\text{m}$)", 0.0, max_z_limit, 50.0, step=5.0)
    else:
        Z_RECEPTOR_M = 0.0

    st.sidebar.markdown(
        """
        <small>
        The Stability Class determines the rate of atmospheric mixing.
        </small>
        """, unsafe_allow_html=True
    )

    st.sidebar.markdown("---")

    # Sidebar footer (fixed)
    st.sidebar.markdown(
        "<p style='font-size: 11px; color: grey;'>Developed by: <b>Subodh Purohit</b><br><i>For educational purposes only.</i></p>",
        unsafe_allow_html=True
    )

    return Q_g_s, H_m, U_m_s, stability_class, Z_RECEPTOR_M, stability_options

def _build_visualizer_tab(Q_g_s, H_m, U_m_s, stability_class, Z_RECEPTOR_M, stability_options):
    st.title("Gaussian Plume Dispersion Visualizer")
    st.caption(f"App version: {APP_VERSION}")
    st.markdown("An interactive model to explore how source parameters and atmospheric stability affect pollutant spread and ground-level concentration.")

    # Tab order: Plume Visualizer, 3D Visualization, Problem Solver, Theory & Assumptions
    tab1, tab2, tab3, tab4 = st.tabs(["Plume Visualizer", "3D Visualization", "Problem Solver", "Theory & Assumptions"])

    # --- Tab 1: Plume Visualizer ---
    with tab1:
        if Z_RECEPTOR_M > 0:
            st.subheader(f"Plume Concentration Map at Receptor Height $z = {Z_RECEPTOR_M}$ m")
            st.warning(f"NOTE: The peak concentration at ground level ($z=0$) may be higher or located at a different distance than shown on this $z={Z_RECEPTOR_M}$ m plane.")
        else:
            st.subheader("Plume Concentration Contour Map (Ground Level, $z=0$)")

        # Visualization domain
        X_MAX = 4000
        Y_MAX = 500
        x_range = np.linspace(0.0, X_MAX, 200)
        y_range = np.linspace(-Y_MAX, Y_MAX, 100)
        X, Y = np.meshgrid(x_range, y_range)

        C_values = gaussian_plume_model(X, Y, Z_RECEPTOR_M, H_m, Q_g_s, U_m_s, stability_class)
        C_ug_m3 = C_values * 1e6

        if np.nanmax(C_ug_m3) > 1e-9:
            x_plot = x_range
            y_plot = y_range
            z_plot = C_ug_m3

            fig_tab1 = go.Figure(
                data=go.Contour(
                    z=z_plot,
                    x=x_plot,
                    y=y_plot,
                    colorscale='Viridis',
                    contours=dict(showlabels=False),
                    colorbar=dict(title="Concentration (µg m⁻³)"),
                    hovertemplate='x: %{x:.1f} m<br>y: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'
                )
            )

            fig_tab1.add_trace(
                go.Scatter(x=[0.0], y=[0.0], mode='markers', marker=dict(color='red', size=7), name='Stack (0,0)', hoverinfo='skip')
            )

            plot_title = f'Concentration Map at z={Z_RECEPTOR_M} m (Stability: {stability_class}, H: {H_m} m)'
            fig_tab1.update_layout(title=plot_title, xaxis_title='Downwind Distance (x, m)', yaxis_title='Crosswind Distance (y, m)', autosize=True, margin=dict(l=40, r=20, t=50, b=40))

            st.markdown("**Optional:** Animate plume advection over time (simple translation by wind speed).")
            animate_tab1 = st.checkbox("Enable time-evolution animation (advect plume with wind)", value=False, key='enable_animation_tab1')

            if animate_tab1:
                colta, coltb = st.columns([1, 1])
                with colta:
                    total_time_t = st.number_input("Total animation time (s)", min_value=1, max_value=3600, value=120, step=10, key='anim_total_time_tab1')
                with coltb:
                    n_frames_t = st.number_input("Number of frames", min_value=2, max_value=60, value=12, step=1, key='anim_n_frames_tab1')

                times_t = np.linspace(0.0, float(total_time_t), int(n_frames_t))
                frames_t = []
                zmin_t = np.nanmin(C_ug_m3)
                zmax_t = np.nanmax(C_ug_m3)
                for ti in times_t:
                    x_shift = float(U_m_s) * ti
                    X_adv = X - x_shift
                    C_adv = gaussian_plume_model(X_adv, Y, Z_RECEPTOR_M, H_m, Q_g_s, U_m_s, stability_class)
                    C_adv_ug = C_adv * 1e6
                    frame = go.Frame(
                        data=[go.Contour(z=C_adv_ug, x=x_plot, y=y_plot, colorscale='Viridis', zmin=zmin_t, zmax=zmax_t, contours=dict(showlabels=False))],
                        name=f"t{int(ti)}"
                    )
                    frames_t.append(frame)

                fig_anim_tab1 = go.Figure(
                    data=go.Contour(z=frames_t[0].data[0].z, x=x_plot, y=y_plot, colorscale='Viridis', zmin=zmin_t, zmax=zmax_t, contours=dict(showlabels=False)),
                    frames=frames_t
                )
                fig_anim_tab1.add_trace(go.Scatter(x=[0.0], y=[0.0], mode='markers', marker=dict(color='red', size=7), name='Stack (0,0)', hoverinfo='skip'))
                fig_anim_tab1.update_layout(
                    title=plot_title + " — Time Evolution (advection)",
                    xaxis_title='Downwind Distance (x, m)',
                    yaxis_title='Crosswind Distance (y, m)',
                    autosize=True,
                    margin=dict(l=40, r=20, t=80, b=40),
                    updatemenus=[
                        dict(
                            type="buttons",
                            showactive=True,
                            direction='left',
                            pad={'r': 10, 't': 10},
                            x=0.5,
                            xanchor='center',
                            y=-0.62,
                            yanchor='top',
                            buttons=[
                                dict(label='▶️ Play',
                                     method='animate',
                                     args=[None, {'frame': {'duration': max(100, int(1000 * total_time_t / len(frames_t))), 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}]),
                                dict(label='⏸️ Pause',
                                     method='animate',
                                     args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])
                            ]
                        )
                    ],
                    sliders=[
                        {
                            "steps": [
                                {
                                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                                    "label": f.name,
                                    "method": "animate"
                                }
                                for f in frames_t
                            ],
                            "currentvalue": {"prefix": "Frame: "},
                            "pad": {"t": 50}
                        }
                    ]
                )

                st.markdown("**Export Animation:** Create and download an MP4 of this time-evolution animation.")
                cole1, cole2, cole3 = st.columns([1, 1, 1])
                with cole1:
                    export_name_tab1 = st.text_input("Output filename (mp4)", value="EIRA_Plume_Animation", key='export_name_tab1')
                with cole2:
                    export_fps_tab1 = st.number_input("Frames per second (fps)", min_value=1, max_value=60, value=max(1, int(n_frames_t / max(1, total_time_t / 10))), step=1, key='export_fps_tab1')
                with cole3:
                    export_total_time_tab1 = st.number_input("Export total animation time (s)", min_value=1, max_value=3600, value=int(total_time_t), step=1, key='export_total_time_tab1')
                export_button_tab1 = st.button("Export animation to MP4", key='export_btn_tab1')

                st.plotly_chart(fig_anim_tab1, use_container_width=True)

                if export_button_tab1:
                    st.info("Rendering MP4 — this may take a moment. Please don't close the tab.")
                    times_export = np.linspace(0.0, float(export_total_time_tab1), int(n_frames_t))
                    frames_img = []
                    prog = st.progress(0)
                    total_frames = len(times_export)
                    base_name = export_name_tab1.strip()
                    if not base_name.lower().endswith('.mp4'):
                        base_name = base_name + '.mp4'
                    if f"_{int(export_total_time_tab1)}s" not in base_name:
                        name_no_ext, ext = os.path.splitext(base_name)
                        base_name = f"{name_no_ext}_{int(export_total_time_tab1)}s{ext}"
                    out_path = os.path.join("/tmp", base_name)
                    try:
                        vmin_export = zmin_t
                        vmax_export = zmax_t
                        x_extent = (x_plot[0], x_plot[-1])
                        y_extent = (y_plot[0], y_plot[-1])
                        for idx, ti in enumerate(times_export):
                            x_shift = float(U_m_s) * ti
                            X_adv = X - x_shift
                            C_adv = gaussian_plume_model(X_adv, Y, Z_RECEPTOR_M, H_m, Q_g_s, U_m_s, stability_class)
                            C_adv_ug = C_adv * 1e6
                            img = _render_frame_with_top_info(C_adv_ug, t_seconds=float(ti), vmin=vmin_export, vmax=vmax_export, x_extent=x_extent, y_extent=y_extent, figsize=(12, 4))
                            frames_img.append(img)
                            prog.progress(int((idx + 1) / total_frames * 100))
                        mp4_path = export_animation_to_mp4(frames_img, fps=int(export_fps_tab1), out_path=out_path)
                        with open(mp4_path, "rb") as f:
                            mp4_bytes = f.read()
                        st.success(f"MP4 created: {mp4_path}")
                        st.download_button("Download MP4", data=mp4_bytes, file_name=os.path.basename(mp4_path), mime="video/mp4")
                    except Exception as e:
                        st.error(f"Failed to create MP4: {e}")

            else:
                st.plotly_chart(fig_tab1, use_container_width=True)

            # Additional cross-sections (left in Tab 1 — unchanged)
            st.markdown("---")
            st.subheader("Additional Cross-Sectional Views")
            cross_section_option = st.selectbox("Choose additional cross-section to view:", ("None", "X vs Z (downwind vs height at y=0)", "Y vs Z (crosswind vs height at chosen x)"), index=0, key="tab1_cross_section")

            max_C_g_m3_loc, max_x_loc = find_max_concentration(H_m, Q_g_s, U_m_s, stability_class)

            if cross_section_option == "X vs Z (downwind vs height at y=0)":
                x_xvz = x_plot
                z_max_plot = max(1.5 * H_m, 300.0)
                z_xvz = np.linspace(0.0, z_max_plot, 120)
                X_xvz, Z_xvz = np.meshgrid(x_xvz, z_xvz)
                Cxz = np.zeros_like(X_xvz, dtype=float)
                for ii in range(X_xvz.shape[0]):
                    for jj in range(X_xvz.shape[1]):
                        xv = float(X_xvz[ii, jj])
                        zv = float(Z_xvz[ii, jj])
                        Cxz[ii, jj] = calculate_single_point_concentration(xv, 0.0, zv, H_m, Q_g_s, U_m_s, stability_class)
                Cxz_ug = Cxz * 1e6
                fig_xvz = go.Figure(data=go.Contour(z=Cxz_ug, x=x_xvz, y=z_xvz, colorscale='Viridis', contours=dict(showlabels=False), colorbar=dict(title="Concentration (µg m⁻³)"), hovertemplate='x: %{x:.1f} m<br>z: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'))
                fig_xvz.update_layout(title=f"X vs Z (y=0) — Downwind vs Height", xaxis_title='Downwind distance x (m)', yaxis_title='Height z (m)', autosize=True, margin=dict(l=40, r=20, t=50, b=40))
                fig_xvz.add_trace(go.Scatter(x=[0.0], y=[H_m], mode='markers', marker=dict(color='red', size=7), name='Stack H'))
                st.plotly_chart(fig_xvz, use_container_width=True)
                st.markdown(
                    """
                    **Significance:** This plot shows how concentration changes with downwind distance and elevation along the plume centerline (y=0). It helps assess plume rise and where elevated receptors (e.g. rooftops) may intersect high concentration zones.
                    """, unsafe_allow_html=True)

            elif cross_section_option == "Y vs Z (crosswind vs height at chosen x)":
                default_x_for_section = float(max_x_loc) if max_x_loc > 0 else float(X_MAX / 2.0)
                x_chosen = st.slider("Choose downwind distance for Y vs Z slice (x, m)", min_value=1.0, max_value=float(X_MAX), value=float(default_x_for_section), step=10.0, key="tab1_yvz_xslider")
                y_yvz = np.linspace(-Y_MAX, Y_MAX, 160)
                z_max_plot = max(1.5 * H_m, 300.0)
                z_yvz = np.linspace(0.0, z_max_plot, 120)
                Y_yvz, Z_yvz = np.meshgrid(y_yvz, z_yvz)
                Cyz = np.zeros_like(Y_yvz, dtype=float)
                for ii in range(Y_yvz.shape[0]):
                    for jj in range(Y_yvz.shape[1]):
                        yv = float(Y_yvz[ii, jj])
                        zv = float(Z_yvz[ii, jj])
                        Cyz[ii, jj] = calculate_single_point_concentration(float(x_chosen), yv, zv, H_m, Q_g_s, U_m_s, stability_class)
                Cyz_ug = Cyz * 1e6
                fig_yvz = go.Figure(data=go.Contour(z=Cyz_ug, x=y_yvz, y=z_yvz, colorscale='Viridis', contours=dict(showlabels=False), colorbar=dict(title="Concentration (µg m⁻³)"), hovertemplate='y: %{x:.1f} m<br>z: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'))
                fig_yvz.update_layout(title=f"Y vs Z at x = {x_chosen:.0f} m — Crosswind vs Height", xaxis_title='Crosswind distance y (m)', yaxis_title='Height z (m)', autosize=True, margin=dict(l=40, r=20, t=50, b=40))
                fig_yvz.add_trace(go.Scatter(x=[0.0], y=[H_m], mode='markers', marker=dict(color='red', size=7), name='Stack H (projection)'))
                st.plotly_chart(fig_yvz, use_container_width=True)
                st.markdown(
                    """
                    **Significance:** This vertical cross-section at a chosen downwind distance shows lateral and vertical dispersion at that location — useful to check ground-level exposure and how concentrations vary with height across the plume.
                    """, unsafe_allow_html=True)

        else:
            st.warning(f"The calculated concentration at $z={Z_RECEPTOR_M}$ m is near zero. The plume may be passing above or below this height, or parameters result in high dilution.")

        # Key model findings
        st.subheader("Key Model Findings (Ground-Level)")
        st.markdown("*(These metrics are always for the maximum ground-level concentration, $\\mathbf{C(x, y, 0)}$)*")

        max_C_g_m3, max_x = find_max_concentration(H_m, Q_g_s, U_m_s, stability_class)
        max_C_ug_m3 = max_C_g_m3 * 1e6

        mcol1, mcol2, mcol3 = st.columns([1.2, 1.0, 1.2])
        with mcol1:
            st.metric(label="Max Ground Conc. (center-line)", value=f"{max_C_ug_m3:,.2f} µg/m³")
        with mcol2:
            st.metric(label=r"$x_{max}$ (downwind)", value=f"{max_x:,.0f} m")
        if max_x > 0:
            sigma_y_max_x, sigma_z_max_x = get_dispersion_coefficients(max_x, stability_class)
            with mcol3:
                st.metric(label="Plume Half-Width ($2\\sigma_y$ at $x_{max}$)", value=f"{2 * sigma_y_max_x:,.1f} m")
        else:
            with mcol3:
                st.metric(label="Plume Half-Width ($2\\sigma_y$)", value="N/A")

        if max_x > 0:
            st.markdown(f"**Plume mixing height estimate ($4.3\\sigma_z$) at $x = {int(max_x)}$ m:** ${4.3 * sigma_z_max_x:,.1f} \\text{{ m}}$")
        else:
            st.info("Plume maximum could not be calculated. Ensure H is not too large or Q is not too small.")

    # --- Tab 2: 3D Visualization ---
    with tab2:
        _build_3d_geometry_tab(H_m, Q_g_s, U_m_s, stability_class)

    # --- Tab 3: Problem Solver ---
    with tab3:
        _build_solver_tab(stability_options)

    # --- Tab 4: Theory & Assumptions (moved to last) ---
    with tab4:
        _build_theory_tab()

def _build_3d_geometry_tab(H_m, Q_g_s, U_m_s, stability_class):
    st.subheader("3D Plume Geometry")
    st.markdown("Physical plume view with stack height, wind direction, centerline, sigma envelope, and downstream concentration slices.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        x_max_geom = st.slider("Downwind extent (m)", 500, 8000, 4000, step=250, key="geom_x_extent")
    with c2:
        n_slices = st.slider("Profile slices", 3, 10, 6, step=1, key="geom_slices")
    with c3:
        envelope_sigma = st.slider("Envelope sigma", 1.0, 4.0, 2.0, step=0.5, key="geom_sigma")
    with c4:
        slice_opacity = st.slider("Slice opacity", 0.15, 0.85, 0.45, step=0.05, key="geom_opacity")
    with c5:
        display_cutoff_pct = st.slider("Slice cutoff (%)", 0.0, 10.0, 1.0, step=0.5, key="geom_cutoff_pct")

    show_envelope = st.checkbox("Show plume envelope", value=True, key="geom_show_envelope")
    show_slice_contours = st.checkbox("Show profile contours", value=True, key="geom_show_contours")
    show_legacy_surface = st.checkbox("Show concentration surface at selected height", value=False, key="geom_show_legacy")

    try:
        fig = go.Figure()

        x_line = np.linspace(1.0, float(x_max_geom), 120)
        sigma_y_line, sigma_z_line = get_dispersion_coefficients(x_line, stability_class)
        y_extent = max(float(envelope_sigma * np.nanmax(sigma_y_line)), 100.0)
        z_extent = max(float(H_m + envelope_sigma * np.nanmax(sigma_z_line)), float(H_m * 1.8), 150.0)
        theta = np.linspace(0.0, 2.0 * np.pi, 42)

        # Ground plane gives the geometry a physical reference.
        gx = np.array([[0.0, float(x_max_geom)], [0.0, float(x_max_geom)]])
        gy = np.array([[-y_extent, -y_extent], [y_extent, y_extent]])
        gz = np.zeros_like(gx)
        fig.add_trace(go.Surface(
            x=gx, y=gy, z=gz,
            surfacecolor=np.zeros_like(gx),
            colorscale=[[0, "rgb(235,238,239)"], [1, "rgb(235,238,239)"]],
            showscale=False,
            opacity=0.35,
            name="Ground plane",
            hoverinfo="skip"
        ))

        pipe_radius = max(y_extent * 0.025, 8.0)
        pipe_theta = np.linspace(0.0, 2.0 * np.pi, 32)
        pipe_z = np.linspace(0.0, float(H_m), 8)
        PipeTheta, PipeZ = np.meshgrid(pipe_theta, pipe_z)
        PipeX = pipe_radius * np.cos(PipeTheta)
        PipeY = pipe_radius * np.sin(PipeTheta)
        fig.add_trace(go.Surface(
            x=PipeX,
            y=PipeY,
            z=PipeZ,
            surfacecolor=np.ones_like(PipeZ),
            colorscale=[[0, "rgb(150,58,46)"], [1, "rgb(178,74,58)"]],
            showscale=False,
            opacity=0.95,
            name="Stack pipe",
            hoverinfo="skip"
        ))
        brick_ring_x, brick_ring_y, brick_ring_z = [], [], []
        for z_ring in np.linspace(0.0, float(H_m), 9):
            brick_ring_x.extend((pipe_radius * np.cos(pipe_theta)).tolist() + [None])
            brick_ring_y.extend((pipe_radius * np.sin(pipe_theta)).tolist() + [None])
            brick_ring_z.extend(np.full_like(pipe_theta, z_ring).tolist() + [None])
        fig.add_trace(go.Scatter3d(
            x=brick_ring_x,
            y=brick_ring_y,
            z=brick_ring_z,
            mode="lines",
            line=dict(color="rgba(70,30,25,0.75)", width=2),
            showlegend=False,
            hoverinfo="skip"
        ))
        brick_seam_x, brick_seam_y, brick_seam_z = [], [], []
        brick_rows = np.linspace(0.0, float(H_m), 9)
        seam_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        for row_idx in range(len(brick_rows) - 1):
            z0 = brick_rows[row_idx]
            z1 = brick_rows[row_idx + 1]
            angle_offset = (np.pi / 8.0) if row_idx % 2 else 0.0
            for seam_angle in seam_angles + angle_offset:
                brick_seam_x.extend([pipe_radius * np.cos(seam_angle), pipe_radius * np.cos(seam_angle), None])
                brick_seam_y.extend([pipe_radius * np.sin(seam_angle), pipe_radius * np.sin(seam_angle), None])
                brick_seam_z.extend([z0, z1, None])
        fig.add_trace(go.Scatter3d(
            x=brick_seam_x,
            y=brick_seam_y,
            z=brick_seam_z,
            mode="lines",
            line=dict(color="rgba(70,30,25,0.55)", width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color="firebrick", width=9),
            name="Stack"
        ))
        height_marker_y = -pipe_radius * 2.8
        height_marker_x = -pipe_radius * 2.2
        fig.add_trace(go.Scatter3d(
            x=[height_marker_x, height_marker_x],
            y=[height_marker_y, height_marker_y],
            z=[0.0, float(H_m)],
            mode="lines+text",
            line=dict(color="black", width=3),
            text=["", "H"],
            textposition="middle right",
            textfont=dict(color="black", size=16),
            name="Height labels",
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=x_line, y=np.zeros_like(x_line), z=np.full_like(x_line, H_m),
            mode="lines",
            line=dict(color="royalblue", width=5),
            name="Plume centerline"
        ))
        wind_y = -y_extent * 0.78
        wind_z = max(H_m * 0.35, 20.0)
        wind_x0 = float(x_max_geom) * 0.08
        wind_x1 = float(x_max_geom) * 0.32
        arrow_len = float(x_max_geom) * 0.035
        arrow_spread = y_extent * 0.045
        fig.add_trace(go.Scatter3d(
            x=[wind_x0, wind_x1],
            y=[wind_y, wind_y],
            z=[wind_z, wind_z],
            mode="lines+text",
            line=dict(color="royalblue", width=6),
            text=["", "Wind direction"],
            textposition="top center",
            textfont=dict(color="royalblue", size=13),
            name="Wind direction"
        ))
        fig.add_trace(go.Scatter3d(
            x=[wind_x1 - arrow_len, wind_x1, wind_x1 - arrow_len],
            y=[wind_y - arrow_spread, wind_y, wind_y + arrow_spread],
            z=[wind_z, wind_z, wind_z],
            mode="lines",
            line=dict(color="royalblue", width=6),
            showlegend=False,
            hoverinfo="skip"
        ))

        if show_envelope:
            X_env, T_env = np.meshgrid(x_line, theta)
            sigma_y_env, sigma_z_env = get_dispersion_coefficients(X_env, stability_class)
            Y_env = envelope_sigma * sigma_y_env * np.cos(T_env)
            Z_env = H_m + envelope_sigma * sigma_z_env * np.sin(T_env)
            Z_env = np.maximum(Z_env, 0.0)
            fig.add_trace(go.Surface(
                x=X_env, y=Y_env, z=Z_env,
                surfacecolor=np.ones_like(X_env),
                colorscale=[[0, "rgb(150,150,150)"], [1, "rgb(150,150,150)"]],
                showscale=False,
                opacity=0.18,
                name=f"{envelope_sigma:g} sigma envelope",
                hoverinfo="skip"
            ))

        slice_xs = np.linspace(max(50.0, float(x_max_geom) / (n_slices + 1)), float(x_max_geom), int(n_slices))
        y_grid = np.linspace(-y_extent, y_extent, 70)
        z_grid = np.linspace(0.0, z_extent, 60)
        Yg, Zg = np.meshgrid(y_grid, z_grid)

        label_slice_x = float(slice_xs[min(len(slice_xs) // 2, len(slice_xs) - 1)])
        label_sigma_y, label_sigma_z = get_dispersion_coefficients(label_slice_x, stability_class)
        fig.add_trace(go.Scatter3d(
            x=[label_slice_x, label_slice_x],
            y=[0.0, float(label_sigma_y)],
            z=[float(H_m), float(H_m)],
            mode="lines",
            line=dict(color="rgb(45,90,180)", width=4),
            name="σy",
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=[label_slice_x],
            y=[float(label_sigma_y) * 1.2],
            z=[float(H_m) + z_extent * 0.035],
            mode="text",
            text=["σy"],
            textfont=dict(color="rgb(45,90,180)", size=15),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=[label_slice_x, label_slice_x],
            y=[float(label_sigma_y), float(label_sigma_y)],
            z=[float(H_m), float(H_m + label_sigma_z)],
            mode="lines+text",
            line=dict(color="red", width=6),
            text=["", "σz"],
            textposition="middle right",
            textfont=dict(color="red", size=16),
            name="σz",
            showlegend=False,
            hoverinfo="skip"
        ))
        cap_half_width = max(float(label_sigma_y) * 0.12, y_extent * 0.015)
        fig.add_trace(go.Scatter3d(
            x=[label_slice_x, label_slice_x, None, label_slice_x, label_slice_x],
            y=[
                float(label_sigma_y) - cap_half_width,
                float(label_sigma_y) + cap_half_width,
                None,
                float(label_sigma_y) - cap_half_width,
                float(label_sigma_y) + cap_half_width,
            ],
            z=[
                float(H_m),
                float(H_m),
                None,
                float(H_m + label_sigma_z),
                float(H_m + label_sigma_z),
            ],
            mode="lines",
            line=dict(color="red", width=4),
            showlegend=False,
            hoverinfo="skip"
        ))

        all_slice_c = []
        slice_payload = []
        for x_slice in slice_xs:
            is_first_slice = bool(x_slice == slice_xs[0])
            Xg = np.full_like(Yg, float(x_slice))
            C_slice = gaussian_plume_model(Xg, Yg, Zg, H_m, Q_g_s, U_m_s, stability_class) * 1e6
            all_slice_c.append(C_slice)
            slice_payload.append((Xg, Yg, Zg, C_slice))

            sy, sz = get_dispersion_coefficients(float(x_slice), stability_class)
            if show_slice_contours:
                for contour_sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
                    y_contour = contour_sigma * sy * np.cos(theta)
                    z_contour = np.maximum(H_m + contour_sigma * sz * np.sin(theta), 0.0)
                    fig.add_trace(go.Scatter3d(
                        x=np.full_like(theta, float(x_slice)),
                        y=y_contour,
                        z=z_contour,
                        mode="lines",
                        line=dict(color="rgba(165,35,35,0.42)", width=2),
                        name="Profile contours" if is_first_slice and contour_sigma == 0.5 else None,
                        showlegend=(is_first_slice and contour_sigma == 0.5),
                        hoverinfo="skip"
                    ))

            for sigma_level, color, width in [(1.0, "rgba(50,50,50,0.65)", 3), (2.0, "rgba(50,50,50,0.42)", 2)]:
                y_ellipse = sigma_level * sy * np.cos(theta)
                z_ellipse = np.maximum(H_m + sigma_level * sz * np.sin(theta), 0.0)
                fig.add_trace(go.Scatter3d(
                    x=np.full_like(theta, float(x_slice)),
                    y=y_ellipse,
                    z=z_ellipse,
                    mode="lines",
                    line=dict(color=color, width=width),
                    name=f"{sigma_level:g} sigma profile" if is_first_slice else None,
                    showlegend=is_first_slice,
                    hoverinfo="skip"
                ))

        max_c = max(float(np.nanmax(c)) for c in all_slice_c) if all_slice_c else 1.0
        color_max = max(np.log10(max_c + 1.0), 1.0)
        cutoff_value = max_c * float(display_cutoff_pct) / 100.0

        for idx, (Xg, Yg, Zg, C_slice) in enumerate(slice_payload):
            visible_mask = C_slice >= cutoff_value
            if display_cutoff_pct <= 0.0:
                visible_mask = np.ones_like(C_slice, dtype=bool)
            X_plot = np.where(visible_mask, Xg, np.nan)
            Y_plot = np.where(visible_mask, Yg, np.nan)
            Z_plot = np.where(visible_mask, Zg, np.nan)
            color_values = np.log10(C_slice + 1.0)
            color_values = np.where(visible_mask, color_values, np.nan)
            fig.add_trace(go.Surface(
                x=X_plot, y=Y_plot, z=Z_plot,
                surfacecolor=color_values,
                cmin=0.0,
                cmax=color_max,
                colorscale="Viridis",
                opacity=float(slice_opacity),
                showscale=(idx == len(slice_payload) - 1),
                colorbar=dict(title="log10(C+1)<br>µg m⁻³", len=0.6),
                name=f"x = {float(Xg[0, 0]):.0f} m",
                contours=dict(
                    z=dict(show=show_slice_contours, color="rgba(20,20,20,0.35)", width=1)
                )
            ))

        if show_legacy_surface:
            x_surf = np.linspace(1.0, float(x_max_geom), 140)
            y_surf = np.linspace(-y_extent, y_extent, 80)
            Xs, Ys = np.meshgrid(x_surf, y_surf)
            Cs = gaussian_plume_model(Xs, Ys, 0.0, H_m, Q_g_s, U_m_s, stability_class) * 1e6
            z_scale = max(z_extent / max(float(np.nanmax(Cs)), 1.0), 1.0)
            fig.add_trace(go.Surface(
                x=Xs, y=Ys, z=Cs * z_scale,
                surfacecolor=np.log10(Cs + 1.0),
                colorscale="Plasma",
                showscale=False,
                opacity=0.35,
                name="Ground concentration surface"
            ))

        fig.update_layout(
            title=f"Slide-style plume geometry (Stability {stability_class}, H = {H_m:g} m, Q = {Q_g_s:g} g/s, U = {U_m_s:g} m/s)",
            height=760,
            margin=dict(l=0, r=0, t=55, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.02),
            scene=dict(
                xaxis_title="Downwind distance x (m)",
                yaxis_title="Crosswind distance y (m)",
                zaxis_title="Height z (m)",
                xaxis=dict(range=[0, float(x_max_geom)]),
                yaxis=dict(range=[-y_extent, y_extent]),
                zaxis=dict(range=[0, z_extent]),
                aspectmode="manual",
                aspectratio=dict(x=2.6, y=1.0, z=0.9),
                camera=dict(eye=dict(x=1.7, y=-1.9, z=1.1))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The vertical curtains are Y-Z concentration profiles at fixed downwind distances. The translucent shell shows the selected sigma envelope around the plume centerline.")

    except Exception as _e:
        st.error(f"3D plume geometry unavailable (internal error): {_e}.")

def _build_solver_tab(stability_options):
    st.subheader("Point Concentration & $x_{max}$ Solver")
    st.markdown(
        "Solve for concentration at any receptor point using **custom parameters** "
        "independent of the sidebar. Includes wind-speed height correction and "
        "automatic Pasquill–Gifford stability class determination."
    )

    # ── Notation guide ───────────────────────────────────────────────────────
    with st.expander("ℹ️  Problem notation guide — C(x, y, z, H)", expanded=False):
        st.markdown(
            r"""
            Problems are often stated as **C(x, y, z, H)** where all four coordinates are given:

            | Symbol | Meaning | Typical example |
            |--------|---------|-----------------|
            | $x$ | Downwind distance from stack (m) | 1000 m |
            | $y$ | Crosswind offset from centreline (m) | 50 m |
            | $z$ | Receptor height above ground (m) | 0 m (ground level) |
            | $H$ | **Effective** stack height = physical height + plume rise Δh (m) | 250 m |

            *Example: "Estimate SO₂ concentration at (1000, 50, 0, 250)"
            → x = 1000 m, y = 50 m, z = 0 m, H = 250 m*
            """
        )

    st.markdown("---")

    # ── 1. Dispersion coefficient source ─────────────────────────────────────
    st.subheader("1. Dispersion Coefficient Source")
    dispersion_mode = st.radio(
        "Choose how dispersion coefficients are obtained:",
        ('Pasquill-Gifford Curves (default)', 'Custom $\\sigma_y$ and $\\sigma_z$ Input'),
        index=0, key='dispersion_mode'
    )
    use_custom_sigma = (dispersion_mode == 'Custom $\\sigma_y$ and $\\sigma_z$ Input')

    if use_custom_sigma:
        st.warning(
            "With fixed custom σ values the $x_{max}$ calculation is not meaningful "
            "because σ is distance-independent."
        )
        colS1, colS2 = st.columns(2)
        with colS1:
            solver_sigma_y = st.number_input(
                "Custom $\\sigma_y$ (Lateral, m)", min_value=0.1, value=100.0,
                step=10.0, key='sigma_y_solver'
            )
        with colS2:
            solver_sigma_z = st.number_input(
                "Custom $\\sigma_z$ (Vertical, m)", min_value=0.1, value=50.0,
                step=5.0, key='sigma_z_solver'
            )
    else:
        solver_sigma_y = None
        solver_sigma_z = None

    st.markdown("---")

    # ── 2. Source parameters ──────────────────────────────────────────────────
    st.subheader("2. Source Parameters")
    colA, colB = st.columns(2)
    with colA:
        solver_Q = st.number_input(
            "Emission Rate ($Q$, g/s)", min_value=0.001, value=100.0,
            step=10.0, format="%.3f", key='Q_solver'
        )
    with colB:
        solver_H = st.number_input(
            "Effective Stack Height ($H$, m)", min_value=1.0, value=100.0,
            step=5.0, key='H_solver'
        )

    st.markdown("---")

    # ── 3. Wind speed ─────────────────────────────────────────────────────────
    st.subheader("3. Wind Speed")
    wind_input_mode = st.radio(
        "Wind speed measurement reference:",
        (
            "At 10 m surface (use directly)",
            "At stack altitude → convert DOWN to surface (power law, fixed p)",
            "At reference height → convert UP to stack height (power law, class-based p)",
        ),
        index=0, key='wind_input_mode'
    )

    if wind_input_mode == "At 10 m surface (use directly)":
        solver_U_surface = st.number_input(
            "Surface Wind Speed $U_{10}$ (m/s)",
            min_value=0.1, value=5.0, step=0.5, key='U_solver_surface'
        )
        solver_U_model = solver_U_surface
        wind_p_used = None
        wind_correction_info = None

    elif wind_input_mode == "At stack altitude → convert DOWN to surface (power law, fixed p)":
        colW1, colW2 = st.columns(2)
        with colW1:
            solver_U_stack = st.number_input(
                "Wind Speed at Stack Altitude $u_H$ (m/s)",
                min_value=0.1, value=6.0, step=0.5, key='U_solver_stack'
            )
        with colW2:
            wind_p_used = st.number_input(
                "Wind Shear Exponent $p$",
                min_value=0.05, max_value=0.60, value=0.25, step=0.05,
                help="Typical values: 0.07–0.15 (unstable), 0.25 (neutral), 0.30–0.35 (stable).",
                key='wind_p_solver'
            )
        solver_U_surface = correct_wind_to_surface(solver_U_stack, float(solver_H), float(wind_p_used))
        solver_U_model = solver_U_surface
        wind_correction_info = {
            'mode': 'stack_to_surface',
            'u_in': solver_U_stack, 'H': float(solver_H),
            'p': float(wind_p_used), 'u_out': solver_U_surface
        }
        st.info(
            f"**Surface wind:**  "
            f"$u_0 = u_H \\times (10/H)^p = {solver_U_stack:.2f} "
            f"\\times (10/{solver_H:.0f})^{{{wind_p_used:.2f}}} = "
            f"\\mathbf{{{solver_U_surface:.3f}}}$ **m/s**"
        )

    else:  # reference height → stack height
        colW1, colW2, colW3 = st.columns(3)
        with colW1:
            solver_U_ref = st.number_input(
                "Reference Wind Speed $u_1$ (m/s)",
                min_value=0.1, value=5.0, step=0.5, key='U_solver_ref'
            )
        with colW2:
            solver_z_ref = st.number_input(
                "Reference Height $z_1$ (m)",
                min_value=0.1, value=10.0, step=1.0, key='z_ref_solver'
            )
        with colW3:
            wind_surface_type = st.selectbox(
                "Surface type",
                options=list(WIND_PROFILE_EXPONENTS.keys()),
                index=0, key='wind_surface_solver'
            )
        # Need stability class for class-based p — use a provisional selection
        prov_stab = st.selectbox(
            "Stability class for p (provisional — used only for wind correction)",
            options=list(stability_options.keys()),
            format_func=lambda x: stability_options[x],
            index=3, key='prov_stab_wind'
        )
        solver_U_model, wind_p_used = calculate_wind_speed_at_height(
            solver_U_ref, float(solver_z_ref), float(solver_H), prov_stab, wind_surface_type
        )
        solver_U_surface = solver_U_ref
        wind_correction_info = {
            'mode': 'ref_to_stack',
            'u_in': solver_U_ref, 'z_ref': float(solver_z_ref),
            'H': float(solver_H), 'p': float(wind_p_used), 'u_out': solver_U_model
        }
        st.info(
            f"**Stack-height wind:**  "
            f"$U(H) = U(z_1)\\,(H/z_1)^p = {solver_U_ref:.2f}"
            f"\\times({solver_H:.0f}/{solver_z_ref:.0f})^{{{wind_p_used:.2f}}} = "
            f"\\mathbf{{{solver_U_model:.3f}}}$ **m/s** (used in model)"
        )

    st.markdown("---")

    # ── 4. Atmospheric stability class ────────────────────────────────────────
    st.subheader("4. Atmospheric Stability Class")

    if not use_custom_sigma:
        stability_input_mode = st.radio(
            "How to determine stability class:",
            ("Manual selection", "Auto-determine from Pasquill–Gifford table"),
            index=0, key='stability_input_mode'
        )
        auto_stability = (stability_input_mode == "Auto-determine from Pasquill–Gifford table")
    else:
        auto_stability = False

    if not use_custom_sigma and auto_stability:
        with st.expander("📋 Pasquill–Gifford Stability Table (reference)", expanded=True):
            st.markdown(
                """
                | Surface Wind (m/s) | Day — Strong ☀️ | Day — Moderate 🌤 | Day — Slight 🌥 | Night — Cloudy ≥4/8 | Night — Clear <3/8 |
                |--------------------|:---:|:---:|:---:|:---:|:---:|
                | < 2  | A   | A–B | B | E | F |
                | 2–3  | A–B | B   | C | E | F |
                | 3–5  | B   | B–C | C | D | E |
                | 5–6  | C   | C–D | D | D | D |
                | > 6  | C   | D   | D | D | D |

                *Mixed classes (e.g. A–B) are resolved to the more stable class for computation.*
                """
            )

        colPG1, colPG2 = st.columns(2)
        with colPG1:
            pg_time = st.selectbox("Time of day", ["Day", "Night"], key='pg_time', index=0)
        with colPG2:
            if pg_time == "Day":
                pg_condition_label = st.selectbox(
                    "Incoming solar radiation",
                    ["Strong (sunny summer day)", "Moderate", "Slight (overcast/winter)"],
                    key='pg_condition_day', index=0
                )
                condition_key = pg_condition_label.split()[0].lower()
            else:
                pg_condition_label = st.selectbox(
                    "Cloud cover at night",
                    ["Cloudy (≥4/8 cloud cover)", "Clear (<3/8 cloud cover)"],
                    key='pg_condition_night', index=0
                )
                condition_key = "cloudy" if "Cloudy" in pg_condition_label else "clear"

        # Use surface wind for the PG table lookup
        raw_label, solver_stab_class = determine_stability_class(
            float(solver_U_surface), pg_time.lower(), condition_key
        )
        st.success(
            f"**PG Table lookup →**  Surface wind: **{solver_U_surface:.2f} m/s** | "
            f"Condition: **{pg_condition_label}** | "
            f"Table entry: **{raw_label}** → resolved to **Class {solver_stab_class}** "
            f"({stability_options[solver_stab_class]})"
        )

    elif not use_custom_sigma:
        solver_stab_class = st.selectbox(
            "Atmospheric Stability Class",
            options=list(stability_options.keys()),
            format_func=lambda x: stability_options[x],
            index=3, key='stability_solver_key'
        )
        raw_label = solver_stab_class
        pg_time = None
        pg_condition_label = None
    else:
        solver_stab_class = 'D'
        raw_label = 'D'
        pg_time = None
        pg_condition_label = None

    st.markdown("---")

    # ── 5. Receptor location ──────────────────────────────────────────────────
    st.subheader("5. Receptor Location — C(x, y, z, H)")
    colX, colY, colZ = st.columns(3)
    with colX:
        solver_x = st.number_input(
            "Downwind Distance ($x$, m)", min_value=1.0, value=1000.0,
            step=10.0, key='x_input_solver'
        )
    with colY:
        solver_y = st.number_input(
            "Crosswind Distance ($y$, m)", value=0.0, step=10.0,
            key='y_input_solver'
        )
    with colZ:
        solver_z = st.number_input(
            "Receptor Height ($z$, m)", value=0.0, step=10.0,
            key='z_input_solver'
        )

    st.markdown("---")

    # ── Calculate ─────────────────────────────────────────────────────────────
    if st.button("▶  Run Calculation", key='solve_button', type='primary'):

        st.subheader("Step-by-Step Solution")

        # Step 1 · Given data ─────────────────────────────────────────────────
        with st.expander("Step 1 — Given Data", expanded=True):
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Emission rate $Q$ | **{solver_Q:.3f} g/s** |
| Effective stack height $H$ | **{solver_H:.1f} m** |
| Wind speed (model input $U$) | **{solver_U_model:.3f} m/s** |
| Receptor $(x,\\,y,\\,z)$ | **({solver_x:.0f}, {solver_y:.0f}, {solver_z:.0f}) m** |
""")
            with col_g2:
                if wind_correction_info is not None:
                    if wind_correction_info['mode'] == 'stack_to_surface':
                        st.markdown("**Wind height correction (stack → surface):**")
                        st.latex(
                            rf"u_0 = u_H \left(\frac{{10}}{{H}}\right)^p = "
                            rf"{wind_correction_info['u_in']:.2f}"
                            rf"\left(\frac{{10}}{{{wind_correction_info['H']:.0f}}}\right)^{{{wind_correction_info['p']:.2f}}}"
                            rf"= {wind_correction_info['u_out']:.3f}\;\text{{m/s}}"
                        )
                    else:
                        st.markdown("**Wind height correction (reference → stack):**")
                        st.latex(
                            rf"U(H) = U(z_1)\left(\frac{{H}}{{z_1}}\right)^p = "
                            rf"{wind_correction_info['u_in']:.2f}"
                            rf"\left(\frac{{{wind_correction_info['H']:.0f}}}{{{wind_correction_info['z_ref']:.0f}}}\right)^{{{wind_correction_info['p']:.2f}}}"
                            rf"= {wind_correction_info['u_out']:.3f}\;\text{{m/s}}"
                        )
                else:
                    st.markdown("Wind speed entered directly at 10 m — no height correction applied.")

        # Step 2 · Stability class ────────────────────────────────────────────
        with st.expander("Step 2 — Stability Class Determination", expanded=True):
            if not use_custom_sigma and auto_stability:
                st.markdown(
                    f"From the Pasquill–Gifford table:  \n"
                    f"Surface wind **{solver_U_surface:.2f} m/s** + **{pg_condition_label}** "
                    f"→ table entry **{raw_label}** → resolved to **Stability Class {solver_stab_class}** "
                    f"({stability_options[solver_stab_class]})"
                )
            elif use_custom_sigma:
                st.markdown(
                    f"Using custom σ values (σ_y = {solver_sigma_y:.2f} m, "
                    f"σ_z = {solver_sigma_z:.2f} m) — stability class not required."
                )
            else:
                st.markdown(
                    f"Manually selected: **Class {solver_stab_class}** "
                    f"({stability_options[solver_stab_class]})"
                )

        # Step 3 · Dispersion coefficients ────────────────────────────────────
        with st.expander(f"Step 3 — Dispersion Coefficients at x = {solver_x:.0f} m", expanded=True):
            if use_custom_sigma:
                sigma_y_used = float(solver_sigma_y)
                sigma_z_used = float(solver_sigma_z)
                st.markdown(
                    f"Custom values supplied:  \n"
                    f"$\\sigma_y = {sigma_y_used:.2f}$ m,  "
                    f"$\\sigma_z = {sigma_z_used:.2f}$ m"
                )
            else:
                sigma_y_used, sigma_z_used = get_dispersion_coefficients(float(solver_x), solver_stab_class)
                idx = SIGMA_COEFFS['index'][solver_stab_class]
                Ay = SIGMA_COEFFS['y']['A'][idx]
                By = SIGMA_COEFFS['y']['B'][idx]
                Az = SIGMA_COEFFS['z']['A'][idx]
                Bz = SIGMA_COEFFS['z']['B'][idx]
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("**Lateral ($\\sigma_y$):**")
                    st.latex(
                        rf"\sigma_y = {Ay}\cdot x^{{{By}}} = "
                        rf"{Ay}\times{solver_x:.0f}^{{{By}}} = {sigma_y_used:.2f}\;\text{{m}}"
                    )
                with col_s2:
                    st.markdown("**Vertical ($\\sigma_z$):**")
                    st.latex(
                        rf"\sigma_z = {Az}\cdot x^{{{Bz}}} = "
                        rf"{Az}\times{solver_x:.0f}^{{{Bz}}} = {sigma_z_used:.2f}\;\text{{m}}"
                    )

        # Step 4 · Concentration at receptor ──────────────────────────────────
        with st.expander("Step 4 — Concentration at Receptor", expanded=True):
            if use_custom_sigma:
                point_C_g_m3 = calculate_point_concentration_custom_sigma(
                    float(solver_x), float(solver_y), float(solver_z),
                    float(solver_H), float(solver_Q), float(solver_U_model),
                    sigma_y_used, sigma_z_used
                )
            else:
                point_C_g_m3 = calculate_single_point_concentration(
                    float(solver_x), float(solver_y), float(solver_z),
                    float(solver_H), float(solver_Q), float(solver_U_model),
                    solver_stab_class
                )
            point_C_ug_m3 = point_C_g_m3 * 1e6

            # Numerical substitution
            exp_y_val  = np.exp(-float(solver_y)**2 / (2 * sigma_y_used**2))
            exp_zr_val = np.exp(-(float(solver_z) - float(solver_H))**2 / (2 * sigma_z_used**2))
            exp_zi_val = np.exp(-(float(solver_z) + float(solver_H))**2 / (2 * sigma_z_used**2))
            denom_val  = 2 * np.pi * float(solver_U_model) * sigma_y_used * sigma_z_used

            st.markdown("**Gaussian plume equation:**")
            st.latex(
                r"C(x,y,z) = \frac{Q}{2\pi\,U\,\sigma_y\,\sigma_z}"
                r"\exp\!\left(-\frac{y^2}{2\sigma_y^2}\right)"
                r"\left[\exp\!\left(-\frac{(z-H)^2}{2\sigma_z^2}\right)"
                r"+\exp\!\left(-\frac{(z+H)^2}{2\sigma_z^2}\right)\right]"
            )
            st.markdown("**Substituting values:**")
            st.latex(
                rf"C = \frac{{{solver_Q:.3f}}}{{{denom_val:.4f}}}"
                rf"\times{exp_y_val:.4f}"
                rf"\times\left[{exp_zr_val:.4f}+{exp_zi_val:.4f}\right]"
                rf"= {point_C_g_m3:.4e}\;\text{{g/m}}^3"
            )
            st.success(
                f"**C({solver_x:.0f}, {solver_y:.0f}, {solver_z:.0f}) = "
                f"{point_C_ug_m3:,.4f} µg/m³ "
                f"= {point_C_g_m3:.4e} g/m³**"
            )

        # Step 5 · Maximum ground concentration ───────────────────────────────
        if not use_custom_sigma:
            with st.expander("Step 5 — Maximum Ground-Level Concentration & Location", expanded=True):
                max_C_g_m3, max_x_loc = find_max_concentration(
                    float(solver_H), float(solver_Q), float(solver_U_model), solver_stab_class
                )
                max_C_ug_m3 = max_C_g_m3 * 1e6
                sigma_y_xmax, sigma_z_xmax = get_dispersion_coefficients(max_x_loc, solver_stab_class)

                col_mx1, col_mx2, col_mx3 = st.columns(3)
                with col_mx1:
                    st.metric("$C_{max}$ (ground, centreline)", f"{max_C_ug_m3:,.4f} µg/m³")
                with col_mx2:
                    st.metric("Location $x_{max}$", f"{max_x_loc:,.0f} m")
                with col_mx3:
                    st.metric("Plume half-width $2\\sigma_y$ at $x_{max}$", f"{2*sigma_y_xmax:,.1f} m")
                st.markdown(
                    f"Plume mixing-height estimate $4.3\\,\\sigma_z$ at $x_{{max}}$: "
                    f"**{4.3*sigma_z_xmax:,.1f} m**"
                )
        else:
            max_C_g_m3 = point_C_g_m3
            max_x_loc = float(solver_x)

        # Results summary row ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Results Summary")
        colR1, colR2, colR3 = st.columns(3)
        with colR1:
            st.metric(
                label=f"C({solver_x:.0f}, {solver_y:.0f}, {solver_z:.0f}) m",
                value=f"{point_C_ug_m3:,.4f} µg/m³"
            )
        with colR2:
            st.metric(label="$\\sigma_y$ at x", value=f"{sigma_y_used:,.2f} m")
        with colR3:
            st.metric(label="$\\sigma_z$ at x", value=f"{sigma_z_used:,.2f} m")

        st.markdown("---")

        # Contour visualisation ───────────────────────────────────────────────
        st.subheader("Concentration Contour Map (hover / zoom / pan)")

        x_plot_max = max(2.0 * float(solver_x), 500.0)
        x_plot_min = max(0.1, float(solver_x) * 0.01)
        x_plot = np.linspace(x_plot_min, x_plot_max, 160)

        crosswind_half = max(3.0 * sigma_y_used, 200.0)
        y_plot = np.linspace(-crosswind_half, crosswind_half, 120)
        Xp, Yp = np.meshgrid(x_plot, y_plot)

        if use_custom_sigma:
            Cp = np.zeros_like(Xp, dtype=float)
            for ii in range(Xp.shape[0]):
                for jj in range(Xp.shape[1]):
                    Cp[ii, jj] = calculate_point_concentration_custom_sigma(
                        float(Xp[ii, jj]), float(Yp[ii, jj]),
                        float(solver_z), float(solver_H),
                        float(solver_Q), float(solver_U_model),
                        solver_sigma_y, solver_sigma_z
                    )
            plot_mode_label = f"Solver — Custom σ, z = {solver_z:.0f} m"
        else:
            Cp = gaussian_plume_model(
                Xp, Yp, float(solver_z), float(solver_H),
                float(solver_Q), float(solver_U_model), solver_stab_class
            )
            plot_mode_label = (
                f"Solver — Class {solver_stab_class}, "
                f"U = {solver_U_model:.2f} m/s, z = {solver_z:.0f} m"
            )

        Cp_ug = Cp * 1e6
        fig_pl = go.Figure(data=go.Contour(
            z=Cp_ug, x=x_plot, y=y_plot,
            colorscale='Viridis', contours=dict(showlabels=False),
            colorbar=dict(title="Concentration (µg m⁻³)"),
            hovertemplate='x: %{x:.1f} m<br>y: %{y:.1f} m<br>C: %{z:.4f} µg m⁻³<extra></extra>'
        ))
        # Mark receptor and stack
        fig_pl.add_trace(go.Scatter(
            x=[float(solver_x)], y=[float(solver_y)],
            mode='markers',
            marker=dict(color='cyan', size=10, symbol='x', line=dict(width=2)),
            name=f'Receptor ({solver_x:.0f}, {solver_y:.0f})'
        ))
        fig_pl.add_trace(go.Scatter(
            x=[0.0], y=[0.0], mode='markers',
            marker=dict(color='red', size=8),
            name='Stack (0, 0)', hoverinfo='skip'
        ))
        fig_pl.update_layout(
            title=plot_mode_label,
            xaxis_title='Downwind distance x (m)',
            yaxis_title='Crosswind distance y (m)',
            autosize=True, margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_pl, use_container_width=True)
        st.caption(
            "Red dot = stack origin (0, 0).  "
            "Cyan × = your receptor point.  "
            "Hover anywhere to read concentration."
        )

def _build_theory_tab():
    st.header("Gaussian Plume Model Theory & Assumptions")
    st.markdown(
        """
        The Gaussian Plume Model (GPM) is the fundamental steady-state model for predicting the dispersion of continuous, buoyant pollutants released from a single point source, such as a chimney stack. It assumes that the pollutant concentration forms a Gaussian (normal) distribution in both the lateral ($y$) and vertical ($z$) directions, normal to the mean wind direction ($x$).
        
        ***

        ### Core Equation for $\mathbf{C(x, y, z)}$

        The general equation, which includes the vertical height $z$ and the effect of total ground reflection (the **virtual image source**), is:
        """
    )

    st.latex(r"""
    C(x, y, z) \;=\; \frac{Q}{2\pi \, U \, \sigma_y \, \sigma_z}
    \exp\!\left(-\frac{y^2}{2\sigma_y^2}\right)
    \left[
    \exp\!\left(-\frac{(z-H)^2}{2\sigma_z^2}\right)
    +
    \exp\!\left(-\frac{(z+H)^2}{2\sigma_z^2}\right)
    \right]
    """)

    st.markdown(
        """
        Where:
        * $C(x, y, z)$: Concentration at point $(x, y, z)$ ($\mu g/m^3$)
        * $Q$: Source Emission Rate ($g/s$)
        * $U$: Mean Wind Speed ($m/s$)
        * $H$: **Effective Stack Height** ($m$) - Sum of physical stack height and plume rise ($\Delta h$).
        * $\sigma_y$ and $\sigma_z$: Lateral and Vertical Dispersion Coefficients ($m$), determined by atmospheric stability and distance $x$.
        
        ***
        
        ### Key Model Assumptions

        1.  **Steady State:** Emission rate ($Q$) and wind speed ($U$) are constant.
        2.  **Uniform Wind:** Wind flows uniformly in the $x$-direction (straight-line flow).
        3.  **Total Reflection:** The pollutant is completely reflected off the ground surface (modeled by the virtual image source term: $\exp(-(z+H)^2 / (2\sigma_z^2))$).
        4.  **Gaussian Distribution:** Concentration profiles are Gaussian in the cross-wind and vertical directions.
        
        ***
        
        ### Visualization in GPM

        The contour map in the **Plume Visualizer** tab represents a 2D slice of the 3D concentration field.

        * **Ground Level ($\mathbf{z=0}$):** This is the default view, showing the concentration at the Earth's surface, critical for assessing the highest ground impact.
        * **Custom Receptor Height ($\mathbf{z>0}$):** When the "Visualize at a Custom Height (z)" toggle is enabled, the map displays a **horizontal slice** of the plume concentration at a specific elevation, $z$ (e.g., the height of a rooftop).
        * **Isopleths/Contours:** The map displays contour lines (isopleths) connecting points of **equal pollutant concentration** on the selected $z$-plane.

        ***
        """, unsafe_allow_html=True)

    st.markdown("### Wind Velocity and Turbulence")
    st.markdown(
        """
        The mean wind speed variation with altitude in the planetary boundary layer can be estimated using the empirical power law:
        """
    )
    st.latex(r"\frac{u}{u_1} = \left(\frac{z}{z_1}\right)^p")
    st.markdown(
        """
        Where $u$ is the wind speed at altitude $z$, $u_1$ is the reference wind speed at altitude $z_1$, and $p$ depends on atmospheric stability and surface roughness.

        In the Gaussian plume equation, $U$ should represent the wind speed at the effective stack height. Field wind measurements are commonly reported near 10 m above ground level, so the app includes an optional sidebar correction that estimates $U(H)$ from a reference wind speed:
        """
    )
    st.latex(r"U(H) = U(z_1)\left(\frac{H}{z_1}\right)^p")
    st.markdown(
        """
        When this option is enabled, the entered wind speed is treated as the reference speed $U(z_1)$ and the calculated stack-height wind speed $U(H)$ is used in all plume concentration calculations. Because concentration is inversely proportional to $U$, larger stack-height wind speeds generally reduce predicted concentrations, all else being equal.
        """
    )
    wind_profile_table = {
        "Stability Class": ["A", "B", "C", "D", "E", "F"],
        "p - Rough Surface (urban)": [0.15, 0.15, 0.20, 0.25, 0.30, 0.30],
        "p - Smooth Surface (rural)": [0.07, 0.07, 0.10, 0.15, 0.35, 0.35],
    }
    st.table(wind_profile_table)
    st.caption("The wind-profile correction is an empirical approximation. It does not replace site-specific meteorological profiling or regulatory dispersion model preprocessing.")

    st.markdown("***")

    # Restored reference & academic profiles block (with links)
    st.markdown(
        """
        ***Primary Source Reference:*** *These notes are adapted from the lecture materials on Gaussian Plumes by E. Savory, available at* [eng.uwo.ca/people/esavory/gaussian plumes.pdf](https://www.eng.uwo.ca/people/esavory/gaussian%20plumes.pdf).
        
        For further study and reference on Environmental Engineering and dispersion modeling, you may consult the work and class notes of:

        **Dr. Abhradeep Majumder**  
        Assistant Professor, Department of Civil Engineering, BITS Pilani-Hyderabad Campus  
        Academic Profiles: [Scopus](https://www.scopus.com/authid/detail.uri?authorId=57191504507), [ORCID](https://orcid.org/0000-0002-0186-6450), [Google Scholar](https://scholar.google.co.in/citations?user=mnJ5zdwAAAAJ&hl=en&oi=ao), [LinkedIn](https://linkedin.com/in/abhradeep-majumder-36503777/)

        **Dr. Krishna C. Etika**  
        Associate Professor, Department of Chemical Engineering, BITS Pilani, Pilani Campus  
        Academic Profiles: [BITS Pilani Profile](https://www.bits-pilani.ac.in/pilani/krishna-c-etika/), [LinkedIn](https://shorturl.at/P64gp)

        ---
        ###### Application Development

        This interactive Gaussian Plume Dispersion Model application was developed by **Subodh Purohit** ([LinkedIn](https://www.linkedin.com/in/subodhpurohit/)) as an educational tool.
        """, unsafe_allow_html=True)

# -------------------------
# Application Entrypoint
# -------------------------
def run_app():
    _configure_page()
    Q_g_s, H_m, U_m_s, stability_class, Z_RECEPTOR_M, stability_options = _sidebar_inputs()
    _build_visualizer_tab(Q_g_s, H_m, U_m_s, stability_class, Z_RECEPTOR_M, stability_options)

if __name__ == '__main__':
    run_app()
