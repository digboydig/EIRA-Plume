"""
Gaussian Plume App.

Capabilities:
- Interactive visualization of Gaussian plume concentration maps at ground level or user-defined receptor heights.
- Time-evolution advection animation with optional MP4 export of rendered frames.
- Point-concentration solver with support for Pasquill–Gifford curves or user-supplied σ_y/σ_z values.
- Cross-sectional views (X vs Z and Y vs Z) and summary metrics for quick assessment of peak ground impacts.

Author: Subodh Purohit
Motivation: Dr. Abhradeep Majumder, Ph.D
Purpose: Educational use only.
"""

# -------------------------
# Imports
# -------------------------
import io
import os
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

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

# -------------------------
# MP4 Export Helpers
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
    U_m_s = st.sidebar.slider('Hidden label', 1.0, 20.0, 5.0, step=0.5, key='U_slider', label_visibility='collapsed')

    stability_options = {'A': 'A - Extremely Unstable', 'B': 'B - Moderately Unstable', 'C': 'C - Slightly Unstable',
                         'D': 'D - Neutral (Overcast/High Wind)', 'E': 'E - Slightly Stable', 'F': 'F - Moderately Stable'}
    stability_class = st.sidebar.selectbox("Atmospheric Stability Class", options=list(stability_options.keys()),
                                         format_func=lambda x: stability_options[x], index=3, key='stability_slider')

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
    st.markdown("An interactive model to explore how source parameters and atmospheric stability affect pollutant spread and ground-level concentration.")

    tab1, tab2, tab3 = st.tabs(["Plume Visualizer", "Problem Solver", "Theory & Assumptions"])

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

            # Additional cross-sections
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

    # --- Tab 2: Problem Solver ---
    with tab2:
        _build_solver_tab(stability_options)

    # --- Tab 3: Theory & Assumptions ---
    with tab3:
        _build_theory_tab()

def _build_solver_tab(stability_options):
    st.subheader("Point Concentration & $x_{max}$ Solver")
    st.markdown("Calculate concentrations and the maximum ground-level location using **custom parameters** independent of the visualizer's sidebar.")

    st.subheader("1. Dispersion Mode")
    dispersion_mode = st.radio("Choose Dispersion Coefficient Source:", ('Pasquill-Gifford Curves (default)', 'Custom $\\sigma_y$ and $\\sigma_z$ Input'), index=0, key='dispersion_mode')
    use_custom_sigma = (dispersion_mode == 'Custom $\\sigma_y$ and $\\sigma_z$ Input')

    if use_custom_sigma:
        st.warning("When using custom $\\sigma$ values, the maximum concentration ($x_{max}$) calculation is not meaningful as $\\sigma$ is fixed, not distance-dependent.")
        colS1, colS2 = st.columns(2)
        with colS1:
            solver_sigma_y = st.number_input("Custom $\\sigma_y$ (Lateral, m)", min_value=0.1, value=100.0, step=10.0, key='sigma_y_solver')
        with colS2:
            solver_sigma_z = st.number_input("Custom $\\sigma_z$ (Vertical, m)", min_value=0.1, value=50.0, step=5.0, key='sigma_z_solver')

    st.subheader("2. Source & Atmospheric Parameters")
    colA, colB, colC = st.columns(3)
    with colA:
        solver_Q = st.number_input("Emission Rate ($Q$, g/s)", min_value=1.0, value=100.0, step=10.0, key='Q_solver')
    with colB:
        solver_H = st.number_input("Effective Stack Height ($H$, m)", min_value=1.0, value=100.0, step=5.0, key='H_solver')
    with colC:
        solver_U = st.number_input("Wind Speed ($U$, m/s)", min_value=0.1, value=5.0, step=0.5, key='U_solver')

    if not use_custom_sigma:
        solver_stab_class = st.selectbox("4. Atmospheric Stability Class", options=list(stability_options.keys()), format_func=lambda x: stability_options[x], index=3, key='stability_solver_key')
    else:
        solver_stab_class = 'D'

    st.markdown("---")
    st.subheader("3. Point Location Input")
    colX, colY, colZ = st.columns(3)
    with colX:
        solver_x = st.number_input("Downwind Distance ($x$, m)", min_value=1.0, value=1000.0, step=10.0, key='x_input_solver')
    with colY:
        solver_y = st.number_input("Crosswind Distance ($y$, m)", value=0.0, step=10.0, key='y_input_solver')
    with colZ:
        solver_z = st.number_input("Vertical Height ($z$, m)", value=0.0, step=10.0, key='z_input_solver')

    if st.button("Run Calculations for Custom Parameters", key='solve_button'):
        st.subheader("Calculated Results")

        if use_custom_sigma:
            point_C_g_m3 = calculate_point_concentration_custom_sigma(float(solver_x), float(solver_y), float(solver_z), float(solver_H), float(solver_Q), float(solver_U), float(solver_sigma_y), float(solver_sigma_z))
            max_C_g_m3 = point_C_g_m3
            max_x_loc = solver_x
            sigma_y_used = solver_sigma_y
            sigma_z_used = solver_sigma_z
            st.info("Since fixed $\\sigma_y$ and $\\sigma_z$ were used, $C_{max}$ is simply the concentration at the point specified, and $x_{max}$ is set to the input $x$ distance.")
        else:
            point_C_g_m3 = calculate_single_point_concentration(float(solver_x), float(solver_y), float(solver_z), float(solver_H), float(solver_Q), float(solver_U), solver_stab_class)
            max_C_g_m3, max_x_loc = find_max_concentration(solver_H, solver_Q, solver_U, solver_stab_class)
            sigma_y_used, sigma_z_used = get_dispersion_coefficients(float(solver_x), solver_stab_class)

        point_C_ug_m3 = point_C_g_m3 * 1e6
        max_C_ug_m3 = max_C_g_m3 * 1e6

        colR1, colR2 = st.columns(2)
        with colR1:
            st.metric(label=f"Concentration at $C({solver_x} m, {solver_y} m, {solver_z} m)$", value=f"{point_C_ug_m3:,.2f} µg/m³")
        with colR2:
            if not use_custom_sigma or (use_custom_sigma and max_x_loc == solver_x):
                st.metric(label="Maximum Ground Concentration ($C_{max}$)", value=f"{max_C_ug_m3:,.2f} µg/m³")

        if not use_custom_sigma:
            st.metric(label="Location of Maximum Ground Concentration ($x_{max}$)", value=f"{max_x_loc:,.1f} m")

        st.markdown(f"""
        **Dispersion Coefficients Used at $x={solver_x} \\,\\text{{m}}$:**
        * $\\sigma_y$: **{sigma_y_used:,.2f} $\\text{{m}}$**
        * $\\sigma_z$: **{sigma_z_used:,.2f} $\\text{{m}}$**
        """)

        st.markdown("---")
        st.subheader("Solver — Interactive Contour (hover to read values, zoom/pan available)")

        x_plot_max = max(2.0 * float(solver_x), 500.0)
        x_plot_min = max(0.1, float(solver_x) * 0.01)
        x_plot = np.linspace(x_plot_min, x_plot_max, 160)

        if use_custom_sigma:
            crosswind_half = max(3.0 * float(solver_sigma_y), 200.0)
        else:
            try:
                est_sigma_y, _ = get_dispersion_coefficients(float(solver_x), solver_stab_class)
            except Exception:
                est_sigma_y = 100.0
            crosswind_half = max(3.0 * est_sigma_y, 200.0)

        y_plot = np.linspace(-crosswind_half, crosswind_half, 120)
        Xp, Yp = np.meshgrid(x_plot, y_plot)
        Cp = np.zeros_like(Xp, dtype=float)

        if use_custom_sigma:
            for ii in range(Xp.shape[0]):
                for jj in range(Xp.shape[1]):
                    xx = float(Xp[ii, jj])
                    yy = float(Yp[ii, jj])
                    Cp[ii, jj] = calculate_point_concentration_custom_sigma(xx, yy, float(solver_z), float(solver_H), float(solver_Q), float(solver_U), float(solver_sigma_y), float(solver_sigma_z))
            plot_mode_label = f"Solver Visualization (Custom σ) at z={solver_z} m"
        else:
            Cp = gaussian_plume_model(Xp, Yp, float(solver_z), float(solver_H), float(solver_Q), float(solver_U), solver_stab_class)
            plot_mode_label = f"Solver Visualization (Pasquill-Gifford: {solver_stab_class}) at z={solver_z} m"

        Cp_ug = Cp * 1e6

        fig_pl = go.Figure(data=go.Contour(z=Cp_ug, x=x_plot, y=y_plot, colorscale='Viridis', contours=dict(showlabels=False), colorbar=dict(title="Concentration (µg m⁻³)"), hovertemplate='x: %{x:.1f} m<br>y: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'))
        fig_pl.add_trace(go.Scatter(x=[0.0], y=[0.0], mode='markers', marker=dict(color='red', size=6), name='Stack (0,0)', hoverinfo='skip'))
        fig_pl.update_layout(title=plot_mode_label, xaxis_title='Downwind distance x (m)', yaxis_title='Crosswind distance y (m)', autosize=True, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_pl, use_container_width=True)

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

    # <-- Added detailed references & acknowledgments block (as in original) -->
    st.markdown(
        """
        ***Primary Source Reference:*** *These notes are adapted from the lecture materials on Gaussian Plumes by E. Savory, available at [eng.uwo.ca/people/esavory/gaussian plumes.pdf](https://www.eng.uwo.ca/people/esavory/gaussian%20plumes.pdf).*

        For further study and reference on Environmental Engineering and dispersion modeling, you may consult the work and class notes of:

        **Dr. Abhradeep Majumder, Ph.D.**
        * Assistant Professor, Department of Civil Engineering, BITS Pilani-Hyderabad Campus
        * Academic Profiles: [Scopus](https://www.scopus.com/authid/detail.uri?authorId=57191504507), [ORCID](https://orcid.org/0000-0002-0186-6450), [Google Scholar](https://scholar.google.co.in/citations?user=mnJ5zdwAAAAJ&hl=en&oi=ao), [LinkedIn](https://linkedin.com/in/abhradeep-majumder-36503777/)

        ---
        ###### Application Development

        This interactive Gaussian Plume Dispersion Model application was developed by **Subodh Purohit** as an educational tool.
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
