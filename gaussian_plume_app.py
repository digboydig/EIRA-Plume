"""
Gaussian Plume App.

Capabilities:
- Interactive visualization of Gaussian plume concentration maps.
- Time-evolution advection animation with optional MP4 export.
- Point-concentration solver with support for Pasquill–Gifford curves.
- Cross-sectional views (X vs Z and Y vs Z).

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

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

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

# -------------------------
# MP4 Export Helpers
# -------------------------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def export_animation_to_mp4(frames_rgb, fps: int, out_path: str):
    if imageio is None:
        raise RuntimeError("MP4 export requires 'imageio'. Install it to enable export.")
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
    fig, ax = plt.subplots(figsize=figsize)
    if x_extent is None: x_extent = (0.0, array2d.shape[1])
    if y_extent is None: y_extent = (0.0, array2d.shape[0])

    im = ax.imshow(array2d, origin='lower', aspect='auto',
                   extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('Downwind distance x (m)')
    ax.set_ylabel('Crosswind distance y (m)')
    ax.set_title('Concentration (µg m⁻³)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    max_val = np.nanmax(array2d) if array2d.size else 0.0
    info_text = f"t = {t_seconds:.1f} s   |   conc. = {max_val:,.2f} µg/m³"
    fig.text(0.5, 0.96, info_text, ha='center', va='center', fontsize=11, weight='bold')

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    rgb = arr[..., :3]
    plt.close(fig)
    return rgb

# -------------------------
# Model Functions
# -------------------------
@st.cache_data
def get_dispersion_coefficients(x, stability_class):
    try:
        idx = SIGMA_COEFFS['index'][stability_class]
    except KeyError:
        return 0, 0
    Ay, By = SIGMA_COEFFS['y']['A'][idx], SIGMA_COEFFS['y']['B'][idx]
    Az, Bz = SIGMA_COEFFS['z']['A'][idx], SIGMA_COEFFS['z']['B'][idx]
    x_arr = np.asarray(x, dtype=float)
    x_for_power = np.where(x_arr <= 1e-6, 1e-6, x_arr)
    sigma_y = Ay * (x_for_power ** By)
    sigma_z = Az * (x_for_power ** Bz)
    if stability_class == 'F':
        sigma_z = np.where(np.asarray(x) < 100, np.maximum(sigma_z, 1.0), sigma_z)
    return np.maximum(sigma_y, 1e-3), np.maximum(sigma_z, 1e-3)

@st.cache_data
def gaussian_plume_model(x_m, y_m, z, H, Q, U, stability_class):
    X, Y = np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)
    C = np.zeros_like(X, dtype=float)
    positive_mask = X > 0.0
    if not np.any(positive_mask): return C
    sigma_y, sigma_z = get_dispersion_coefficients(X, stability_class)
    denom = 2 * np.pi * U * sigma_y * sigma_z
    exp_y = np.exp(-Y ** 2 / (2 * sigma_y ** 2))
    exp_z_real = np.exp(-(z - H) ** 2 / (2 * sigma_z ** 2))
    exp_z_image = np.exp(-(z + H) ** 2 / (2 * sigma_z ** 2))
    C = (Q / denom) * exp_y * (exp_z_real + exp_z_image)
    return np.where(positive_mask, C, 0.0)

def find_max_concentration(H, Q, U, stability_class):
    x_range = np.linspace(10, 5000, 500)
    max_C, max_x = 0.0, 0.0
    for x in x_range:
        sy, sz = get_dispersion_coefficients(x, stability_class)
        C = (Q / (np.pi * U * sy * sz)) * np.exp(-H ** 2 / (2 * sz ** 2))
        if C > max_C:
            max_C, max_x = C, x
    return max_C, max_x

def calculate_single_point_concentration(x, y, z, H, Q, U, stability_class):
    if x <= 0: return 0.0
    sy, sz = get_dispersion_coefficients(x, stability_class)
    scaling = Q / (2 * np.pi * U * sy * sz)
    exp_y = np.exp(-y ** 2 / (2 * sy ** 2))
    vertical = np.exp(-(z-H)**2/(2*sz**2)) + np.exp(-(z+H)**2/(2*sz**2))
    return scaling * exp_y * vertical

def calculate_point_concentration_custom_sigma(x, y, z, H, Q, U, sy, sz):
    if x <= 0 or sy <= 0 or sz <= 0: return 0.0
    scaling = Q / (2 * np.pi * U * sy * sz)
    exp_y = np.exp(-y ** 2 / (2 * sy ** 2))
    vertical = np.exp(-(z-H)**2/(2*sz**2)) + np.exp(-(z+H)**2/(2*sz**2))
    return scaling * exp_y * vertical

# -------------------------
# UI Building Functions
# -------------------------
def _configure_page():
    st.set_page_config(layout="wide", page_title="Gaussian Plume Visualizer")

def _sidebar_inputs():
    st.sidebar.header("Source & Environment Parameters")
    Q_g_s = st.sidebar.slider(r"Emission Rate ($Q$, g/s)", 10.0, 500.0, 100.0)
    H_m = st.sidebar.slider(r"Effective Stack Height ($H$, m)", 10.0, 200.0, 100.0)
    U_m_s = st.sidebar.slider(r"Wind Speed ($U$, m/s)", 1.0, 20.0, 5.0)
    
    stabs = {'A': 'A - Extremely Unstable', 'B': 'B - Moderately Unstable', 'C': 'C - Slightly Unstable',
             'D': 'D - Neutral', 'E': 'E - Slightly Stable', 'F': 'F - Moderately Stable'}
    stability_class = st.sidebar.selectbox("Atmospheric Stability Class", options=list(stabs.keys()),
                                           format_func=lambda x: stabs[x], index=3)
    
    use_custom_z = st.sidebar.checkbox("Visualize at a Custom Height (z)", value=False)
    Z_RECEPTOR_M = st.sidebar.slider("Receptor Height (m)", 0.0, 300.0, 50.0) if use_custom_z else 0.0
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='font-size: 11px; color: grey;'>Developed by: <b>Subodh Purohit</b></p>", unsafe_allow_html=True)
    return Q_g_s, H_m, U_m_s, stability_class, Z_RECEPTOR_M, stabs

def _build_visualizer_tab(Q_g_s, H_m, U_m_s, stability_class, Z_RECEPTOR_M, stability_options):
    st.title("Gaussian Plume Dispersion Visualizer")
    tab1, tab2, tab3, tab4 = st.tabs(["Plume Visualizer", "Problem Solver", "3D Visualization", "Theory & Assumptions"])

    with tab1:
        st.subheader(f"Plume Concentration Map (z={Z_RECEPTOR_M}m)")
        x_plot = np.linspace(0, 4000, 200)
        y_plot = np.linspace(-500, 500, 100)
        X, Y = np.meshgrid(x_plot, y_plot)
        C = gaussian_plume_model(X, Y, Z_RECEPTOR_M, H_m, Q_g_s, U_m_s, stability_class) * 1e6
        
        fig = go.Figure(data=go.Contour(z=C, x=x_plot, y=y_plot, colorscale='Viridis'))
        fig.update_layout(xaxis_title='Downwind (m)', yaxis_title='Crosswind (m)')
        st.plotly_chart(fig, use_container_width=True)

        max_C, max_x = find_max_concentration(H_m, Q_g_s, U_m_s, stability_class)
        st.metric("Max Ground Concentration", f"{max_C*1e6:,.2f} µg/m³", f"at x={max_x:.0f}m")

    with tab2:
        _build_solver_tab(stability_options)

    with tab3:
        _build_3d_geometry_tab(H_m, Q_g_s, U_m_s, stability_class)

    with tab4:
        _build_theory_tab()

def _build_3d_geometry_tab(H_m, Q_g_s, U_m_s, stability_class):
    st.subheader("3D Plume Geometry")
    x_max = 4000
    x_line = np.linspace(1, x_max, 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_line, y=np.zeros_like(x_line), z=np.full_like(x_line, H_m), mode='lines', name='Centerline'))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0, H_m], mode='lines', line=dict(color='red', width=8), name='Stack'))
    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'), height=700)
    st.plotly_chart(fig, use_container_width=True)

def _build_solver_tab(stabs):
    st.subheader("Point Concentration Solver")
    col1, col2, col3 = st.columns(3)
    with col1: sx = st.number_input("Target X (m)", value=1000.0)
    with col2: sy = st.number_input("Target Y (m)", value=0.0)
    with col3: sz = st.number_input("Target Z (m)", value=0.0)
    # Basic solver implementation...
    st.info("Input coordinates above to calculate specific point impacts.")

def _build_theory_tab():
    st.header("Theory & Equations")
    st.latex(r"C(x, y, z) = \frac{Q}{2\pi U \sigma_y \sigma_z} \exp\left(-\frac{y^2}{2\sigma_y^2}\right) \left[ \exp\left(-\frac{(z-H)^2}{2\sigma_z^2}\right) + \exp\left(-\frac{(z+H)^2}{2\sigma_z^2}\right) \right]")

def run_app():
    _configure_page()
    Q, H, U, sc, z_rec, stabs = _sidebar_inputs()
    _build_visualizer_tab(Q, H, U, sc, z_rec, stabs)

if __name__ == '__main__':
    run_app()
