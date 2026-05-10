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
# Plume Behaviour Types — Catalogue, Physics & Visualisation
# -------------------------

PLUME_CATALOGUE = [
    dict(key='looping',    name='Looping',    stability='A', model='gaussian',
         badge='#e74c3c', condition='Extremely Unstable',
         sky_wind='Clear sky, strong insolation, light wind (1–3 m/s)',
         time_of_day='Sunny summer afternoon',
         blurb='Large thermal eddies carry the plume steeply up and down. '
               'The Gaussian envelope is very wide vertically. '
               'Worst case for intermittent high ground concentrations.'),
    dict(key='coning',     name='Coning',     stability='C', model='gaussian',
         badge='#e67e22', condition='Slightly Unstable',
         sky_wind='Partly cloudy, moderate wind (4–6 m/s)',
         time_of_day='Cloudy daytime',
         blurb='Symmetric Gaussian spread in y and z — the classic textbook plume. '
               'Moderate ground-level impact at intermediate distances.'),
    dict(key='fanning',    name='Fanning',    stability='F', model='gaussian',
         badge='#2980b9', condition='Moderately Stable',
         sky_wind='Clear night, very light wind (<2 m/s)',
         time_of_day='Clear calm night',
         blurb='Strong stability suppresses vertical mixing. The plume stretches '
               'as a thin horizontal ribbon. Very low ground concentrations near the source.'),
    dict(key='neutral',    name='Neutral',    stability='D', model='gaussian',
         badge='#7f8c8d', condition='Neutral',
         sky_wind='Full overcast or wind > 6 m/s',
         time_of_day='Overcast / high wind',
         blurb='Mechanical turbulence dominates. Moderate symmetric spreading. '
               'The reference condition used in most regulatory dispersion models.'),
    dict(key='lofting',    name='Lofting',    stability='B', model='lofting',
         badge='#27ae60', condition='Unstable above stack, stable below',
         sky_wind='Surface cooling, warm residual layer aloft',
         time_of_day='Late afternoon / early evening',
         blurb='A stable surface layer prevents the plume from mixing downward. '
               'Disperses freely upward. Best condition for ground-level receptors — '
               'surface concentrations near the source are very low.'),
    dict(key='fumigating', name='Fumigating', stability='D', model='fumigating',
         badge='#f39c12', condition='Inversion breaking down, convective layer growing',
         sky_wind='Post-sunrise, sun eroding nocturnal inversion',
         time_of_day='Early morning after sunrise',
         blurb='The overnight stable layer stores an elevated plume. When the growing '
               'convective layer reaches it, pollution is rapidly swept to the ground — '
               'highest short-term ground concentrations of any plume regime.'),
    dict(key='trapping',   name='Trapping',   stability='E', model='trapping',
         badge='#8e44ad', condition='Inversion lid above plume',
         sky_wind='Anticyclone / subsidence inversion',
         time_of_day='Anticyclonic days or nights',
         blurb='An elevated inversion acts as a ceiling at height L. The plume bounces '
               'between ground and lid. Concentrations accumulate — dangerous in valleys '
               'or under stagnant anticyclones.'),
]

_PLUME_KEY_MAP = {p['key']: p for p in PLUME_CATALOGUE}

_TEMP_PROFILES = {
    'A': dict(lapse=-12.5, color='#e74c3c'),
    'B': dict(lapse=-11.0, color='#e67e22'),
    'C': dict(lapse=-10.2, color='#f39c12'),
    'D': dict(lapse=-9.8,  color='#7f8c8d'),
    'E': dict(lapse=-5.0,  color='#3498db'),
    'F': dict(lapse=-1.5,  color='#2980b9'),
}

_TEMP_EXPLANATIONS = {
    'A': ("**Class A — Superadiabatic:** The environment cools faster with height than a rising parcel would. "
          "Displaced parcels keep rising — strong convective eddies produce the chaotic **looping** plume. "
          "Worst case for intermittent peak ground concentrations."),
    'B': ("**Class B — Moderately superadiabatic:** Less convective than Class A. The plume rises and "
          "meanders noticeably but without extreme loops. Moderate-to-high vertical mixing."),
    'C': ("**Class C — Slightly unstable:** Close to neutral with a slight convective tendency. Produces a "
          "regular **coning** plume under broken cloud cover or light sunshine."),
    'D': ("**Class D — Neutral:** Environmental lapse rate equals the DALR exactly. No buoyancy-driven "
          "mixing — mechanical wind shear dominates. Moderate symmetric spreading. The baseline regulatory "
          "reference condition."),
    'E': ("**Class E — Slightly stable:** The environment cools more slowly than a rising parcel. "
          "Displaced parcels return to their original level — vertical mixing suppressed. Plume spreads "
          "mainly laterally with limited vertical growth."),
    'F': ("**Class F — Moderately stable:** Near-isothermal or slight temperature inversion. Vertical "
          "motions strongly suppressed — the **fanning** plume stretches as a thin ribbon, "
          "sometimes persisting for tens of kilometres with little vertical dilution."),
    'lofting': ("**Lofting:** A surface-based stable layer underlies an unstable residual layer at stack height. "
                "Below the stack, vertical mixing is suppressed — the plume cannot reach the ground. "
                "Above, it disperses freely upward. Ground-level concentrations near the source are very low — "
                "the safest short-term condition for surface receptors."),
    'fumigating': ("**Fumigating:** A nocturnal stable layer stored an elevated plume overnight. After sunrise, "
                   "solar heating grows the convective boundary layer upward. When it reaches the plume height, "
                   "the entire stored pollution mass is rapidly mixed to the ground — producing the "
                   "**highest short-term ground concentrations** of any plume regime. Critical for morning "
                   "air quality exceedances near industrial stacks."),
    'trapping': ("**Trapping:** A subsidence or elevated inversion acts as a lid at height L. The plume "
                 "cannot escape upward. As σ_z grows with downwind distance and approaches L, multiple "
                 "reflections off the ground and lid build up concentration in the trapped layer. "
                 "Particularly dangerous in valleys or under slow-moving anticyclones where the lid "
                 "can persist for days."),
}


def _gaussian_lofting_xz(x_arr, z_arr, H, Q, U):
    """X-Z field for lofting: Class B dispersion, no ground-reflection term. Returns (Nz,Nx) g/m³."""
    positive = x_arr > 0
    sigma_y, sigma_z = get_dispersion_coefficients(x_arr, 'B')
    sy = sigma_y[np.newaxis, :]
    sz = sigma_z[np.newaxis, :]
    Zg = z_arr[:, np.newaxis]
    exp_zr = np.exp(-(Zg - float(H)) ** 2 / (2 * sz ** 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(positive[np.newaxis, :], Q / (2 * np.pi * U * sy * sz) * exp_zr, 0.0)
    return C


def _gaussian_trapping_xz(x_arr, z_arr, H, Q, U, stability_class, L, N=5):
    """X-Z field with inversion lid at L: image-source sum. Returns (Nz,Nx) g/m³."""
    positive = x_arr > 0
    sigma_y, sigma_z = get_dispersion_coefficients(x_arr, stability_class)
    sy = sigma_y[np.newaxis, :]
    sz = sigma_z[np.newaxis, :]
    Zg = z_arr[:, np.newaxis]
    vert = np.zeros((len(z_arr), len(x_arr)))
    for n in range(-N, N + 1):
        vert += (np.exp(-(Zg - float(H) - 2 * n * float(L)) ** 2 / (2 * sz ** 2))
                 + np.exp(-(Zg + float(H) - 2 * n * float(L)) ** 2 / (2 * sz ** 2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(positive[np.newaxis, :], Q / (2 * np.pi * U * sy * sz) * vert, 0.0)
    return C


def _gaussian_fumigating_centreline(x_arr, H, Q, U):
    """Ground-level fumigating centreline (y=0). Returns (Nx,) g/m³."""
    positive = x_arr > 0
    sigma_y_D, _ = get_dispersion_coefficients(x_arr, 'D')
    _, sigma_z_F = get_dispersion_coefficients(x_arr, 'F')
    denom = np.sqrt(2 * np.pi) * U * sigma_y_D * np.maximum(float(H) + 2 * sigma_z_F, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(positive, Q / denom, 0.0)
    return C


@st.cache_data
def _pt_xz_field(plume_key, H, Q, U, mixing_height, mix_layer_h):
    """Cached X-Z concentration field at y=0. Returns (x_arr, z_arr, C_ug)."""
    x_arr = np.linspace(10.0, 3000.0, 200)
    z_max = float(max(H * 2.5, mixing_height * 1.2, 400.0))
    z_arr = np.linspace(0.0, z_max, 150)
    p = _PLUME_KEY_MAP[plume_key]

    if p['model'] == 'gaussian':
        positive = x_arr > 0
        sigma_y, sigma_z = get_dispersion_coefficients(x_arr, p['stability'])
        sy = sigma_y[np.newaxis, :]
        sz = sigma_z[np.newaxis, :]
        Zg = z_arr[:, np.newaxis]
        exp_zr = np.exp(-(Zg - float(H)) ** 2 / (2 * sz ** 2))
        exp_zi = np.exp(-(Zg + float(H)) ** 2 / (2 * sz ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive[np.newaxis, :],
                         Q / (2 * np.pi * U * sy * sz) * (exp_zr + exp_zi), 0.0)
    elif p['model'] == 'lofting':
        C = _gaussian_lofting_xz(x_arr, z_arr, H, Q, U)
    elif p['model'] == 'fumigating':
        C_ground = _gaussian_fumigating_centreline(x_arr, H, Q, U)  # (Nx,)
        Z2 = z_arr[:, np.newaxis]  # (Nz,1)
        C = np.where(Z2 <= float(mix_layer_h), C_ground[np.newaxis, :], 0.0)
    elif p['model'] == 'trapping':
        C = _gaussian_trapping_xz(x_arr, z_arr, H, Q, U, p['stability'], mixing_height)
    else:
        C = np.zeros((len(z_arr), len(x_arr)))
    return x_arr, z_arr, C * 1e6


@st.cache_data
def _pt_xy_field(plume_key, H, Q, U, mixing_height, mix_layer_h):
    """Cached X-Y ground-level field. Returns (x_arr, y_arr, C_ug)."""
    x_arr = np.linspace(10.0, 3000.0, 200)
    y_arr = np.linspace(-500.0, 500.0, 150)
    X2, Y2 = np.meshgrid(x_arr, y_arr)
    p = _PLUME_KEY_MAP[plume_key]

    if p['model'] == 'gaussian':
        C = gaussian_plume_model(X2, Y2, 0.0, H, Q, U, p['stability'])
    elif p['model'] == 'lofting':
        positive = X2 > 0
        sigma_y, sigma_z = get_dispersion_coefficients(np.where(positive, X2, 1e-6), 'B')
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive,
                         Q / (2 * np.pi * U * sigma_y * sigma_z)
                         * np.exp(-Y2 ** 2 / (2 * sigma_y ** 2))
                         * np.exp(-float(H) ** 2 / (2 * sigma_z ** 2)), 0.0)
    elif p['model'] == 'fumigating':
        positive = X2 > 0
        sigma_y_D, _ = get_dispersion_coefficients(np.where(positive, X2, 1e-6), 'D')
        _, sigma_z_F = get_dispersion_coefficients(np.where(positive, X2, 1e-6), 'F')
        denom = np.sqrt(2 * np.pi) * U * sigma_y_D * np.maximum(float(H) + 2 * sigma_z_F, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive,
                         Q / denom * np.exp(-Y2 ** 2 / (2 * sigma_y_D ** 2)), 0.0)
    elif p['model'] == 'trapping':
        positive = X2 > 0
        sigma_y, sigma_z = get_dispersion_coefficients(np.where(positive, X2, 1e-6), p['stability'])
        vert = np.zeros_like(X2)
        for n in range(-5, 6):
            vert += (np.exp(-(float(H) + 2 * n * float(mixing_height)) ** 2 / (2 * sigma_z ** 2))
                     + np.exp(-(float(H) - 2 * n * float(mixing_height)) ** 2 / (2 * sigma_z ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive,
                         Q / (2 * np.pi * U * sigma_y * sigma_z)
                         * np.exp(-Y2 ** 2 / (2 * sigma_y ** 2)) * vert, 0.0)
    else:
        C = np.zeros_like(X2)
    return x_arr, y_arr, C * 1e6


@st.cache_data
def _pt_centreline(plume_key, H, Q, U, mixing_height, mix_layer_h):
    """Cached ground-level centreline C(x,0,0). Returns (x_arr, C_ug)."""
    x_arr = np.linspace(10.0, 3000.0, 300)
    positive = x_arr > 0
    p = _PLUME_KEY_MAP[plume_key]

    if p['model'] == 'gaussian':
        sigma_y, sigma_z = get_dispersion_coefficients(x_arr, p['stability'])
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive,
                         Q / (2 * np.pi * U * sigma_y * sigma_z)
                         * 2 * np.exp(-float(H) ** 2 / (2 * sigma_z ** 2)), 0.0)
    elif p['model'] == 'lofting':
        sigma_y, sigma_z = get_dispersion_coefficients(x_arr, 'B')
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive,
                         Q / (2 * np.pi * U * sigma_y * sigma_z)
                         * np.exp(-float(H) ** 2 / (2 * sigma_z ** 2)), 0.0)
    elif p['model'] == 'fumigating':
        C = _gaussian_fumigating_centreline(x_arr, H, Q, U)
    elif p['model'] == 'trapping':
        sigma_y, sigma_z = get_dispersion_coefficients(x_arr, p['stability'])
        vert = np.zeros_like(x_arr)
        for n in range(-5, 6):
            vert += (np.exp(-(float(H) + 2 * n * float(mixing_height)) ** 2 / (2 * sigma_z ** 2))
                     + np.exp(-(float(H) - 2 * n * float(mixing_height)) ** 2 / (2 * sigma_z ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.where(positive,
                         Q / (2 * np.pi * U * sigma_y * sigma_z) * vert, 0.0)
    else:
        C = np.zeros_like(x_arr)
    return x_arr, C * 1e6


@st.cache_data
def _pt_thumbnail(plume_key, mixing_height=300, mix_layer_h=150):
    """
    Schematic line-art thumbnail for gallery cards.
    Each plume type is drawn as a characteristic shape — no model computation.
    Returns PNG bytes.
    """
    BG = '#111827'
    fig, ax = plt.subplots(figsize=(2.8, 1.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = np.linspace(0.0, 1.0, 300)
    H0 = 0.46      # normalised stack height
    GROUND = 0.04  # normalised ground level

    # ── Stack marker (common) ────────────────────────────────────────────────
    ax.plot([0, 0], [GROUND, H0], color='#9ca3af', lw=1.6, solid_capstyle='round')
    ax.plot(0, H0, marker='^', ms=5, color='#ef4444', zorder=6)

    # ── Per-type drawing ─────────────────────────────────────────────────────
    if plume_key == 'looping':
        # Sinusoidal centerline with growing amplitude and width
        amp   = np.linspace(0, 0.20, len(x))
        phase = 4.5 * np.pi * x
        center = H0 + amp * np.sin(phase)
        width  = 0.035 + 0.065 * x
        ax.fill_between(x, center - width, center + width,
                        color='#3b82f6', alpha=0.55, zorder=2)
        ax.plot(x, center + width, color='#93c5fd', lw=0.9, alpha=0.8)
        ax.plot(x, center - width, color='#93c5fd', lw=0.9, alpha=0.8)
        # Extra faint wisps to suggest turbulent mixing
        for ph_off, col_a in [(-0.9, 0.22), (1.1, 0.18), (-2.0, 0.14)]:
            wc = H0 + (amp * 0.55) * np.sin(phase + ph_off)
            ax.plot(x[x > 0.15], wc[x > 0.15], color='#bfdbfe', lw=0.6, alpha=col_a)

    elif plume_key == 'coning':
        spread = 0.24 * x
        ax.fill_between(x, H0 - spread, H0 + spread, color='#f97316', alpha=0.50, zorder=2)
        ax.plot(x, H0 + spread, color='#fb923c', lw=1.2)
        ax.plot(x, H0 - spread, color='#fb923c', lw=1.2)
        # Faint iso-concentration rings
        for frac, a in [(0.5, 0.25), (0.25, 0.15)]:
            s2 = frac * 0.24 * x
            ax.fill_between(x, H0 - s2, H0 + s2, color='#fed7aa', alpha=a, zorder=3)

    elif plume_key == 'fanning':
        spread = 0.025 + 0.018 * x  # very thin
        ax.fill_between(x, H0 - spread, H0 + spread, color='#60a5fa', alpha=0.65, zorder=2)
        ax.plot(x, H0 + spread, color='#93c5fd', lw=1.1)
        ax.plot(x, H0 - spread, color='#93c5fd', lw=1.1)
        # Dashed centreline emphasises how flat the plume is
        ax.plot(x, np.full_like(x, H0), color='white', lw=0.5,
                linestyle='--', alpha=0.30, zorder=4)
        # Ground dashed line (stable at night, no mixing down)
        ax.axhline(GROUND, color='#475569', lw=0.8, linestyle=':', alpha=0.6)

    elif plume_key == 'neutral':
        spread = 0.155 * x
        ax.fill_between(x, H0 - spread, H0 + spread, color='#94a3b8', alpha=0.50, zorder=2)
        ax.plot(x, H0 + spread, color='#cbd5e1', lw=1.2)
        ax.plot(x, H0 - spread, color='#cbd5e1', lw=1.2)
        s2 = 0.07 * x
        ax.fill_between(x, H0 - s2, H0 + s2, color='#e2e8f0', alpha=0.18, zorder=3)

    elif plume_key == 'lofting':
        # Plume fans upward only; tiny downward extent (stable surface layer blocks it)
        upper = H0 + 0.28 * x
        lower = H0 - 0.025 * x     # barely any downward spread
        ax.fill_between(x, lower, upper, color='#34d399', alpha=0.50, zorder=2)
        ax.plot(x, upper, color='#6ee7b7', lw=1.2)
        ax.plot(x, lower, color='#6ee7b7', lw=0.7, linestyle='--')
        # Stable surface layer (shaded band near the ground)
        stable_top = H0 - 0.16
        ax.fill_between(x, GROUND, stable_top,
                        color='#1e3a5f', alpha=0.55, zorder=1)
        ax.plot(x, np.full_like(x, stable_top),
                color='#fbbf24', lw=0.9, linestyle=':', alpha=0.80)

    elif plume_key == 'fumigating':
        mix_h_n = 0.36   # mixing layer in normalised coords
        x_onset = 0.28   # where fumigation kicks in
        # Before fumigation onset: small elevated gaussian
        xb = x[x <= x_onset]
        sb = 0.035 + 0.03 * xb
        ax.fill_between(xb, H0 - sb, H0 + sb, color='#fbbf24', alpha=0.55, zorder=2)
        ax.plot(xb, H0 + sb, color='#fde68a', lw=1.0)
        ax.plot(xb, H0 - sb, color='#fde68a', lw=1.0)
        # After onset: uniform fill from ground to mixing height (well-mixed)
        xa = x[x > x_onset]
        ax.fill_between(xa, GROUND, mix_h_n, color='#f59e0b', alpha=0.48, zorder=2)
        ax.plot(xa, np.full_like(xa, mix_h_n), color='#fcd34d', lw=1.1, linestyle='--')
        # Small downward arrows showing convective mixing
        for xp in [0.42, 0.58, 0.74, 0.90]:
            ax.annotate('', xy=(xp, GROUND + 0.06), xytext=(xp, mix_h_n - 0.04),
                        arrowprops=dict(arrowstyle='->', color='#fde68a',
                                        lw=0.7, mutation_scale=6))
        # Mixing layer label line at onset
        ax.axvline(x_onset, color='#fbbf24', lw=0.7, linestyle=':', alpha=0.5)

    elif plume_key == 'trapping':
        lid_h = 0.80   # inversion lid in normalised coords
        spread = np.minimum(0.24 * x, (lid_h - GROUND) / 2.0)
        upper  = np.minimum(H0 + spread, lid_h - 0.01)
        lower  = np.maximum(H0 - spread, GROUND + 0.01)
        ax.fill_between(x, lower, upper, color='#a78bfa', alpha=0.52, zorder=2)
        ax.plot(x, upper, color='#c4b5fd', lw=0.8)
        ax.plot(x, lower, color='#c4b5fd', lw=0.8)
        # The plume fills the full trapped column far downwind
        far = x > 0.68
        ax.fill_between(x[far], GROUND, lid_h, color='#7c3aed', alpha=0.22, zorder=3)
        # Inversion lid — prominent dashed red line
        ax.axhline(lid_h, color='#f87171', lw=1.6, linestyle='--', alpha=0.90, zorder=5)
        # Ground
        ax.axhline(GROUND, color='#6b7280', lw=0.9, alpha=0.55)

    # ── Shared axis cleanup ──────────────────────────────────────────────────
    ax.set_xlim(-0.03, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout(pad=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=88, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _make_plume_figure(plume_key, view_mode, H, Q, U, mixing_height, mix_layer_h, show_sigma):
    """Return a Plotly figure for the selected plume type and view mode."""
    p = _PLUME_KEY_MAP[plume_key]

    if view_mode == 'X–Z cross-section':
        x_arr, z_arr, C_ug = _pt_xz_field(plume_key, H, Q, U, mixing_height, mix_layer_h)
        fig = go.Figure(data=go.Contour(
            z=C_ug, x=x_arr, y=z_arr, colorscale='Viridis',
            contours=dict(showlabels=False),
            colorbar=dict(title='µg/m³', len=0.75, thickness=12),
            hovertemplate='x: %{x:.0f} m<br>z: %{y:.0f} m<br>C: %{z:.2f} µg/m³<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[H], mode='markers',
            marker=dict(color='red', size=9, symbol='triangle-up'),
            name=f'Stack (H={H} m)'
        ))
        fig.add_shape(type='line', x0=x_arr[0], x1=x_arr[-1], y0=H, y1=H,
                      line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'))
        if show_sigma and p['model'] in ('gaussian', 'lofting'):
            stab_env = 'B' if p['model'] == 'lofting' else p['stability']
            _, sz_e = get_dispersion_coefficients(x_arr, stab_env)
            for mult, alpha, lbl in [(1, 0.55, '±1σz'), (2, 0.30, '±2σz')]:
                hi = np.minimum(float(H) + mult * sz_e, z_arr[-1])
                lo = np.maximum(float(H) - mult * sz_e, 0.0)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_arr, x_arr[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba(255,255,255,{alpha * 0.09})',
                    line=dict(color=f'rgba(255,255,255,{alpha})', width=1, dash='dash'),
                    name=lbl
                ))
        if p['model'] == 'trapping':
            fig.add_shape(type='line', x0=x_arr[0], x1=x_arr[-1],
                          y0=mixing_height, y1=mixing_height,
                          line=dict(color='rgba(255,100,100,0.85)', width=2, dash='dash'))
            fig.add_annotation(x=x_arr[-1] * 0.97, y=mixing_height,
                                text=f'Inversion lid  L = {mixing_height} m',
                                showarrow=False, font=dict(color='#ff6b6b', size=10),
                                xanchor='right', yanchor='bottom')
        if p['model'] == 'fumigating':
            fig.add_shape(type='line', x0=x_arr[0], x1=x_arr[-1],
                          y0=mix_layer_h, y1=mix_layer_h,
                          line=dict(color='rgba(241,196,15,0.8)', width=1.5, dash='dash'))
            fig.add_annotation(x=x_arr[-1] * 0.97, y=mix_layer_h,
                                text=f'Mixing layer  h = {mix_layer_h} m',
                                showarrow=False, font=dict(color='#f1c40f', size=10),
                                xanchor='right', yanchor='bottom')
        fig.update_layout(
            xaxis_title='Downwind distance x (m)', yaxis_title='Height z (m)',
            height=400, margin=dict(l=50, r=120, t=35, b=50),
            legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                        bgcolor='rgba(0,0,0,0.45)', bordercolor='rgba(255,255,255,0.2)',
                        borderwidth=1, font=dict(color='white', size=10))
        )

    elif view_mode == 'X–Y ground level':
        x_arr, y_arr, C_ug = _pt_xy_field(plume_key, H, Q, U, mixing_height, mix_layer_h)
        fig = go.Figure(data=go.Contour(
            z=C_ug, x=x_arr, y=y_arr, colorscale='Viridis',
            contours=dict(showlabels=False),
            colorbar=dict(title='µg/m³', len=0.75, thickness=12),
            hovertemplate='x: %{x:.0f} m<br>y: %{y:.0f} m<br>C: %{z:.2f} µg/m³<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers', marker=dict(color='red', size=8),
            name='Stack (0, 0)', hoverinfo='skip'
        ))
        if show_sigma:
            stab_env = 'B' if p['model'] == 'lofting' else p['stability']
            if p['model'] != 'fumigating':
                sy_e, _ = get_dispersion_coefficients(x_arr, stab_env)
                for mult, alpha, lbl in [(1, 0.70, '±1σy'), (2, 0.45, '±2σy')]:
                    fig.add_trace(go.Scatter(
                        x=x_arr, y=mult * sy_e,
                        line=dict(color=f'rgba(255,255,255,{alpha})', width=1, dash='dash'),
                        name=lbl, showlegend=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_arr, y=-mult * sy_e,
                        line=dict(color=f'rgba(255,255,255,{alpha})', width=1, dash='dash'),
                        name=f'-{mult}σy', showlegend=False
                    ))
        fig.update_layout(
            xaxis_title='Downwind distance x (m)', yaxis_title='Crosswind distance y (m)',
            height=400, margin=dict(l=50, r=120, t=35, b=50),
            legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom',
                        bgcolor='rgba(0,0,0,0.45)', bordercolor='rgba(255,255,255,0.2)',
                        borderwidth=1, font=dict(color='white', size=10))
        )

    else:  # Centreline C vs x
        x_arr, C_ug = _pt_centreline(plume_key, H, Q, U, mixing_height, mix_layer_h)
        peak_idx = int(np.nanargmax(C_ug)) if np.any(C_ug > 0) else 0
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_arr, y=C_ug, mode='lines',
            line=dict(color=p['badge'], width=2.5), name=p['name'],
            hovertemplate='x: %{x:.0f} m<br>C: %{y:.3f} µg/m³<extra></extra>'
        ))
        if C_ug[peak_idx] > 0:
            fig.add_trace(go.Scatter(
                x=[x_arr[peak_idx]], y=[C_ug[peak_idx]],
                mode='markers+text', marker=dict(color='red', size=10),
                text=[f'  C_max = {C_ug[peak_idx]:.1f} µg/m³  at x = {x_arr[peak_idx]:.0f} m'],
                textposition='middle right', showlegend=False
            ))
        fig.update_layout(
            xaxis_title='Downwind distance x (m)',
            yaxis_title='Ground-level concentration (µg/m³)',
            height=400, margin=dict(l=60, r=40, t=35, b=50)
        )
    return fig


def _make_temp_figure(plume_key, H_m, mixing_height):
    """Return (matplotlib_fig, explanation_str) for the temperature profile panel."""
    p = _PLUME_KEY_MAP[plume_key]
    prof_key = p['model'] if p['model'] in ('lofting', 'fumigating', 'trapping') else p['stability']
    explanation = _TEMP_EXPLANATIONS.get(prof_key, '')

    z_temp = np.linspace(0.0, max(float(H_m) * 3.5, float(mixing_height) * 1.35, 700.0), 300)
    T0 = 15.0
    DALR = T0 + z_temp * (-9.8 / 1000.0)

    if prof_key == 'lofting':
        T_env = np.where(z_temp <= float(H_m),
                         T0 + z_temp * (-2.0 / 1000.0),
                         T0 + z_temp * (-12.0 / 1000.0))
        color = '#27ae60'
    elif prof_key == 'fumigating':
        brk = min(float(H_m) * 0.75, 150.0)
        T_env = np.where(z_temp <= brk,
                         T0 + z_temp * (-12.0 / 1000.0),
                         T0 + brk * (-12.0 / 1000.0) + (z_temp - brk) * (4.5 / 1000.0))
        color = '#f39c12'
    elif prof_key == 'trapping':
        T_env = np.where(z_temp <= float(mixing_height),
                         T0 + z_temp * (-2.5 / 1000.0),
                         T0 + float(mixing_height) * (-2.5 / 1000.0)
                         + (z_temp - float(mixing_height)) * (5.5 / 1000.0))
        color = '#8e44ad'
    else:
        pp = _TEMP_PROFILES.get(prof_key, _TEMP_PROFILES['D'])
        T_env = T0 + z_temp * (pp['lapse'] / 1000.0)
        color = pp['color']

    fig, ax = plt.subplots(figsize=(2.8, 4.0))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.fill_betweenx(z_temp, T_env, DALR, where=(T_env <= DALR),
                     color='#e74c3c', alpha=0.13, label='Unstable zone')
    ax.fill_betweenx(z_temp, T_env, DALR, where=(T_env > DALR),
                     color='#3498db', alpha=0.13, label='Stable zone')
    ax.plot(DALR, z_temp, 'k--', lw=1.5, alpha=0.55, label='DALR (−9.8°C/km)')
    ax.plot(T_env, z_temp, color=color, lw=2.5, label='Env. lapse rate')
    ax.axhline(float(H_m), color='red', lw=1.2, linestyle=':', alpha=0.85,
               label=f'H = {int(H_m)} m')
    if prof_key == 'trapping':
        ax.axhline(float(mixing_height), color='#8e44ad', lw=1.5, linestyle='--',
                   alpha=0.9, label=f'Lid  L = {int(mixing_height)} m')
    ax.set_xlabel('Temperature (°C)', fontsize=9)
    ax.set_ylabel('Height (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.5)
    return fig, explanation


def _build_plume_types_tab(Q_g_s, H_m, U_m_s):
    st.subheader("Plume Behaviour Types")
    st.markdown(
        "Explore how atmospheric stability shapes a stack plume. "
        "Select any type from the gallery to load the interactive detail view."
    )

    # Session state
    if 'selected_plume' not in st.session_state:
        st.session_state.selected_plume = 'coning'

    # ── GALLERY ───────────────────────────────────────────────────────────────
    st.markdown("#### Gallery — click to select")
    gcols = st.columns(4)
    for i, p in enumerate(PLUME_CATALOGUE):
        with gcols[i % 4]:
            is_sel = (st.session_state.selected_plume == p['key'])
            border_w = '2.5px' if is_sel else '1px'
            border_c = p['badge'] if is_sel else '#4b5563'
            bg = p['badge'] + '22' if is_sel else 'transparent'
            st.markdown(
                f"""<div style="border:{border_w} solid {border_c};border-radius:10px;
                padding:10px 8px 6px 8px;margin-bottom:4px;background:{bg};
                text-align:center">
                <span style="background:{p['badge']};color:#fff;font-size:10px;
                font-weight:700;padding:2px 8px;border-radius:20px">
                Class {p['stability']}</span>
                <p style="font-size:13px;font-weight:700;margin:7px 0 2px;
                color:var(--color-text-primary)">{p['name']}</p>
                <p style="font-size:10px;color:#9ca3af;margin:0;
                line-height:1.3">{p['time_of_day']}</p>
                </div>""",
                unsafe_allow_html=True
            )
            thumb_bytes = _pt_thumbnail(p['key'])
            st.image(thumb_bytes, use_container_width=True)
            if st.button(
                '✓ Viewing' if is_sel else 'View',
                key=f'gal_{p["key"]}',
                use_container_width=True,
                type='primary' if is_sel else 'secondary'
            ):
                st.session_state.selected_plume = p['key']
                st.rerun()

    st.markdown("---")

    # ── DETAIL VIEW ───────────────────────────────────────────────────────────
    sel = _PLUME_KEY_MAP[st.session_state.selected_plume]

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown(
            f"#### {sel['name']} Plume &nbsp;"
            f"<span style='background:{sel['badge']};color:#fff;font-size:12px;"
            f"padding:3px 11px;border-radius:12px'>Class {sel['stability']} — "
            f"{sel['condition']}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"🕐  {sel['time_of_day']}   ·   🌤  {sel['sky_wind']}")
        st.info(sel['blurb'])
    with hc2:
        compare_neutral = st.checkbox(
            "Compare with Neutral (D)", value=False, key='cmp_neutral'
        )

    # Parameter sliders (seed from sidebar values)
    st.markdown("**Adjust parameters:**")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        pt_H = st.slider("Stack height H (m)", 20, 250, int(H_m), step=10, key='pt_H')
    with sc2:
        pt_U = st.slider("Wind speed U (m/s)", 1.0, 15.0,
                          float(round(U_m_s * 2) / 2), step=0.5, key='pt_U')
    with sc3:
        pt_Q = st.slider("Emission rate Q (g/s)", 10, 500, int(Q_g_s), step=10, key='pt_Q')
    with sc4:
        view_mode = st.radio(
            "View", ['X–Z cross-section', 'X–Y ground level', 'Centreline C vs x'],
            key='pt_view', index=0
        )

    # Model-specific extra controls
    mixing_height = 300
    mix_layer_h = 150
    if sel['model'] == 'trapping':
        mixing_height = st.slider(
            "Inversion lid height L (m)", 50, 800, 300, step=25, key='pt_mixh',
            help="Drag to see concentration build up between the ground and the inversion lid."
        )
    if sel['model'] == 'fumigating':
        mlh_max = max(int(pt_H) - 5, 35)
        mlh_def = max(min(int(pt_H) // 2, 150), 30)
        mlh_def = min(mlh_def, mlh_max)
        mix_layer_h = st.slider(
            "Convective mixing layer height (m)", 30, mlh_max, mlh_def, step=5, key='pt_mlh',
            help="Height of the growing convective layer below the stable inversion."
        )

    tg1, tg2, _ = st.columns([1, 1, 2])
    with tg1:
        show_sigma = st.checkbox("Show σ envelope", value=True, key='pt_sig')
    with tg2:
        show_temp = st.checkbox("Show temperature profile", value=True, key='pt_temp')

    # Main plot(s)
    if compare_neutral and st.session_state.selected_plume != 'neutral':
        lc, rc = st.columns(2)
        with lc:
            st.markdown(f"**{sel['name']} — Class {sel['stability']}**")
            st.plotly_chart(
                _make_plume_figure(st.session_state.selected_plume, view_mode,
                                   pt_H, pt_Q, pt_U, mixing_height, mix_layer_h, show_sigma),
                use_container_width=True
            )
        with rc:
            st.markdown("**Neutral — Class D (reference)**")
            st.plotly_chart(
                _make_plume_figure('neutral', view_mode, pt_H, pt_Q, pt_U,
                                   mixing_height, mix_layer_h, show_sigma),
                use_container_width=True
            )
    else:
        st.plotly_chart(
            _make_plume_figure(st.session_state.selected_plume, view_mode,
                               pt_H, pt_Q, pt_U, mixing_height, mix_layer_h, show_sigma),
            use_container_width=True
        )

    # Temperature profile
    if show_temp:
        st.markdown("---")
        st.markdown("**Atmospheric temperature profile — why this plume shape occurs**")
        tp_col, exp_col = st.columns([1, 2])
        with tp_col:
            t_fig, t_exp = _make_temp_figure(
                st.session_state.selected_plume, pt_H, mixing_height
            )
            st.pyplot(t_fig, use_container_width=True)
            plt.close(t_fig)
        with exp_col:
            st.markdown(t_exp)


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

    # Tab order: Plume Visualizer, 3D Visualization, Problem Solver, Plume Behaviour Types, Theory & Assumptions
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plume Visualizer", "3D Visualization", "Problem Solver", "Plume Behaviour Types", "Theory & Assumptions"])

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

    # --- Tab 4: Plume Behaviour Types ---
    with tab4:
        _build_plume_types_tab(Q_g_s, H_m, U_m_s)

    # --- Tab 5: Theory & Assumptions ---
    with tab5:
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
            autosize=True,
            margin=dict(l=40, r=120, t=50, b=40),
            legend=dict(
                x=0.01,
                y=0.01,
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(0,0,0,0.45)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1,
                font=dict(color='white', size=11),
            )
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
