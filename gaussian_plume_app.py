import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 1. CONFIGURATION AND MODEL PARAMETERS ---

# Dispersion coefficient formulas (sigma_y and sigma_z)
# These are the Pasquill-Gifford curves, approximated by power laws (A*x^B)
# x is the downwind distance in meters
# Stability classes: A (Extremely Unstable) to F (Extremely Stable)
# Coefficients are for 10^2 < x < 10^5 meters.

# Format: [A, B] for sigma_y and sigma_z
SIGMA_COEFFS = {
    # Sigma_y coefficients (Lateral Dispersion)
    'y': {
        'A': [0.22, 0.16, 0.11, 0.08, 0.06, 0.04],  # A to F
        'B': [0.90, 0.90, 0.90, 0.90, 0.90, 0.90]  # Simplified, constant exponent
    },
    # Sigma_z coefficients (Vertical Dispersion)
    'z': {
        'A': [0.20, 0.12, 0.08, 0.06, 0.03, 0.016], # A to F
        'B': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]  # Simplified, constant exponent
    },
    # Mapping stability class to index for coefficient lookup
    'index': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
}

# --- 2. MODEL FUNCTIONS (MODIFIED) ---

@st.cache_data
def get_dispersion_coefficients(x, stability_class):
    """Calculates sigma_y and sigma_z based on downwind distance and stability class."""
    
    # Get the index (0-5) for the stability class
    try:
        idx = SIGMA_COEFFS['index'][stability_class]
    except KeyError:
        return 0, 0

    # Retrieve coefficients
    Ay = SIGMA_COEFFS['y']['A'][idx]
    By = SIGMA_COEFFS['y']['B'][idx]
    
    Az = SIGMA_COEFFS['z']['A'][idx]
    Bz = SIGMA_COEFFS['z']['B'][idx]

    # Calculate sigma values (using a simplified power law approximation)
    sigma_y = Ay * (x**By)
    sigma_z = Az * (x**Bz)
    
    # Simple adjustment for stable conditions (F) to prevent extremely small sigma_z at short distances
    if stability_class == 'F' and x < 100:
        sigma_z = max(sigma_z, 1.0) # Ensure some initial mixing

    return sigma_y, sigma_z

@st.cache_data
# MODIFIED FUNCTION SIGNATURE AND LOGIC
def gaussian_plume_model(x_m, y_m, z, H, Q, U, stability_class):
    """
    Calculates the concentration C(x, y, z) for a 2D slice (meshgrid) at a fixed height z.
    If z=0, this is the ground-level concentration.
    """
    C = np.zeros_like(x_m, dtype=float)
    
    for i in range(x_m.shape[0]):
        for j in range(x_m.shape[1]):
            x = x_m[i, j]
            y = y_m[i, j]
            
            # Avoid division by zero at x=0
            if x <= 0:
                C[i, j] = 0.0
                continue
                
            sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)
            
            if sigma_y == 0 or sigma_z == 0:
                C[i, j] = 0.0
                continue
                
            # 1. Crosswind (y) term
            exp_y = np.exp(-y**2 / (2 * sigma_y**2))
            
            # 2. Vertical (z) term (real source + virtual image source)
            # Full term: exp(-(z-H)^2 / (2*sigma_z^2)) + exp(-(z+H)^2 / (2*sigma_z^2))
            exp_z_real = np.exp(-(z - H)**2 / (2 * sigma_z**2))
            exp_z_image = np.exp(-(z + H)**2 / (2 * sigma_z**2))
            
            vertical_term = exp_z_real + exp_z_image
            
            # 3. Scaling Factor
            # The full equation has 2*pi in the denominator
            scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
            
            # Total concentration
            C[i, j] = scaling_factor * exp_y * vertical_term
            
    return C

def find_max_concentration(H, Q, U, stability_class):
    """Finds the maximum ground-level concentration and its downwind distance (x) on the center-line (y=0, z=0)."""
    
    # Search range for x (downwind distance)
    x_range = np.linspace(10, 5000, 500) # Search from 10m to 5000m
    
    max_C = 0.0
    max_x = 0.0
    
    # Use the full equation with z=0 for robustness
    z = 0.0
    
    for x in x_range:
        sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)
        
        # Avoid division by zero
        if sigma_y == 0 or sigma_z == 0:
            continue
            
        # Simplified C(x, 0, 0) logic from the full equation (H^2 term)
        # Vertical Term at z=0: exp(-(-H)^2 / (2*sigma_z^2)) + exp(-(H)^2 / (2*sigma_z^2)) = 2 * exp(-H^2 / (2*sigma_z^2))
        
        vertical_term = 2.0 * np.exp(-H**2 / (2 * sigma_z**2))
        
        # Scaling Factor: Q / (2 * np.pi * U * sigma_y * sigma_z)
        scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
        
        # Total concentration (C(x,0,0))
        C_centerline = scaling_factor * vertical_term
        
        if C_centerline > max_C:
            max_C = C_centerline
            max_x = x
            
    return max_C, max_x

def calculate_single_point_concentration(x, y, z, H, Q, U, stability_class):
    """Calculates the concentration C(x, y, z) at a single specified point (scalar inputs). (Renamed from any_point_concentration)"""
    if x <= 0:
        return 0.0

    sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)

    if sigma_y == 0 or sigma_z == 0:
        return 0.0

    # 1. Crosswind (y) term
    exp_y = np.exp(-y**2 / (2 * sigma_y**2))
    
    # 2. Vertical (z) term (real source + virtual image source)
    exp_z_real = np.exp(-(z - H)**2 / (2 * sigma_z**2))
    exp_z_image = np.exp(-(z + H)**2 / (2 * sigma_z**2))
    vertical_term = exp_z_real + exp_z_image
    
    # 3. Scaling Factor
    scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
    
    # Total concentration (g/m^3)
    C = scaling_factor * exp_y * vertical_term
    
    return C

# --- NEW HELPER FUNCTIONS FOR SOLVER TAB (Uses custom sigma values) ---

def calculate_point_concentration_custom_sigma(x, y, z, H, Q, U, sigma_y, sigma_z):
    """Calculates concentration using explicitly provided sigma values."""
    if x <= 0 or sigma_y <= 0 or sigma_z <= 0:
        return 0.0
        
    # 1. Crosswind (y) term
    exp_y = np.exp(-y**2 / (2 * sigma_y**2))
    
    # 2. Vertical (z) term (real source + virtual image source)
    exp_z_real = np.exp(-(z - H)**2 / (2 * sigma_z**2))
    exp_z_image = np.exp(-(z + H)**2 / (2 * sigma_z**2))
    vertical_term = exp_z_real + exp_z_image
    
    # 3. Scaling Factor
    scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
    
    # Total concentration (g/m^3)
    C = scaling_factor * exp_y * vertical_term
    
    return C

def find_max_concentration_custom_sigma_fixed_ratio(H, Q, U, stability_class):
    """
    Finds the max ground-level concentration.
    NOTE: We must still rely on the original find_max_concentration as the max_x 
    calculation depends on the x-dependent Pasquill-Gifford curves, which are 
    complex to invert for arbitrary sigma values. If custom sigmas are used, 
    we only calculate the point concentration, not x_max.
    """
    # Fallback to the original function to get the max location if needed for comparison
    return find_max_concentration(H, Q, U, stability_class)
    
# --- 3. STREAMLIT APP LAYOUT (MAIN PAGE) ---

st.set_page_config(layout="wide", page_title="Gaussian Plume Visualizer")

st.title("Gaussian Plume Dispersion Visualizer")
st.markdown("An interactive model to explore how source parameters and atmospheric stability affect pollutant spread and ground-level concentration.")

# --- SIDEBAR (GLOBAL USER INPUTS) ---

st.sidebar.header("Source & Environment Parameters")

# 1. Emission Parameters
st.sidebar.subheader("1. Source Strength")
st.sidebar.markdown(r"Emission Rate ($Q$, $\text{g/s}$)") # Use markdown for correct LaTeX rendering
Q_g_s = st.sidebar.slider("", 10.0, 500.0, 100.0, step=10.0, key='Q_slider')
st.sidebar.markdown(r"Effective Stack Height ($H$, $\text{m}$)")
H_m = st.sidebar.slider("", 10.0, 200.0, 100.0, step=5.0, key='H_slider')

# 2. Atmospheric Conditions
st.sidebar.subheader("2. Atmospheric Conditions")
st.sidebar.markdown(r"Wind Speed ($U$, $\text{m/s}$)")
U_m_s = st.sidebar.slider("", 1.0, 20.0, 5.0, step=0.5, key='U_slider')

stability_options = {'A': 'A - Extremely Unstable', 'B': 'B - Moderately Unstable', 'C': 'C - Slightly Unstable',
                     'D': 'D - Neutral (Overcast/High Wind)', 'E': 'E - Slightly Stable', 'F': 'F - Moderately Stable'}
stability_class = st.sidebar.selectbox("Atmospheric Stability Class", options=list(stability_options.keys()), format_func=lambda x: stability_options[x], index=3, key='stability_slider')

# 3. VISUALIZATION TOGGLE (NEW FEATURE)
st.sidebar.subheader("3. Visualization Plane (z)")
use_custom_z = st.sidebar.checkbox(r"Visualize at a Custom Height ($z$)", value=False, help="Toggle to view concentration on a horizontal plane above the ground.")

# Conditional slider for z
if use_custom_z:
    # Set a maximum reasonable height, maybe up to 1.5 times the stack height
    max_z_limit = max(150.0, H_m * 1.5)
    Z_RECEPTOR_M = st.sidebar.slider(r"Receptor Plane Height ($z$, $\text{m}$)", 0.0, max_z_limit, 50.0, step=5.0)
else:
    Z_RECEPTOR_M = 0.0 # Force z=0 when the checkbox is off (standard ground-level)

st.sidebar.markdown(
    """
    <small>
    The Stability Class determines the rate of atmospheric mixing.
    </small>
    """, unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size: 11px; color: grey;">Developed by: <b>Subodh Purohit</b></p>', unsafe_allow_html=True)

# --- TABS CONTAINER ---
tab1, tab2, tab3 = st.tabs(["Plume Visualizer", "Problem Solver", "Theory & Assumptions"])

# --- TAB 1: PLUME VISUALIZER (single-column layout) ---
with tab1:
    
    if Z_RECEPTOR_M > 0:
        st.subheader(f"Plume Concentration Map at Receptor Height $z = {Z_RECEPTOR_M}$ m")
        st.warning(f"NOTE: The peak concentration at ground level ($z=0$) may be higher or located at a different distance than shown on this $z={Z_RECEPTOR_M}$ m plane.")
    else:
        st.subheader("Plume Concentration Contour Map (Ground Level, $z=0$)")


    # --- Visualization domain ---
    X_MAX = 4000  # meters downwind
    Y_MAX = 500   # meters crosswind (half-width, total 1000m)

    # Create meshgrid for X and Y coordinates
    x_range = np.linspace(0.0, X_MAX, 200)
    y_range = np.linspace(-Y_MAX, Y_MAX, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate concentrations (C in g/m^3) using the new function structure
    C_values = gaussian_plume_model(X, Y, Z_RECEPTOR_M, H_m, Q_g_s, U_m_s, stability_class)

    # Convert to micrograms per cubic meter (μg/m^3) for display
    C_ug_m3 = C_values * 1e6

    # Plot (or warn if near-zero)
    if np.nanmax(C_ug_m3) > 1e-9:
        # -----------------------
        # Interactive Plotly contour (Tab 1)
        # -----------------------
        x_plot = x_range
        y_plot = y_range
        z_plot = C_ug_m3  # shape matches meshgrid (ny, nx)

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

        # Add stack marker as scatter
        fig_tab1.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode='markers',
                marker=dict(color='red', size=7),
                name='Stack (0,0)',
                hoverinfo='skip'
            )
        )

        plot_title = f'Concentration Map at z={Z_RECEPTOR_M} m (Stability: {stability_class}, H: {H_m} m)'
        fig_tab1.update_layout(
            title=plot_title,
            xaxis_title='Downwind Distance (x, m)',
            yaxis_title='Crosswind Distance (y, m)',
            autosize=True,
            margin=dict(l=40, r=20, t=50, b=40)
        )

        st.plotly_chart(fig_tab1, use_container_width=True)

        # --- ADDITIONAL CROSS-SECTIONS (X vs Z, Y vs Z) ---
        st.markdown("---")
        st.subheader("Additional Cross-Sectional Views")

        cross_section_option = st.selectbox(
            "Choose additional cross-section to view:",
            ("None", "X vs Z (downwind vs height at y=0)", "Y vs Z (crosswind vs height at chosen x)"),
            index=0,
            key="tab1_cross_section"
        )

        # Compute x_max for default chooser
        max_C_g_m3_loc, max_x_loc = find_max_concentration(H_m, Q_g_s, U_m_s, stability_class)

        if cross_section_option == "X vs Z (downwind vs height at y=0)":
            # Build x vs z grid at y=0
            x_xvz = x_plot  # use same downwind resolution
            # choose reasonable height range: up to max(1.5*H, 300) m
            z_max_plot = max(1.5 * H_m, 300.0)
            z_xvz = np.linspace(0.0, z_max_plot, 120)
            X_xvz, Z_xvz = np.meshgrid(x_xvz, z_xvz)

            # Compute concentrations at y=0 for each (x,z)
            Cxz = np.zeros_like(X_xvz, dtype=float)
            # loop over grid (vectorization could be used but this is simple and consistent)
            for ii in range(X_xvz.shape[0]):
                for jj in range(X_xvz.shape[1]):
                    xv = float(X_xvz[ii, jj])
                    zv = float(Z_xvz[ii, jj])
                    Cxz[ii, jj] = calculate_single_point_concentration(xv, 0.0, zv, H_m, Q_g_s, U_m_s, stability_class)

            Cxz_ug = Cxz * 1e6

            # Plotly contour x vs z
            fig_xvz = go.Figure(
                data=go.Contour(
                    z=Cxz_ug,
                    x=x_xvz,
                    y=z_xvz,
                    colorscale='Viridis',
                    contours=dict(showlabels=False),
                    colorbar=dict(title="Concentration (µg m⁻³)"),
                    hovertemplate='x: %{x:.1f} m<br>z: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'
                )
            )
            fig_xvz.update_layout(
                title=f"X vs Z (y=0) — Downwind vs Height",
                xaxis_title='Downwind distance x (m)',
                yaxis_title='Height z (m)',
                autosize=True,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            # show marker for stack height H at x=0
            fig_xvz.add_trace(go.Scatter(x=[0.0], y=[H_m], mode='markers', marker=dict(color='red', size=7), name='Stack H'))

            st.plotly_chart(fig_xvz, use_container_width=True)

            st.markdown(
                """
                **Significance:** This plot shows how concentration changes with downwind distance and elevation along the plume centerline (y=0). It helps assess plume rise and where elevated receptors (e.g. rooftops) may intersect high concentration zones.
                """, unsafe_allow_html=True
            )

        elif cross_section_option == "Y vs Z (crosswind vs height at chosen x)":
            # let user choose x location for the vertical cross-section
            # default to x of maximum ground concentration if present, else midpoint
            default_x_for_section = float(max_x_loc) if max_x_loc > 0 else float(X_MAX / 2.0)
            x_chosen = st.slider("Choose downwind distance for Y vs Z slice (x, m)", min_value=1.0, max_value=float(X_MAX), value=float(default_x_for_section), step=10.0, key="tab1_yvz_xslider")

            # build y vs z grid
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

            fig_yvz = go.Figure(
                data=go.Contour(
                    z=Cyz_ug,
                    x=y_yvz,
                    y=z_yvz,
                    colorscale='Viridis',
                    contours=dict(showlabels=False),
                    colorbar=dict(title="Concentration (µg m⁻³)"),
                    hovertemplate='y: %{x:.1f} m<br>z: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'
                )
            )
            fig_yvz.update_layout(
                title=f"Y vs Z at x = {x_chosen:.0f} m — Crosswind vs Height",
                xaxis_title='Crosswind distance y (m)',
                yaxis_title='Height z (m)',
                autosize=True,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            # mark centerline y=0
            fig_yvz.add_trace(go.Scatter(x=[0.0], y=[H_m], mode='markers', marker=dict(color='red', size=7), name='Stack H (projection)'))
            st.plotly_chart(fig_yvz, use_container_width=True)

            st.markdown(
                """
                **Significance:** This vertical cross-section at a chosen downwind distance shows lateral and vertical dispersion at that location — useful to check ground-level exposure and how concentrations vary with height across the plume.
                """, unsafe_allow_html=True
            )

    else:
        st.warning(f"The calculated concentration at $z={Z_RECEPTOR_M}$ m is near zero. The plume may be passing above or below this height, or parameters result in high dilution.")

    # --- Key model findings (display below the plot) ---
    st.subheader("Key Model Findings (Ground-Level)")
    st.markdown("*(These metrics are always for the maximum ground-level concentration, $\mathbf{C(x, y, 0)}$)*")

    # 1. Max ground concentration and location
    max_C_g_m3, max_x = find_max_concentration(H_m, Q_g_s, U_m_s, stability_class)
    max_C_ug_m3 = max_C_g_m3 * 1e6

    # small row of metrics
    mcol1, mcol2, mcol3 = st.columns([1.2, 1.0, 1.2])
    with mcol1:
        # FINAL FIX: Simplified Mathtext notation for st.metric label
        st.metric(label="Max Ground Conc. (center-line)", value=f"{max_C_ug_m3:,.2f} µg/m³")
    with mcol2:
        st.metric(label=r"$x_{max}$ (downwind)", value=f"{max_x:,.0f} m")
    # compute plume dims if available
    if max_x > 0:
        sigma_y_max_x, sigma_z_max_x = get_dispersion_coefficients(max_x, stability_class)
        with mcol3:
            st.metric(label="Plume Half-Width ($2\sigma_y$ at $x_{max}$)", value=f"{2 * sigma_y_max_x:,.1f} m")
    else:
        with mcol3:
            st.metric(label="Plume Half-Width ($2\sigma_y$)", value="N/A")

    # Additional plume height info and short explanation
    if max_x > 0:
        st.markdown(f"**Plume mixing height estimate ($4.3\\sigma_z$) at $x = {int(max_x)}$ m:** ${4.3 * sigma_z_max_x:,.1f} \\text{{ m}}$")
    else:
        st.info("Plume maximum could not be calculated. Ensure H is not too large or Q is not too small.")

# ----------------------------------------------------------------------------------------------------------------------
# --- TAB 2: PROBLEM SOLVER ---
with tab2:
    st.subheader("Point Concentration & $x_{max}$ Solver")
    st.markdown("Calculate concentrations and the maximum ground-level location using **custom parameters** independent of the visualizer's sidebar.")

    # 1. Dispersion Mode Selection
    st.subheader("1. Dispersion Mode")
    dispersion_mode = st.radio(
        "Choose Dispersion Coefficient Source:",
        ('Pasquill-Gifford Curves (default)', 'Custom $\\sigma_y$ and $\\sigma_z$ Input'),
        index=0,
        key='dispersion_mode'
    )
    
    # Custom Sigma Inputs
    use_custom_sigma = (dispersion_mode == 'Custom $\\sigma_y$ and $\\sigma_z$ Input')

    if use_custom_sigma:
        st.warning("When using custom $\\sigma$ values, the maximum concentration ($x_{max}$) calculation is not meaningful as $\\sigma$ is fixed, not distance-dependent.")
        colS1, colS2 = st.columns(2)
        with colS1:
            solver_sigma_y = st.number_input("Custom $\\sigma_y$ (Lateral, m)", min_value=0.1, value=100.0, step=10.0, key='sigma_y_solver')
        with colS2:
            solver_sigma_z = st.number_input("Custom $\\sigma_z$ (Vertical, m)", min_value=0.1, value=50.0, step=5.0, key='sigma_z_solver')
    
    # 2. Source and Atmosphere Parameters
    st.subheader("2. Source & Atmospheric Parameters")
    colA, colB, colC = st.columns(3)
    with colA:
        solver_Q = st.number_input("Emission Rate ($Q$, g/s)", min_value=1.0, value=100.0, step=10.0, key='Q_solver')
    with colB:
        solver_H = st.number_input("Effective Stack Height ($H$, m)", min_value=1.0, value=100.0, step=5.0, key='H_solver')
    with colC:
        solver_U = st.number_input("Wind Speed ($U$, m/s)", min_value=0.1, value=5.0, step=0.5, key='U_solver')

    # Stability class is only needed if using Pasquill-Gifford curves
    if not use_custom_sigma:
        solver_stab_class = st.selectbox(
            "4. Atmospheric Stability Class",
            options=list(stability_options.keys()),
            format_func=lambda x: stability_options[x],
            index=3,
            key='stability_solver_key'
        )
    else:
        # Placeholder for stability class if custom sigma is used (required for original x_max finder)
        solver_stab_class = 'D' # Default to D if custom is selected

    st.markdown("---")
    st.subheader("3. Point Location Input")
    colX, colY, colZ = st.columns(3)
    with colX:
        solver_x = st.number_input("Downwind Distance ($x$, m)", min_value=1.0, value=1000.0, step=10.0, key='x_input_solver')
    with colY:
        solver_y = st.number_input("Crosswind Distance ($y$, m)", value=0.0, step=10.0, key='y_input_solver')
    with colZ:
        # Calls the generalized function `calculate_single_point_concentration`
        solver_z = st.number_input("Vertical Height ($z$, m)", value=0.0, step=10.0, key='z_input_solver')

    # Run calculations when button pressed
    if st.button("Run Calculations for Custom Parameters", key='solve_button'):
        st.subheader("Calculated Results")
        
        # --- CALCULATION LOGIC BASED ON DISPERSION MODE ---
        if use_custom_sigma:
            # Mode 1: Use Custom Sigma Values
            point_C_g_m3 = calculate_point_concentration_custom_sigma(
                float(solver_x), float(solver_y), float(solver_z),
                float(solver_H), float(solver_Q), float(solver_U),
                float(solver_sigma_y), float(solver_sigma_z)
            )
            max_C_g_m3 = point_C_g_m3 # Max C is just the point C because the max_x calculation is not applicable
            max_x_loc = solver_x
            sigma_y_used = solver_sigma_y
            sigma_z_used = solver_sigma_z
            
            st.info("Since fixed $\\sigma_y$ and $\\sigma_z$ were used, $C_{max}$ is simply the concentration at the point specified, and $x_{max}$ is set to the input $x$ distance.")
            
        else:
            # Mode 2: Use Pasquill-Gifford Curves (Original Logic)
            point_C_g_m3 = calculate_single_point_concentration(
                float(solver_x), float(solver_y), float(solver_z),
                float(solver_H), float(solver_Q), float(solver_U),
                solver_stab_class
            )
            max_C_g_m3, max_x_loc = find_max_concentration(solver_H, solver_Q, solver_U, solver_stab_class)
            sigma_y_used, sigma_z_used = get_dispersion_coefficients(float(solver_x), solver_stab_class)
        
        # --- DISPLAY RESULTS ---
        point_C_ug_m3 = point_C_g_m3 * 1e6
        max_C_ug_m3 = max_C_g_m3 * 1e6

        colR1, colR2 = st.columns(2)
        with colR1:
            st.metric(
                label=f"Concentration at $C({solver_x} m, {solver_y} m, {solver_z} m)$",
                value=f"{point_C_ug_m3:,.2f} µg/m³"
            )
        with colR2:
            # Display Max C only if using PG curves or if Max C is the Point C
            if not use_custom_sigma or (use_custom_sigma and max_x_loc == solver_x):
                st.metric(
                    label="Maximum Ground Concentration ($C_{max}$)",
                    value=f"{max_C_ug_m3:,.2f} µg/m³"
                )

        if not use_custom_sigma:
            st.metric(label="Location of Maximum Ground Concentration ($x_{max}$)", value=f"{max_x_loc:,.1f} m")
        
        st.markdown(f"""
        **Dispersion Coefficients Used at $x={solver_x} \\,\\text{{m}}$:**
        * $\\sigma_y$: **{sigma_y_used:,.2f} $\\text{{m}}$**
        * $\\sigma_z$: **{sigma_z_used:,.2f} $\\text{{m}}$**
        """)

        # -----------------------
        # Custom interactive visualization for Problem Solver (below results)
        # (only contour; centerline option removed)
        # -----------------------
        st.markdown("---")
        st.subheader("Solver — Interactive Contour (hover to read values, zoom/pan available)")

        # Domain based on solver inputs
        x_plot_max = max(2.0 * float(solver_x), 500.0)
        x_plot_min = max(0.1, float(solver_x) * 0.01)  # avoid zero
        x_plot = np.linspace(x_plot_min, x_plot_max, 160)

        # Crosswind window
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

        # Compute concentration field for solver inputs (g/m^3)
        Cp = np.zeros_like(Xp, dtype=float)

        if use_custom_sigma:
            for ii in range(Xp.shape[0]):
                for jj in range(Xp.shape[1]):
                    xx = float(Xp[ii, jj])
                    yy = float(Yp[ii, jj])
                    Cp[ii, jj] = calculate_point_concentration_custom_sigma(
                        xx, yy, float(solver_z),
                        float(solver_H), float(solver_Q), float(solver_U),
                        float(solver_sigma_y), float(solver_sigma_z)
                    )
            plot_mode_label = f"Solver Visualization (Custom σ) at z={solver_z} m"
        else:
            Cp = gaussian_plume_model(
                Xp, Yp, float(solver_z),
                float(solver_H), float(solver_Q), float(solver_U),
                solver_stab_class
            )
            plot_mode_label = f"Solver Visualization (Pasquill-Gifford: {solver_stab_class}) at z={solver_z} m"

        Cp_ug = Cp * 1e6

        # Build a Plotly contour for interactivity (hover + zoom/pan)
        fig_pl = go.Figure(
            data=go.Contour(
                z=Cp_ug,
                x=x_plot,  # downwind
                y=y_plot,  # crosswind
                colorscale='Viridis',
                contours=dict(showlabels=False),
                colorbar=dict(title="Concentration (µg m⁻³)"),
                hovertemplate='x: %{x:.1f} m<br>y: %{y:.1f} m<br>C: %{z:.2f} µg m⁻³<extra></extra>'
            )
        )

        # Add stack marker as scatter
        fig_pl.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode='markers',
                marker=dict(color='red', size=6),
                name='Stack (0,0)',
                hoverinfo='skip'
            )
        )

        fig_pl.update_layout(
            title=plot_mode_label,
            xaxis_title='Downwind distance x (m)',
            yaxis_title='Crosswind distance y (m)',
            autosize=True,
            margin=dict(l=40, r=20, t=50, b=40)
        )

        # Show interactive plot (zoom, pan, hover)
        st.plotly_chart(fig_pl, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
# --- TAB 3: THEORY & ASSUMPTIONS (UPDATED) ---
with tab3:
    st.header("Gaussian Plume Model Theory & Assumptions")

    st.markdown(
        """
        The Gaussian Plume Model (GPM) is the fundamental steady-state model for predicting the dispersion of continuous, buoyant pollutants released from a single point source, such as a chimney stack. It assumes that the pollutant concentration forms a Gaussian (normal) distribution in both the lateral ($y$) and vertical ($z$) directions, normal to the mean wind direction ($x$).
        
        ***

        ### Core Equation for $\mathbf{C(x, y, z)}$

        The general equation, which includes the vertical height $z$ and the effect of total ground reflection (the **virtual image source**), is:
        """
    )

    # --- Diagram of the Gaussian Plume Model added for clarity ---
    st.markdown("")

    # Render the equation cleanly using Streamlit's latex renderer
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
        
        ### References and Acknowledgments

        ***Primary Source Reference:*** *These notes are adapted from the lecture materials on Gaussian Plumes by E. Savory, available at [eng.uwo.ca/people/esavory/gaussian plumes.pdf](https://www.eng.uwo.ca/people/esavory/gaussian%20plumes.pdf).*

        For further study and reference on Environmental Engineering and dispersion modeling, you may consult the work and class notes of:

        **Dr. Abhradeep Majumder, Ph.D.**
        * Assistant Professor, Department of Civil Engineering, BITS Pilani-Hyderabad Campus
        * Academic Profiles: [Scopus](https://www.scopus.com/authid/detail.uri?authorId=57191504507), [ORCID](https://orcid.org/0000-0002-0186-6450), [Google Scholar](https://scholar.google.co.in/citations?user=mnJ5zdwAAAAJ&hl=en&oi=ao), [LinkedIn](https://linkedin.com/in/abhradeep-majumder-36503777/)
        
        ---
        ###### Application Development

        This interactive Gaussian Plume Dispersion Model application was developed by **Subodh Purohit** as an educational tool.
        """
    )
