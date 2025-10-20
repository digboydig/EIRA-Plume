import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

# --- 2. MODEL FUNCTIONS ---

@st.cache_data
def get_dispersion_coefficients(x, stability_class):
    """Calculates sigma_y and sigma_z based on downwind distance and stability class."""
    
    # Get the index (0-5) for the stability class
    try:
        idx = SIGMA_COEFFS['index'][stability_class]
    except KeyError:
        # Since this is a utility function, we use st.error sparingly
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
def gaussian_plume_model(x_m, y_m, H, Q, U, stability_class):
    """
    Calculates the ground-level concentration C(x, y, 0) using the Gaussian Plume Equation for a meshgrid.
    """
    # Initialize concentration array
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
            
            # Check for division by zero (shouldn't happen with the x>0 check and sigma_z floor)
            if sigma_y == 0 or sigma_z == 0:
                C[i, j] = 0.0
                continue
            
            # 1. Crosswind (y) term
            exp_y = np.exp(-y**2 / (2 * sigma_y**2))
            
            # 2. Vertical (z) term (ground reflection)
            exp_z = np.exp(-H**2 / (2 * sigma_z**2))
            
            # 3. Scaling Factor (Concentration depends linearly on Q/U)
            scaling_factor = Q / (np.pi * U * sigma_y * sigma_z)
            
            # Total concentration
            C[i, j] = scaling_factor * exp_y * exp_z
            
    return C

def find_max_concentration(H, Q, U, stability_class):
    """Finds the maximum ground-level concentration and its downwind distance (x) on the center-line (y=0)."""
    
    # Search range for x (downwind distance)
    x_range = np.linspace(10, 5000, 500) # Search from 10m to 5000m
    
    max_C = 0.0
    max_x = 0.0
    
    for x in x_range:
        sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)
        
        # Avoid division by zero
        if sigma_y == 0 or sigma_z == 0:
            continue
            
        # Simplified equation for center-line C(x, 0, 0)
        C_centerline = (Q / (np.pi * U * sigma_y * sigma_z)) * np.exp(-H**2 / (2 * sigma_z**2))
        
        if C_centerline > max_C:
            max_C = C_centerline
            max_x = x
            
    return max_C, max_x

def calculate_single_point_concentration(x, y, H, Q, U, stability_class):
    """Calculates the concentration C(x, y, 0) at a single specified ground-level point (scalar inputs)."""
    if x <= 0:
        return 0.0

    sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)

    if sigma_y == 0 or sigma_z == 0:
        return 0.0

    # 1. Crosswind (y) term
    exp_y = np.exp(-y**2 / (2 * sigma_y**2))
    
    # 2. Vertical (z=0) term (ground reflection)
    exp_z_ground = np.exp(-H**2 / (2 * sigma_z**2))
    
    # 3. Scaling Factor
    scaling_factor = Q / (np.pi * U * sigma_y * sigma_z)
    
    # Total concentration (g/m^3)
    C = scaling_factor * exp_y * exp_z_ground
    
    return C

# --- NEW FUNCTION FOR ARBITRARY Z (THE PRIMARY CORRECTION) ---
def calculate_any_point_concentration(x, y, z, H, Q, U, stability_class):
    """
    Calculates the concentration C(x, y, z) at a single specified point (scalar inputs)
    using the full Gaussian Plume Equation.
    """
    if x <= 0:
        return 0.0

    sigma_y, sigma_z = get_dispersion_coefficients(x, stability_class)

    if sigma_y == 0 or sigma_z == 0:
        return 0.0

    # 1. Crosswind (y) term
    exp_y = np.exp(-y**2 / (2 * sigma_y**2))
    
    # 2. Vertical (z) term (real source + virtual image source)
    # The term in the full equation is: exp(-(z-H)^2 / (2*sigma_z^2)) + exp(-(z+H)^2 / (2*sigma_z^2))
    exp_z_real = np.exp(-(z - H)**2 / (2 * sigma_z**2))
    exp_z_image = np.exp(-(z + H)**2 / (2 * sigma_z**2))
    
    # 3. Scaling Factor
    # Note the factor of 0.5 compared to the C(x,y,0) simplified equation
    scaling_factor = Q / (2 * np.pi * U * sigma_y * sigma_z)
    
    # Total concentration (g/m^3)
    C = scaling_factor * exp_y * (exp_z_real + exp_z_image)
    
    return C
# -----------------------------------------------------------------


# --- 3. STREAMLIT APP LAYOUT (MAIN PAGE) ---

st.set_page_config(layout="wide", page_title="Gaussian Plume Visualizer")

st.title("Gaussian Plume Dispersion Visualizer")
st.markdown("An interactive model to explore how source parameters and atmospheric stability affect pollutant spread and ground-level concentration.")

# --- SIDEBAR (GLOBAL USER INPUTS) ---

st.sidebar.header("Source & Environment Parameters")

# 1. Emission Parameters
st.sidebar.subheader("1. Source Strength")
Q_g_s = st.sidebar.slider("Emission Rate ($Q$, g/s)", 10.0, 500.0, 100.0, step=10.0)
H_m = st.sidebar.slider("Effective Stack Height ($H$, m)", 10.0, 200.0, 100.0, step=5.0)

# 2. Atmospheric Conditions
st.sidebar.subheader("2. Atmospheric Conditions")
U_m_s = st.sidebar.slider("Wind Speed ($U$, m/s)", 1.0, 20.0, 5.0, step=0.5)

stability_options = {'A': 'A - Extremely Unstable', 'B': 'B - Moderately Unstable', 'C': 'C - Slightly Unstable',
                     'D': 'D - Neutral (Overcast/High Wind)', 'E': 'E - Slightly Stable', 'F': 'F - Moderately Stable'}
stability_class = st.sidebar.selectbox("Atmospheric Stability Class", options=list(stability_options.keys()), format_func=lambda x: stability_options[x], index=3)

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

# --- TAB 1: PLUME VISUALIZER ---
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Plume Concentration Contour Map")

        # Define the domain for visualization
        X_MAX = 4000  # meters downwind
        Y_MAX = 500   # meters crosswind (half-width, total 1000m)
        
        # Create meshgrid for X and Y coordinates
        x_range = np.linspace(0, X_MAX, 200)
        y_range = np.linspace(-Y_MAX, Y_MAX, 100)
        X, Y = np.meshgrid(x_range, y_range)

        # Calculate concentrations (C in g/m^3)
        C_values = gaussian_plume_model(X, Y, H_m, Q_g_s, U_m_s, stability_class)

        # Convert to micrograms per cubic meter (μg/m^3) for better scale display
        C_ug_m3 = C_values * 1e6
        
        # Ensure there is data to plot before proceeding
        if np.max(C_ug_m3) > 1e-6:

            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Use log scale for contours to better show the edges of the plume
            levels = np.linspace(0, np.max(C_ug_m3), 15)
            
            # Plot the contour map
            c = ax.contourf(X, Y, C_ug_m3, levels=levels, cmap='jet')
            
            # Add a color bar
            cbar = fig.colorbar(c, ax=ax, label=r'Concentration ($C(x,y,0)$ in $\mu g/m^3$)')
            
            # Plot center-line and source location
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.plot([0], [0], 'ro', markersize=8, label='Stack Location (x=0, y=0)')
            
            # Set titles and labels
            ax.set_title(f'Ground-Level Concentration Map (Stability: {stability_class}, H: {H_m}m)')
            ax.set_xlabel('Downwind Distance ($x$, m)')
            ax.set_ylabel('Crosswind Distance ($y$, m)')
            ax.set_xlim(0, X_MAX)
            ax.set_ylim(-Y_MAX, Y_MAX)
            ax.legend(loc='upper right')
            ax.grid(linestyle=':', alpha=0.5)
            
            st.pyplot(fig)
        else:
            st.warning("The calculated concentration is near zero. Please adjust parameters (e.g., lower H, increase Q, or select a more unstable class).")

    with col2:
        st.header("Key Model Findings")
        
        # 1. Calculate Max Concentration
        max_C_g_m3, max_x = find_max_concentration(H_m, Q_g_s, U_m_s, stability_class)
        max_C_ug_m3 = max_C_g_m3 * 1e6
        
        # Max Concentration Metric
        st.metric(
            label="Max Ground Concentration (on center-line)",
            value=f"{max_C_ug_m3:,.2f}"
        )
        st.markdown(r"**Units:** $\mu g/m^3$") 

        st.metric(
            label="Downwind Distance of Max Concentration ($x_{max}$)",
            value=f"{max_x:,.0f} m"
        )

        # 2. Calculate Plume Dimensions at max_x
        if max_x > 0:
            sigma_y_max_x, sigma_z_max_x = get_dispersion_coefficients(max_x, stability_class)
            
            st.markdown(f"### Plume Dimensions (at $x={int(max_x)}$ m)")
            
            # Plume Half-Width (y)
            st.metric(
                label="Plume Half-Width ($2\sigma_y$)",
                value=f"{2 * sigma_y_max_x:,.1f} m"
            )

            # Plume Height (z)
            st.metric(
                label="Plume Mixing Height ($4.3\sigma_z$ from ground)",
                value=f"{4.3 * sigma_z_max_x:,.1f} m"
            )
        else:
            st.info("Plume maximum could not be calculated. Ensure H is not too large or Q is not too small.")

# --- TAB 2: PROBLEM SOLVER ---
with tab2:
    st.header("Point Concentration & $x_{max}$ Solver (Custom Inputs)")
    st.markdown("Calculate concentrations and the maximum ground-level location using **custom parameters** independent of the visualizer's sidebar.")
    
    # --- CUSTOM INPUTS FOR THE SOLVER ---
    colA, colB, colC = st.columns(3)
    
    with colA:
        solver_Q = st.number_input("1. Emission Rate ($Q$, g/s)", min_value=1.0, value=100.0, step=10.0, key='Q_solver')
    with colB:
        solver_H = st.number_input("2. Effective Stack Height ($H$, m)", min_value=1.0, value=100.0, step=5.0, key='H_solver')
    with colC:
        solver_U = st.number_input("3. Wind Speed ($U$, m/s)", min_value=0.1, value=5.0, step=0.5, key='U_solver')

    solver_stab_class = st.selectbox("4. Atmospheric Stability Class", options=list(stability_options.keys()), format_func=lambda x: stability_options[x], index=3, key='stability_solver_key')
    
    st.markdown("---")
    st.subheader("5. Point Location Input")
    colX, colY, colZ = st.columns(3)
    
    with colX:
        solver_x = st.number_input("Downwind Distance ($x$, m)", min_value=1.0, value=1000.0, step=10.0, key='x_input_solver')
    with colY:
        solver_y = st.number_input("Crosswind Distance ($y$, m)", value=0.0, step=10.0, key='y_input_solver')
    with colZ:
        solver_z = st.number_input("Vertical Height ($z$, m) (Use 0 for ground level)", value=0.0, step=10.0, key='z_input_solver')

    
    # --- CALCULATION AND DISPLAY ---
    if st.button("Run Calculations for Custom Parameters", key='solve_button'):
        st.subheader("Calculated Results")
        
        # 1. Calculate Concentration at the specified point (x, y, z)
        point_C_g_m3 = calculate_any_point_concentration(solver_x, solver_y, solver_z, solver_H, solver_Q, solver_U, solver_stab_class)
        point_C_ug_m3 = point_C_g_m3 * 1e6
        
        # 2. Calculate Max Ground Concentration and its location (x_max)
        max_C_g_m3, max_x_loc = find_max_concentration(solver_H, solver_Q, solver_U, solver_stab_class)
        max_C_ug_m3 = max_C_g_m3 * 1e6
        
        colR1, colR2 = st.columns(2)
        
        with colR1:
            st.metric(
                label=f"Concentration at $C({solver_x}m, {solver_y}m, {solver_z}m)$",
                value=f"{point_C_ug_m3:,.2f} $\mu g/m^3$"
            )
        
        with colR2:
            st.metric(
                label="Maximum Ground Concentration ($C_{max}$)",
                value=f"{max_C_ug_m3:,.2f} $\mu g/m^3$"
            )

        st.metric(
            label="Location of Maximum Ground Concentration ($x_{max}$)",
            value=f"{max_x_loc:,.1f} m"
        )

        sigma_y_point, sigma_z_point = get_dispersion_coefficients(solver_x, solver_stab_class)
        st.markdown(f"""
        **Dispersion Coefficients Used at $x={solver_x} \text{ m}$:**
        * $\sigma_y$: **{sigma_y_point:,.2f} m**
        * $\sigma_z$: **{sigma_z_point:,.2f} m**
        """)
        
    st.markdown("---")
    st.header("Advanced Problem Set (Matching Textbook Examples)")
    st.markdown("The following sections solve the questions from the provided academic notes using the Gaussian Plume Model.")
    
    
    # --- QUESTION 1: Sulphur Dioxide (SO2) ---
    st.subheader("Question 1: Fixed Plume Rise ($H=130 \text{ m}$)")
    st.markdown("""
    **Parameters:** $Q=2000 \text{ g/s}$, $h_s=120 \text{ m}$, $\Delta h=10 \text{ m} \implies H=130 \text{ m}$, $U=15 \text{ m/s}$, Stability Class **D** (Neutral).
    """)
    
    Q1_Q = 2000.0   # g/s
    Q1_H = 130.0   # m (120 + 10)
    Q1_U = 15.0    # m/s
    Q1_X = 800.0   # m
    Q1_STAB = 'D'

    # Set z = H for part (a) - centerline height
    Q1_Z_centerline = Q1_H 
    
    sigma_y_Q1, sigma_z_Q1 = get_dispersion_coefficients(Q1_X, Q1_STAB)
    
    # (a) Concentration on plume centre-line (C(800, 0, H))
    C_plume_center_g_m3 = calculate_any_point_concentration(Q1_X, 0.0, Q1_Z_centerline, Q1_H, Q1_Q, Q1_U, Q1_STAB)
    C_plume_center_ug_m3 = C_plume_center_g_m3 * 1e6
    
    # (b) Ground level concentration at 800m (C(800, 0, 0))
    C_ground_g_m3 = calculate_single_point_concentration(Q1_X, 0.0, Q1_H, Q1_Q, Q1_U, Q1_STAB)
    C_ground_ug_m3 = C_ground_g_m3 * 1e6
    
    # (c) & (d) Maximum Ground Concentration location and value
    max_C_g_m3, max_x_loc = find_max_concentration(Q1_H, Q1_Q, Q1_U, Q1_STAB)
    max_C_ug_m3 = max_C_g_m3 * 1e6

    colQ1_1, colQ1_2, colQ1_3, colQ1_4 = st.columns(4)

    with colQ1_1:
        st.metric(label="(a) $C(800, 0, H)$", value=f"{C_plume_center_ug_m3:,.1f} $\mu g/m^3$", delta=f"Target: 5311 $\mu g/m^3$")
    with colQ1_2:
        st.metric(label="(b) $C(800, 0, 0)$", value=f"{C_ground_ug_m3:,.1f} $\mu g/m^3$", delta=f"Target: 257 $\mu g/m^3$")
    with colQ1_3:
        st.metric(label="(c) $x_{max}$", value=f"{max_x_loc:,.1f} m", delta=f"Target: 1542.3 m")
    with colQ1_4:
        st.metric(label="(d) $C_{max}$", value=f"{max_C_ug_m3:,.1f} $\mu g/m^3$", delta=f"Target: 1051 $\mu g/m^3$")
    
    st.caption(f"Dispersion coefficients used at $x=800 \text{ m}$: $\sigma_y={sigma_y_Q1:,.2f} \text{ m}$, $\sigma_z={sigma_z_Q1:,.2f} \text{ m}$.")
    
    st.markdown("---")
    
    # --- QUESTION 2: Hydrogen Chloride (HCl) ---
    st.subheader("Question 2: Non-Centerline Concentration")
    st.markdown("""
    **Derived Parameters:** $h_s=100 \text{ m}$, $\Delta h=5 \text{ m} \implies H=105 \text{ m}$, $U=10 \text{ m/s}$, Stability Class **D**.
    **Emission Rate:** $Q = \text{Volume Flow} \times \text{Density} = 1 \text{ m}^3/\text{s} \times 1.64 \text{ kg}/\text{m}^3 = 1.64 \text{ kg/s} = 1640 \text{ g/s}$.
    """)
    
    Q2_Q = 1640.0   # g/s
    Q2_H = 105.0   # m
    Q2_U = 10.0    # m/s
    Q2_X = 1500.0  # m
    Q2_STAB = 'D'
    
    # (a) C(1500, 500, 0)
    C_off_center_g_m3 = calculate_single_point_concentration(Q2_X, 500.0, Q2_H, Q2_Q, Q2_U, Q2_STAB)
    C_off_center_ug_m3 = C_off_center_g_m3 * 1e6
    
    # (b) C(1500, 0, 0)
    C_on_center_g_m3 = calculate_single_point_concentration(Q2_X, 0.0, Q2_H, Q2_Q, Q2_U, Q2_STAB)
    C_on_center_ug_m3 = C_on_center_g_m3 * 1e6
    
    sigma_y_Q2, sigma_z_Q2 = get_dispersion_coefficients(Q2_X, Q2_STAB)
    
    colQ2_1, colQ2_2 = st.columns(2)
    with colQ2_1:
        st.metric(label="(a) $C(1500, 500, 0)$", value=f"{C_off_center_ug_m3:,.1f} $\mu g/m^3$", delta=f"Target: 43 $\mu g/m^3$")
    with colQ2_2:
        st.metric(label="(b) $C(1500, 0, 0)$", value=f"{C_on_center_ug_m3:,.1f} $\mu g/m^3$", delta=f"Target: 1537 $\mu g/m^3$")
        
    st.caption(f"Dispersion coefficients used at $x=1500 \text{ m}$: $\sigma_y={sigma_y_Q2:,.2f} \text{ m}$, $\sigma_z={sigma_z_Q2:,.2f} \text{ m}$.")
    
    st.markdown("---")
    
    # --- QUESTION 3: Variable Plume Rise ---
    st.subheader("Question 3: Distance-Dependent Plume Rise")
    st.markdown("""
    **Parameters:** $Q=2500 \text{ g/s}$, $h_s=100 \text{ m}$, $U=10 \text{ m/s}$, $x=2500 \text{ m}$.
    **Plume Rise Function:** $\Delta h(x) = 0.13 x^{2/3}$.
    **Stability Assumption:** Since the plume is highly buoyant, we assume **Stability Class B** (Moderately Unstable) for a greater, though not extreme, vertical spread.
    """)
    
    Q3_Q = 2500.0   # g/s
    Q3_hs = 100.0  # m
    Q3_U = 10.0    # m/s
    Q3_X = 2500.0  # m
    Q3_STAB = 'B'
    
    # Calculate H at x=2500m
    delta_h_Q3 = 0.13 * (Q3_X**(2/3))
    Q3_H = Q3_hs + delta_h_Q3
    
    # C(2500, 0, 0)
    C_Q3_g_m3 = calculate_single_point_concentration(Q3_X, 0.0, Q3_H, Q3_Q, Q3_U, Q3_STAB)
    C_Q3_ug_m3 = C_Q3_g_m3 * 1e6
    
    sigma_y_Q3, sigma_z_Q3 = get_dispersion_coefficients(Q3_X, Q3_STAB)
    
    colQ3_1, colQ3_2 = st.columns(2)
    with colQ3_1:
        st.metric(label="Calculated Plume Rise ($\Delta h$)", value=f"{delta_h_Q3:,.2f} m")
    with colQ3_2:
        st.metric(label="Effective Stack Height ($H$)", value=f"{Q3_H:,.2f} m")
        
    st.metric(
        label="Ground Level Concentration ($C(2500, 0, 0)$)",
        value=f"{C_Q3_ug_m3:,.1f} $\mu g/m^3$",
        delta=f"Target: 1400 $\mu g/m^3$"
    )
    st.caption(f"Dispersion coefficients used at $x=2500 \text{ m}$ (Stability B): $\sigma_y={sigma_y_Q3:,.2f} \text{ m}$, $\sigma_z={sigma_z_Q3:,.2f} \text{ m}$.")
    
    st.markdown("---")
    
    # --- QUESTION 4: Gross Screening vs. Gaussian ---
    st.subheader("Question 4: Gross Screening vs. Gaussian Plume")
    st.markdown("""
    **Parameters:** Ground level release $H=4 \text{ m}$, $Q=0.5 \text{ g/s}$, $U=1 \text{ m/s}$, $x=4000 \text{ m}$.
    **Stability Assumption:** For the Gaussian model to show a reasonable 'worst case' near the target, we use **Stability Class F** (Moderately Stable), which minimizes vertical mixing.
    
    **Gross Screening Formula:** $C_{wc} = \frac{10^9 Q}{U H_{wc} W_{wc}}$, where $H_{wc}=50 \text{ m}$ and $W_{wc}=0.1x$.
    """)
    
    Q4_Q = 0.5     # g/s
    Q4_H = 4.0     # m
    Q4_U = 1.0     # m/s
    Q4_X = 4000.0  # m
    Q4_STAB = 'F'
    
    # 1. Gross Screening Method
    Q4_Hwc = 50.0  # m
    Q4_Wwc = 0.1 * Q4_X # 400.0 m
    
    C_screening_ug_m3 = (1e9 * Q4_Q) / (Q4_U * Q4_Hwc * Q4_Wwc)
    
    # 2. Gaussian Plume Equation (C(4000, 0, 0))
    C_gaussian_g_m3 = calculate_single_point_concentration(Q4_X, 0.0, Q4_H, Q4_Q, Q4_U, Q4_STAB)
    C_gaussian_ug_m3 = C_gaussian_g_m3 * 1e6
    
    sigma_y_Q4, sigma_z_Q4 = get_dispersion_coefficients(Q4_X, Q4_STAB)
    
    colQ4_1, colQ4_2 = st.columns(2)
    with colQ4_1:
        st.metric(
            label="Gross Screening Concentration",
            value=f"{C_screening_ug_m3:,.2f} $\mu g/m^3$",
            delta=f"Target: 25 $\mu g/m^3$"
        )
    with colQ4_2:
        st.metric(
            label="Gaussian Plume Concentration (Stability F)",
            value=f"{C_gaussian_ug_m3:,.2f} $\mu g/m^3$",
            delta=f"Target: 0.56 $\mu g/m^3$"
        )
    st.caption(f"Dispersion coefficients used at $x=4000 \text{ m}$ (Stability F): $\sigma_y={sigma_y_Q4:,.2f} \text{ m}$, $\sigma_z={sigma_z_Q4:,.2f} \text{ m}$.")

# --- TAB 3: THEORY & ASSUMPTIONS ---
with tab3:
    st.header("Gaussian Plume Model Theory & Assumptions")
    
    st.markdown(
        """
    ***Primary Source Reference:*** *These notes are adapted from the lecture materials on Gaussian Plumes by E. Savory, available at [eng.uwo.ca/people/esavory/gaussian plumes.pdf](https://www.eng.uwo.ca/people/esavory/gaussian%20plumes.pdf)*.

    ---
    
    ### Additional Resource: 
    
    For further study and reference on Environmental Engineering and dispersion modeling, you may consult the work and class notes of:
    
    **Dr. Abhradeep Majumder, Ph.D.**
    * Assistant Professor, Department of Civil Engineering, BITS Pilani-Hyderabad Campus
    * Academic Profiles:
        * Scopus: [https://www.scopus.com/authid/detail.uri?authorId=57191504507](https://www.scopus.com/authid/detail.uri?authorId=57191504507)
        * ORCID: [https://orcid.org/0000-0002-0186-6450](https://orcid.org/0000-0002-0186-6450)
        * Google Scholar: [https://scholar.google.co.in/citations?user=mnJ5zdwAAAAJ&hl=en&oi=ao](https://scholar.google.co.in/citations?user=mnJ5zdwAAAAJ&hl=en&oi=ao)
        * LinkedIn: [linkedin.com/in/abhradeep-majumder-36503777/](https://linkedin.com/in/abhradeep-majumder-36503777/)
        
    ---
    

    The Gaussian Plume Model (GPM) is the fundamental steady-state model for predicting the dispersion of continuous, buoyant pollutants released from a single point source, such as a chimney stack. It assumes that the pollutant concentration forms a Gaussian (normal) distribution in both the lateral ($y$) and vertical ($z$) directions, normal to the mean wind direction ($x$).
    
    ### Key Assumptions
    1.  **Steady State:** Emission rate ($Q$) and wind speed ($U$) are constant.
    2.  **Uniform Wind:** Wind flows uniformly in the $x$-direction.
    3.  **Total Reflection:** The pollutant is completely reflected off the ground surface (a virtual "mirror image" source is used).
    4.  **Gaussian Distribution:** Concentration profiles are Gaussian in the cross-wind and vertical directions.
    
    ### Core Ground-Level Equation
    For ground-level concentrations ($z=0$), the total concentration $C(x, y, 0)$ is the sum of the concentration from the real source and its virtual image source. The simplified equation is:
    
    $$
    C(x, y, 0) = \frac{Q}{\pi U \sigma_y \sigma_z} \exp\left(-\frac{y^2}{2\sigma_y^2}\right) \exp\left(-\frac{H^2}{2\sigma_z^2}\right)
    $$
    
    Where:
    * $C(x, y, 0)$: Ground-level concentration ($\mu g/m^3$)
    * $Q$: Source Emission Rate ($g/s$)
    * $U$: Mean Wind Speed ($m/s$)
    * $H$: **Effective Stack Height** ($m$) - Sum of physical stack height and plume rise ($\Delta h$).
    * $\sigma_y$ and $\sigma_z$: Lateral and Vertical Dispersion Coefficients ($m$)
    
    ### Dispersion Coefficients ($\sigma_y$ and $\sigma_z$)
    These coefficients represent the standard deviations of the Gaussian distributions in the cross-wind and vertical directions, respectively. Their values are empirical (experimentally derived) and depend heavily on two main factors:
    1.  **Downwind Distance ($x$):** The plume spreads as it travels, so $\sigma_y$ and $\sigma_z$ increase with $x$.
    2.  **Atmospheric Stability Class (A-F):** This is the most crucial parameter.
        * **Unstable (A, B, C):** Characterized by high turbulence (e.g., sunny day). $\sigma_y$ and $\sigma_z$ are large, leading to rapid mixing and a plume that "touches down" quickly.
        * **Neutral (D):** Moderate mixing (e.g., overcast day, high wind).
        * **Stable (E, F):** Characterized by low turbulence (e.g., clear night). $\sigma_z$ is very small, leading to poor vertical mixing and a narrow plume that travels far downwind before reaching maximum ground concentration.

    The model uses power law approximations ($\sigma = A \cdot x^B$) where $A$ and $B$ are constants derived from the chosen Pasquill stability class.


    ---

    ###### Application Development
    
    This interactive Gaussian Plume Dispersion Model application was developed by **Subodh Purohit** to provide a dynamic, educational tool for exploring pollutant spread based on atmospheric and source conditions.

    ---

    """)
