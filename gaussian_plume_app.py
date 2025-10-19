import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Pasquill-Gifford Dispersion Coefficients ---
# These are simplified power-law coefficients for sigma_y (crosswind) and sigma_z (vertical)
# based on downwind distance X (in meters).
# The coefficients are based on the approximation: sigma(x) = a * x^b
# Note: Real-world models use piecewise functions, but this simple set is good for demonstration.

PG_COEFFICIENTS = {
    # Stability Class: [a_y, b_y, a_z, b_z]
    'A (Very Unstable)': [0.22, 0.90, 0.20, 0.91],
    'B (Moderately Unstable)': [0.16, 0.90, 0.12, 0.91],
    'C (Slightly Unstable)': [0.11, 0.90, 0.08, 0.91],
    'D (Neutral)': [0.08, 0.90, 0.06, 0.894], # Most common/neutral
    'E (Slightly Stable)': [0.06, 0.90, 0.03, 0.894],
    'F (Moderately Stable)': [0.04, 0.90, 0.016, 0.894],
}

# --- 2. Gaussian Plume Model Functions ---

def calculate_sigmas(x, stability_class):
    """Calculates crosswind (sigma_y) and vertical (sigma_z) dispersion coefficients."""
    if x == 0:
        # Avoid division by zero at the source point
        return 0.0, 0.0
    
    a_y, b_y, a_z, b_z = PG_COEFFICIENTS[stability_class]
    
    # Ensure x is in meters for the formula
    sigma_y = a_y * (x) ** b_y
    
    # A common correction for stable classes (E, F) at short distance 
    # is sometimes applied, but we stick to the simple power law here.
    sigma_z = a_z * (x) ** b_z
    
    # Apply a minimum sigma value to prevent numerical instability at small x
    min_sigma = 0.5 
    return max(sigma_y, min_sigma), max(sigma_z, min_sigma)

def calculate_ground_concentration(Q, H, u, x, y, stability_class):
    """
    Calculates the ground-level concentration C(x, y, 0) using the Gaussian Plume Model.
    
    Units:
    Q: Source strength (g/s)
    H: Effective stack height (m)
    u: Wind speed (m/s)
    x: Downwind distance (m)
    y: Crosswind distance (m)
    C: Concentration (g/m^3)
    """
    if x <= 1.0: # Close to stack (x=0) is a singularity, handle it gracefully
        return 0.0 
        
    sigma_y, sigma_z = calculate_sigmas(x, stability_class)
    
    if sigma_y == 0 or sigma_z == 0 or u == 0:
        return 0.0

    # Crosswind dispersion term
    exp_y = np.exp(-0.5 * (y / sigma_y)**2)
    
    # Vertical dispersion term (ground-level z=0, total reflection)
    exp_z = np.exp(-0.5 * (H / sigma_z)**2)
    
    # Full Gaussian Plume Equation for C(x, y, 0)
    C = (Q / (np.pi * u * sigma_y * sigma_z)) * exp_y * exp_z
    
    return C

# --- 3. Streamlit Application Layout and Logic ---

st.set_page_config(
    page_title="Gaussian Plume Model Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Gaussian Plume Model Visualizer")
st.markdown("An interactive tool to explore pollutant dispersion and ground-level concentration based on the Pasquill-Gifford Gaussian Plume Model.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Model Parameters")
    
    # Source Parameters
    st.subheader("Source & Meteorological Data")
    Q = st.slider("Source Strength ($Q$, g/s)", 10.0, 500.0, 100.0, 10.0)
    H = st.slider("Effective Stack Height ($H$, m)", 10.0, 200.0, 50.0, 5.0)
    u = st.slider("Wind Speed ($u$, m/s)", 1.0, 15.0, 5.0, 0.5)
    
    stability_class = st.selectbox(
        "Atmospheric Stability Class",
        options=list(PG_COEFFICIENTS.keys()),
        index=3 # Default to Neutral (D)
    )
    
    st.subheader("Simulation Area")
    max_x = st.slider("Max Downwind Distance (m)", 500, 5000, 1500, 100)
    max_y = st.slider("Max Crosswind Distance (m)", 100, 1000, 400, 50)

    ## SIDEBAR ATTRIBUTION
    st.sidebar.markdown("---") # Visual separator from the summary
    
    # Using st.sidebar.markdown with HTML for custom styling:
    st.sidebar.markdown(
        "<div style='text-align: left; font-size: 10px; color: gray;'>"
        "<b>Plume Visualizer</b><br>"
        "Developed by: <b>Subodh Purohit</b><br>"
        "</div>", 
        unsafe_allow_html=True
    )


# --- Main Area Visualization ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Ground-Level Concentration Contour Plot")
    
    # Create X and Y arrays
    X = np.linspace(0, max_x, 100)
    Y = np.linspace(-max_y, max_y, 100)
    X, Y = np.meshgrid(X, Y)
    
    # Calculate Concentration Z for all (X, Y) points
    Z = np.zeros_like(X)
    
    # Vectorize the calculation to apply it across the entire grid efficiently
    # We use a wrapper for the vector function
    vectorized_calc = np.vectorize(calculate_ground_concentration)
    
    # Calculate Z
    Z = vectorized_calc(Q, H, u, X, Y, stability_class)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the contour plot
    # Use logarithmic spacing for levels for better visualization of the tail
    max_c = np.max(Z)
    if max_c > 1e-9:
        levels = np.logspace(np.log10(max_c * 0.01), np.log10(max_c), 10)
    else:
        levels = np.linspace(0.01, 1.0, 10) * 1e-6 # Fallback small values

    # Plot the contours
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    
    # Add a white dot for the stack location
    ax.plot(0, 0, 'w*', markersize=12, label='Stack Location')
    
    # Color bar
    cbar = fig.colorbar(contour, ax=ax, format='%.2e')
    cbar.set_label('Concentration ($C(x, y, 0)$, g/m$^3$)')
    
    ax.set_xlabel("Downwind Distance ($x$, m)")
    ax.set_ylabel("Crosswind Distance ($y$, m)")
    ax.set_title(f"Concentration Profile for Class: {stability_class}")
    ax.grid(True, linestyle='--')
    
    # Display the plot in Streamlit
    st.pyplot(fig)


with col2:
    st.header("Key Findings")
    
    # Calculate key metrics
    # 1. Max Ground Concentration
    if max_c > 0:
        max_c_value = np.max(Z)
        # Find the coordinates (x, y) of the max concentration
        y_idx, x_idx = np.unravel_index(np.argmax(Z), Z.shape)
        x_max_c = X[y_idx, x_idx]
        y_max_c = Y[y_idx, x_idx]
        
        st.metric("Max Ground Concentration", f"{max_c_value:.4e} g/m³")
        st.markdown(f"**Occurs at:** $x \\approx {x_max_c:.0f}$ m, $y \\approx {y_max_c:.0f}$ m")
        
        # 2. Sigma values at Max X
        sigma_y_end, sigma_z_end = calculate_sigmas(max_x, stability_class)
        st.markdown("---")
        # CORRECTED: Changed "$x=$$%d$ m" to an f-string using inline math "$x={max_x}$ m"
        st.subheader(f"Plume Dimensions (at $x={max_x}$ m)")
        st.metric("Crosswind Dispersion ($\sigma_y$)", f"{sigma_y_end:.2f} m")
        st.metric("Vertical Dispersion ($\sigma_z$)", f"{sigma_z_end:.2f} m")
    else:
        st.info("Adjust parameters to generate a measurable plume.")
        
    st.markdown("---")
    st.caption("""
        **Model Note:** This visualizer uses the basic Gaussian Plume Model 
        for ground-level concentration (z=0) with total reflection. 
        It employs simplified power-law coefficients for the dispersion 
        parameters ($\sigma_y, \sigma_z$).
    """)

# Ensure the plot is cleared after Streamlit uses it
plt.close(fig)
