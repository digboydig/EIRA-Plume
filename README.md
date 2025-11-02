🏭 Gaussian Plume Model Visualizer

This is an interactive Streamlit application designed to visualize the dispersion of pollutants from a stack source using the classic Pasquill-Gifford Gaussian Plume Model. It allows users to dynamically adjust meteorological and source parameters to instantly see the resulting ground-level concentration contour plot.

Key capabilities:

- Interactive contour maps of concentration at ground level or any user-specified receptor height.
- Time-evolution advection animation (simple translation by wind speed) with optional MP4 export of rendered frames.
- Point-concentration solver with Pasquill–Gifford dispersion curves or user-specified σ_y / σ_z values.
- Cross-sectional views: X vs Z (downwind vs height along centerline) and Y vs Z (crosswind vs height at chosen x).
- Summary metrics: maximum ground concentration, location of maximum, plume half-width, and mixing-height estimate.


✨ Features

Interactive Parameter Control: Adjust Source Strength ($Q$), Effective Stack Height ($H$), Wind Speed ($u$), and Simulation Area limits using sidebar sliders.

Atmospheric Stability: Select the Pasquill-Gifford Stability Class (A to F) to model different atmospheric mixing conditions.

Real-time Visualization: Generates a 2D contour plot of ground-level pollutant concentration ($C(x, y, 0)$) in real-time as parameters are changed.

Key Findings: Displays calculated metrics, including the maximum ground-level concentration and the plume's dispersion ($\sigma_y, \sigma_z$) at the far field.

**Usage summary**
Sidebar — set source parameters:
Emission rate Q (g/s)
Effective stack height H (m)
Wind speed U (m/s)
Atmospheric stability class (A–F)
Optional receptor height z (toggle)
Plume Visualizer tab:
Interactive Plotly contour map of concentration (µg/m³).
Optional advection animation (advects plume downstream by wind speed).
MP4 export: renders frames with matplotlib and bundles to MP4 via imageio (may require ffmpeg).
Problem Solver tab:
Compute concentration at a chosen point using PG curves or custom σ values.
Optional interactive contour for the solver parameters.
Theory & Assumptions tab:
Displays the model equation, assumptions and references/acknowledgements.
