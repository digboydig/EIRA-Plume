🏭 Gaussian Plume Model Visualizer

This is an interactive Streamlit application designed to visualize the dispersion of pollutants from a stack source using the classic Pasquill-Gifford Gaussian Plume Model. It allows users to dynamically adjust meteorological and source parameters to instantly see the resulting ground-level concentration contour plot.

✨ Features

Interactive Parameter Control: Adjust Source Strength ($Q$), Effective Stack Height ($H$), Wind Speed ($u$), and Simulation Area limits using sidebar sliders.

Atmospheric Stability: Select the Pasquill-Gifford Stability Class (A to F) to model different atmospheric mixing conditions.

Real-time Visualization: Generates a 2D contour plot of ground-level pollutant concentration ($C(x, y, 0)$) in real-time as parameters are changed.

Key Findings: Displays calculated metrics, including the maximum ground-level concentration and the plume's dispersion ($\sigma_y, \sigma_z$) at the far field.
