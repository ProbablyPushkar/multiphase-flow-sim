# Multiphase Well Performance Simulator
*by ProbablyPushkar*

A 1D steady-state nodal analysis and flow assurance tool built with Python and Streamlit. This application simulates multiphase flow in a wellbore, allowing petroleum and production engineers to visualize well performance, configure completions, and assess flow assurance risks.

## Features

* **System Nodal Analysis:** Calculates and plots the Inflow Performance Relationship (IPR) using a composite straight-line/Vogel model and intersects it with the Vertical Lift Performance (VLP) curve to find the operating point.
* **Dynamic Wellbore Schematic:** A highly visual, interactive 2D schematic that renders up to 4 casing strings, tubing, packer placement, and perforations. Includes dynamic flow arrows that adapt based on whether the packer is set or unset.
* **Flow Assurance Profiling:** Maps pressure and temperature gradients from bottomhole to wellhead. Automatically flags Hydrate Formation risks and API RP 14E Erosional Velocity limits.
* **Sensitivity Analysis:** Batch overlay VLP curves by varying key parameters like Tubing ID, Water Cut, or Wellhead Pressure.
* **Black Oil PVT Engine:** Dynamically calculates fluid properties, phase behaviors, and densities using industry-standard empirical correlations at local pressure and temperature steps.

## Tech Stack

* **Frontend:** Streamlit
* **Visualization:** Plotly
* **Computations:** NumPy, SciPy, math

## Installation & Setup

1. **Clone the repository** (or download the source code):
   ```bash
   git clone [https://github.com/yourusername/well-performance-simulator.git](https://github.com/yourusername/well-performance-simulator.git)
   cd well-performance-simulator
