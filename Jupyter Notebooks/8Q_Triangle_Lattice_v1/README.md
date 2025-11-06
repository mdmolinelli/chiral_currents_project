# 8Q Triangle Lattice v1

This directory contains simulations and analysis tools for an 8-qubit triangle lattice quantum system. The project focuses on current measurements, quantum state dynamics, and various calibration procedures.

## Directory Structure

## Notebook Descriptions

### Current Measurements Directory

- **[`sympy_eigenstate_analysis.ipynb`](current_measurements/sympy_eigenstate_analysis.ipynb)**: Symbolic analysis of eigenstates using SymPy for theoretical understanding of the triangle lattice system and current flow patterns.

- **[`current_measurement_analysis_fits.ipynb`](current_measurements/current_measurement_analysis_fits.ipynb)**: Comprehensive analysis and fitting of current measurement data including single-shot measurements, population dynamics, and covariance analysis. Performs parameter optimization to match theoretical simulations with experimental data.

- **[`current_measurement_analysis_reduced.ipynb`](current_measurements/current_measurement_analysis_reduced.ipynb)**: Simulations of current correlation beamsplitter measurements on the reduced density matrix of the 4 qubits involved in the measurement. We can still prepare the eigenstate of the full 8 qubit system. This is useful for including effects like T1 decay or non-adiabatic ramp effects that cause population of states other than 4 particles.

- **[`current_measurement_analysis_particle_sector.ipynb`](current_measurements/current_measurement_analysis_reduced.ipynb)**: Simulations of current correlation beamsplitter measurements restricted to only states with a certain (4) particle number. This is useful for simulating the full 8 qubit system with only considering effects that maintain particle number, such as dephasing.


- **[`post_selection.ipynb`](current_measurements/post_selection.ipynb)**: Post-selection analysis of measurement data based on particle number conservation. Analyzes current correlations between different qubit pairs after filtering for specific particle number states in the 8-qubit triangle lattice.

- **[`error_bars.ipynb`](current_measurements/error_bars.ipynb)**: Statistical analysis and error estimation for current correlation measurements using bootstrap methods. Computes confidence intervals and error bars for order parameter measurements with readout correction.

- **[`offset_sweep_measurement_analysis.ipynb`](current_measurements/offset_sweep_measurement_analysis.ipynb)**: Analysis of offset sweep measurements for characterizing phase control and timing effects in qubit swap operations.

### Current Measurement Data Processing

- **[`correlation_data_processing.ipynb`](current_measurement_data_processing/correlation_data_processing.ipynb)**: Processes experimental correlation data to extract current measurements and analyze quantum correlations between qubits in the triangle lattice.

### Adiabatic Ramps

- **`ramp_dynamics_simulations.ipynb`**: Simulates adiabatic ramping protocols for preparing desired quantum states in the 8-qubit triangle lattice system.

## Key Features

- **Current Measurement Simulations**: Comprehensive simulation framework for analyzing current flows in the triangle lattice geometry
- **Calibration Tools**: Automated calibration procedures for experimental measurements
- **Data Processing Pipeline**: Tools for processing and analyzing experimental data
- **State Preparation**: Adiabatic ramping protocols for quantum state initialization
- **Theoretical Analysis**: Symbolic computation tools for eigenstate analysis

## Source Code Modules

- **`src_current_measurement.py`**: Core classes for current measurement calibration and data acquisition
- **`src_current_measurement_simulations.py`**: Simulation engine for current measurement protocols including the `CurrentMeasurementSimulation` class
- **`src_current_measurement_fits.py`**: Fitting routines and analysis tools for extracting physical parameters from measurement data

## Usage

The main entry points are the Jupyter notebooks in the root directory. Start with `current_simulations.ipynb` for an overview of the current measurement capabilities, or `current_calibration.ipynb` for calibration procedures.

## Dependencies

- NumPy
- Matplotlib  
- QuTiP (Quantum Toolbox in Python)
- SciPy
- SymPy (for symbolic analysis)
- H5PY (for data file handling)