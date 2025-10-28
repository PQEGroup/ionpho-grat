# Ion Photonics Grating Toolkit

A comprehensive toolkit for designing, simulating, and analyzing photonic grating couplers for quantum photonics and ion trapping applications. This repository provides end-to-end functionality from look-up table generation through grating design to full 3D electromagnetic simulation.

The design process closely follows the paper - G. J. Beck, J. P. Home and K. K. Mehta, "Grating Design Methodology for Tailored Free-Space Beam-Forming," *Journal of Lightwave Technology*, vol. 42, no. 14, pp. 4939-4951, 15 July 2024. [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10479983](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10479983)

## Repository Structure

### ionpho-grat-LUT
Tools for generating look-up tables (LUTs) through 2D electromagnetic simulations of grating structures. Performs sweeps over grating period and duty cycle to characterize emission angles and coupling strengths. These LUTs form the foundation for the grating design process.

**Key Features:**
- Single and sweep 2D grating simulations using Tidy3D
- Material stack configuration via JSON
- Extraction of grating strength, emission angle, and effective index
- CSV and MATLAB output formats

### ionpho-grat-design
Modular grating design framework that uses LUTs to create optimized beam-forming gratings with tailored focal properties. Implements both longitudinal (apodization) and transverse (curvature) design methodologies.

**Key Features:**
- Longitudinal apodization design for specified beam waists and emission angles
- Transverse design for curved grating teeth
- Optional 2D Tidy3D validation simulations
- JSON-based configuration
- MATLAB file export for simulation

### ionpho-grat-sim
Full 3D electromagnetic simulation suite for validating complete grating designs. Supports GDS file import and provides comprehensive far-field analysis.

**Key Features:**
- Full 3D Tidy3D simulations
- GDS or MATLAB file import
- Single and dual-layer grating support
- Far-field radiation pattern analysis
- Configurable substrates (Si or metal)
- Exports final GDS file

## Quick Start

Each subdirectory contains its own detailed README with specific installation instructions and usage examples. The typical workflow is:

1. **Generate LUT**: Use `ionpho-grat-LUT` to create a look-up table for your material stack and wavelength
2. **Design Grating**: Use `ionpho-grat-design` with the LUT to design a grating meeting your specifications
3. **Validate Design**: Use `ionpho-grat-sim` to run full 3D simulations of the final design

## Requirements

- Python 3.x
- Tidy3D (requires Flexcompute account)
- NumPy, SciPy, Matplotlib
- Additional dependencies listed in individual subdirectories

## Contributors

**Authors:**
- Vighnesh Natarajan
- Gillenhaal Beck
- Jay Sun

**Research Groups:**
- PQE group (Photonics and quantum electronics), Cornell University
- TIQI group (Trapped ion quantum information), ETH Zurich

## License

This project is licensed under the MIT License

## Contact

For questions or issues, please contact Vighnesh Natarajan at vn95@cornell.edu

