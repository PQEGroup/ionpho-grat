# Ion photonics grating 3D simulation

This repository contains simulation code for photonic grating structures using [Tidy3D](https://github.com/flexcompute/tidy3d). It focuses on full 3D simulation of photonic gratings for applications in quantum photonics and ion trapping systems.

## Features

- Single and dual-layer grating simulations
- GDS file import capability for custom grating designs
- MATLAB file import for grating patterns
- Far-field radiation pattern calculation and visualization
- Support for substrate configurations - Si or metal
- Configurable simulation parameters via JSON files

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ionpho-grat-sim
   ```

2. Install required packages:
   ```bash
   pip install tidy3d numpy matplotlib scipy h5py gdspy gdstk
   ```

3. Verify your installation by running one of the example simulations


To test the repository and functionality, and to run the examples, a free tidy3d account is sufficient that should give you 10 flex credits. However, for active use, a group account is recommended since the number of flex credits needed can build.

To give an estimate of cost, typical costs for red wavelength grating simulations are 1-2 flex credits (can be more depending on the layer stackup) and 4-6 flex credits for blue wavelength grating simulations (for diverging beams). Tight focus gratings less than 1.5um can be expensive - even around 30-50 flex credits for a simulation.

The best way to use the code is to first test the simulation setup without running it (without passing the '--run' argument), and that will show an estimate of the cost, and some plots showing how the simulation is setup to validate everything is correct.



## Usage

Simulations can be run in two ways - one is by using a mat file exported by our grating design scripts (that repository will be available soon). This is run using the loadMatFile.py file. The second way is by directly importing a gds file of the grating using the loadGdsFile.py file. A set of json configuration files are a convenient way to specify the parameters of the layer stackup and the simulation settings. For a gds file import simulation, another json file specifying the details of the cell to import, and some basic parameters necessary, including the taper edge width, the beam emission angle and the wavelength of the simulation. The angle specified should be roughly accurate, since it is used in generating the relevant plot ranges when viewing the simulation results.


### Configuration

Simulations are configured using JSON files with the following structure:

1. "material_stack.json": Has the details of the thicknesses of the different core/cladding thicknesses, and whether it is a single/dual layer waveguide. It also contains the indices of the core/clad/substrate at particular wavelengths. This sets the indeices of the structures and since the gratings are for a particular wavelength, the structures are described by non dispersive media in the simulation and the simulation monitors are single wavelength.

2. "simulation_config.json": Another necessary json file - that specifies the simulation mesh/time/bottom boundary condition. It also has details on what the target ion height is such that relevant plots can be generated about that. Based on that, there are parameters to create monitors that will visualize the field at different planes above the trap surface.

3. "gds_config.json": This is needed only if using the loadGdsFile.py file to run the simulation. It has information that will be used to load the relevant cell, and setup the simulation wavelength and monitor locations (based on the intended design angle). It also will specify the width of the waveguide at the edge of the grating, since the simulation adds a small feeder waveguide.

### Running Simulations

To run via mat files:

```bash
python loadMatFile.py <mat_file_path> <material_file> <config_file> [--run]
```

```bash
python loadMatFile.py example/example_grat1_single/perDC_interp_initial_729_w0x5p5_w0y2p5_thet60_z0_1um_tox6um_v73.mat example/example_grat1_single/SiN_single.json example/example_grat1_single/simulation_config.json
```

(add the --run flag to actualy run the simulation)

To run via gds import

```bash
python loadGdsFile.py <gds_file_path> <material_file> <config_file> <gds_config_file> [--run]
```
```bash
python loadGdsFile.py example/example_grat3_import_gds/2018_03_16_854-866_wgwi0.55.gds example/example_grat3_import_gds/SiN_single.json example/example_grat3_import_gds/simulation_config.json example/example_grat3_import_gds/gds_config_example.json
```

(add the --run flag to actualy run the simulation)

### Visualizing Results

After running a simulation, you can visualize the far-field radiation patterns:

```bash
python plot_farfield.py <simulation_output_file>
```

This will create a few plots. First one is intended to focus on the beam spot, with varying sizes of the XY region about the spot. For the smallest window to contain the beam spot, in the import gds case, the angle specified needs to be roughly accurate. As backups, larger window sizes are also shown. This may just lead to a small issue for tight focus gratings, where the resolution of points would need to be increases. In the example the resolution is 101 points.

The XZ plane plots are a work in progress, and not final yet.

There is a plot of far field projections at different distances over the whole theta phi space, that hopefully will give a good idea about stray scatter/higher order emission.

Fields at different z heights are aggregated to show a gif of the beam profile evolving at different z heights.

Last plot is the power radiation efficiency of the device.

## File Structure

- `simulation_setup.py`: Core simulation setup and execution
- `plot_farfield.py`: Far-field analysis and visualization
- `loadGdsFile.py`: GDS file import functionality
- `loadMatFile.py`: MATLAB file import functionality
- `grating_structures.py`: Definition of grating structure geometries
- `load_material.py`: Material property handling
- `plot_helpers.py`: Helper functions for plotting
- `convert_mat_to_v73.py`: Convert MATLAB files to v7.3 format

## Examples

The repository includes three main examples:

1. `example_grat1_single`: Single-layer grating simulation
2. `example_grat2_dual`: Dual-layer grating simulation
3. `example_grat3_import_gds`: GDS file import example

Each example includes its own configuration files and demonstrates different aspects of the simulation framework.


## Contributors
Vighnesh Natarajan, Gillenhaal Beck

PQE group, TIQI group

Please contact Vighnesh Natarajan (vn95@cornell.edu) with any questions/if any issues arise
