# pqePDK-grat Examples

This directory contains example files for using the pqePDK-grat package.

## Sample Configuration (`sample_config.json`)

The `sample_config.json` file provides a starting point for configuring your grating design. This is a copy of a working configuration that you can modify for your specific needs.

### Using the Sample Configuration

To use this sample configuration, run the following command from the root directory of the project:

```bash
python gillen_AIM_new.py --config examples/sample_config.json --lut_file path/to/your/lut_file.mat [--show_plots]
```

## Configuration Parameters

The configuration file includes the following key parameters:

| Parameter | Description |
|-----------|-------------|
| `wavelength` | Wavelength in μm (e.g., 0.397 for 397 nm) |
| `w0x`, `w0y` | Beam waist sizes in x and y directions (μm) |
| `x0`, `y0`, `z0` | Initial beam position coordinates (μm) |
| `tOx` | Oxide thickness (μm) |
| `nPts` | Number of simulation points |
| `thetaIncDegrees` | Incidence angle (degrees) |
| `minSize` | Minimum feature size (μm) |
| `nu` | Coupling efficiency target (0.0-1.0) |
| `clipBeginning`, `clipEnd` | Clipping factors for grating start/end |
| `material` | Material type (e.g., "AO_AIM") |
| `forward_em` | Forward emission mode flag (true/false) |
| `LUTparams` | Additional parameters for the LUT processing |

## LUT Files

The Look-Up Table (LUT) files contain precomputed simulation data for various grating configurations. A typical LUT filename follows this pattern:

```
LUTsweepData_lam397_AO_AIM_boxt3340nm_wghBot130nm_wghTop0nm_inth0nm_toxt3000nm_AIM_SL1_forward.mat
```

You must specify the path to a compatible LUT file when running the grating design script. 