import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import time
from tidy3d import web
import pandas as pd
from tidy3d import material_library

# Import material helper functions
sys.path.append(os.path.dirname(__file__))
from material_helper import load_json, get_medium_from_index, get_material_params, load_material_data, get_medium_from_csv

def getfilename(dirname, wavelength, material, perStr, DCStr, l_grat, boxt, wghb, wght, inth, shift_wgs, toxt, N=None):
    """Generate a filename based on the simulation parameters"""
    # Generate filename with common parameters
    print("GET FILENAME ", wavelength)
    if wavelength <= 0:
        wavelength_int = 0
    else:
        wavelength_int = int(round(wavelength * 1e3))  # Convert to nm and round
    
    boxt_int = int(round(boxt * 1e3))  # Convert to nm and round
    wghb_int = int(round(wghb * 1e3))  # Convert to nm and round
    toxt_int = int(round(toxt * 1e3))  # Convert to nm and round
    
    # Create period string with optional number of periods
    l_grat_str = f"{l_grat:.1f}"  # Format to one decimal place
    period_str = f"_N{N}" if N else ""  # Only include if N is specified

    # Handle dual waveguide configuration
    if wght > 0 and inth > 0:
        wght_int = int(round(wght * 1e3))  # Convert to nm and round
        inth_int = int(round(inth * 1e3))  # Convert to nm and round
        return os.path.join(dirname, f"{material}_lam{wavelength_int}nm_per{perStr}nm_pertDC{DCStr}{period_str}_lgrat{l_grat_str}um_boxt{boxt_int}nm_wghb{wghb_int}nm_wght{wght_int}nm_inth{inth_int}nm_toxt{toxt_int}nm")
    else:
        # Single waveguide configuration
        return os.path.join(dirname, f"{material}_lam{wavelength_int}nm_per{perStr}nm_pertDC{DCStr}{period_str}_lgrat{l_grat_str}um_boxt{boxt_int}nm_wgh{wghb_int}nm_toxt{toxt_int}nm")

def create_grating_structure(gratper, gratDC, N, l_pre, l_post, material, wg_w, wghb, boxt, toxt, tSi, 
                           material_file=None, csv_file=None, oxide_csv_file=None, oxide_index=None, 
                           max_num_poles=1, is_dual_waveguide=False, wght=None, inth=None, use_pec_bottom=False):
    """Create a grating structure with the specified parameters
    
    Args:
        gratper: Grating period in microns
        gratDC: Grating duty cycle (0-1)
        N: Number of grating periods
        l_pre: Pre-grating length in microns
        l_post: Post-grating length in microns
        material: Material name for the waveguide
        wg_w: Waveguide width in microns
        wghb: Bottom waveguide height in microns
        boxt: Bottom oxide thickness in microns
        toxt: Top oxide thickness in microns
        tSi: Silicon thickness in microns
        material_file: Path to material JSON file (default: None)
        csv_file: Path to CSV file with waveguide refractive index data (default: None)
        oxide_csv_file: Path to CSV file with oxide refractive index data (default: None)
        oxide_index: Fixed refractive index for oxide if CSV file not provided (default: None)
        max_num_poles: Maximum number of poles to use in dispersion fitting (default: 1)
        is_dual_waveguide: Whether to create a dual waveguide structure (default: False)
        wght: Top waveguide height for dual waveguide (default: None)
        inth: Intermediate layer height for dual waveguide (default: None)
        use_pec_bottom: Whether to use PEC boundary condition at bottom instead of silicon substrate (default: False)
        
    Returns:
        structures: List of structure objects
        l_grat: Total grating length
        oxide_source: Description of the oxide medium source
    """
    # Calculate grating length
    l_grat = gratper * N
    
    # Create structures list
    structures = []

    extend_into_sim_boundary = 5 # um
    
    # Create oxide medium based on provided parameters
    if oxide_csv_file and os.path.exists(oxide_csv_file):
        # Create oxide medium from CSV file
        print(f"Creating dispersive medium for oxide from CSV data: {oxide_csv_file}")
        mat_oxide = get_medium_from_csv(oxide_csv_file, max_num_poles=max_num_poles)
        oxide_source = f"CSV file ({os.path.basename(oxide_csv_file)}) with {max_num_poles} poles"
    elif oxide_index is not None:
        # Use fixed index
        print(f"Creating fixed index medium for oxide with n = {oxide_index}")
        mat_oxide = td.Medium(permittivity=oxide_index**2)
        oxide_source = f"Fixed index (n = {oxide_index})"
    else:
        # Default to Tidy3D material library
        print("Using default Tidy3D SiO2 (Horiba) for oxide material")
        mat_oxide = material_library['SiO2']['Horiba']
        oxide_source = "Tidy3D material library (SiO2 Horiba)"
    
    print(f"Oxide medium source: {oxide_source}")
    
    # Total waveguide stack height
    if is_dual_waveguide:
        if wght is None or inth is None:
            raise ValueError("For dual waveguide configuration, both wght and inth must be specified")
        total_wg_height = wghb + inth + wght
    else:
        total_wg_height = wghb
    
    # Oxide layer
    oxide = td.Box.from_bounds(
        rmin=(-l_pre - extend_into_sim_boundary, -wg_w/2, -boxt),
        rmax=(l_grat + l_post + extend_into_sim_boundary, wg_w/2, toxt + total_wg_height)
    )
    structures.append(td.Structure(geometry=oxide, medium=mat_oxide))
    
    mat_wg = get_material_medium(material, material_file=material_file, dispersion_csv=csv_file, max_num_poles=max_num_poles)

    # Pre-grating waveguide(s)
    pre_wg_bottom = td.Box.from_bounds(
        rmin=(-l_pre - extend_into_sim_boundary, -wg_w/2, 0),
        rmax=(0, wg_w/2, wghb)
    )
    structures.append(td.Structure(geometry=pre_wg_bottom, medium=mat_wg))
    
    # Add top waveguide and intermediate layer for dual waveguide configuration
    if is_dual_waveguide:
        pre_wg_top = td.Box.from_bounds(
            rmin=(-l_pre - extend_into_sim_boundary, -wg_w/2, wghb + inth),
            rmax=(0, wg_w/2, wghb + inth + wght)
        )
        structures.append(td.Structure(geometry=pre_wg_top, medium=mat_wg))
    
    # Grating structure(s)
    for i in range(N):
        # Bottom waveguide grating
        grating_box_bottom = td.Box.from_bounds(
            rmin=(gratper*i, -wg_w/2, 0),
            rmax=(gratper*i + gratDC*gratper, wg_w/2, wghb)
        )
        structures.append(td.Structure(geometry=grating_box_bottom, medium=mat_wg))
        
        # Top waveguide grating for dual waveguide configuration
        if is_dual_waveguide:
            grating_box_top = td.Box.from_bounds(
                rmin=(gratper*i, -wg_w/2, wghb + inth),
                rmax=(gratper*i + gratDC*gratper, wg_w/2, wghb + inth + wght)
            )
            structures.append(td.Structure(geometry=grating_box_top, medium=mat_wg))
    
    # Post-grating waveguide(s)
    post_wg_bottom = td.Box.from_bounds(
        rmin=(N*gratper, -wg_w/2, 0),
        rmax=(l_grat + l_post + extend_into_sim_boundary, wg_w/2, wghb)
    )
    structures.append(td.Structure(geometry=post_wg_bottom, medium=mat_wg))
    
    # Add top waveguide for dual waveguide configuration
    if is_dual_waveguide:
        post_wg_top = td.Box.from_bounds(
            rmin=(N*gratper, -wg_w/2, wghb + inth),
            rmax=(l_grat + l_post + extend_into_sim_boundary, wg_w/2, wghb + inth + wght)
        )
        structures.append(td.Structure(geometry=post_wg_top, medium=mat_wg))
    
    # Silicon substrate (only if not using PEC bottom)
    if not use_pec_bottom:
        silicon = td.Box.from_bounds(
            rmin=(-l_pre - extend_into_sim_boundary, -wg_w/2, -boxt - tSi-extend_into_sim_boundary),
            rmax=(l_grat + l_post + extend_into_sim_boundary, wg_w/2, -boxt)
        )
        structures.append(td.Structure(geometry=silicon, medium=material_library['cSi']['Green2008']))
    
    return structures, l_grat, oxide_source

def get_material_medium(material, wavelength=None, material_file=None, dispersion_csv=None, max_num_poles=1):
    """Get material medium based on material name and dispersion data
    
    Creates a dispersive medium from the CSV file containing wavelength and 
    refractive index data. If material_file is provided, the CSV path will be
    extracted from it.
    
    Args:
        material: Material name
        wavelength: Specific wavelength in nm (not used, kept for backward compatibility)
        material_file: Path to material JSON file (required unless dispersion_csv is provided directly)
        dispersion_csv: Path to CSV file with wavelength and refractive index data (required, 
                       can be specified directly or via material_file)
        max_num_poles: Maximum number of poles to use in dispersion fitting (default: 1)
        
    Returns:
        td.Medium: A medium object with the appropriate refractive index properties
    
    Raises:
        ValueError: If neither material_file nor dispersion_csv is provided
    """
    import os
    from material_helper import load_material_data, get_medium_from_csv
    
    # Check if we have a dispersion CSV file path
    if dispersion_csv is None and material_file is None:
        error_msg = (f"Error: Neither material_file nor dispersion_csv is provided for material '{material}'.\n"
                    f"Please provide either:\n"
                    f"1. A path to a material JSON file with a 'csv_file' field using the material_file parameter, or\n"
                    f"2. A direct path to a CSV file with wavelength and refractive index data using the dispersion_csv parameter.\n\n"
                    f"The material JSON file MUST contain a 'csv_file' field pointing to a CSV file with dispersion data.\n"
                    f"The CSV file should have columns for wavelength (nm), n, and optionally k. First row is skipped as header.\n\n"
                    f"Example material JSON format:\n"
                    f"  {{\n"
                    f"    \"process_name\": \"AIM_SL1\",\n"
                    f"    \"material_name\": \"AO_AIM\",\n"
                    f"    \"csv_file\": \"AO_AIM.csv\",\n"
                    f"    \"layer_stack\": {{ ... }}\n"
                    f"  }}")
        print(error_msg)
        raise ValueError("Either material_file or dispersion_csv is required")
    
    # If material_file is provided but dispersion_csv is not, get the CSV path from the material file
    if dispersion_csv is None and material_file is not None:
        try:
            material_data = load_material_data(material_file)
            
            # Check for csv_file field in material data
            if "csv_file" not in material_data:
                print(f"Error: No 'csv_file' field in material file: {material_file}")
                raise ValueError("Material file must contain a 'csv_file' field with path to CSV file")
            
            # Get the CSV file path from the material file
            csv_path = material_data["csv_file"]
            
            # If CSV path is relative, resolve it relative to the material_file directory
            if not os.path.isabs(csv_path):
                material_dir = os.path.dirname(os.path.abspath(material_file))
                csv_path = os.path.join(material_dir, csv_path)
            
            dispersion_csv = csv_path
            print(f"Using dispersion CSV file from material file: {dispersion_csv}")
            
        except Exception as e:
            print(f"Error getting CSV file from material file: {str(e)}")
            raise
    
    # Make sure the CSV file exists
    if not os.path.exists(dispersion_csv):
        print(f"Error: Dispersion CSV file not found: {dispersion_csv}")
        raise FileNotFoundError(f"Dispersion CSV file not found: {dispersion_csv}")
    
    # Create medium from the CSV file using the updated implementation
    print(f"Creating dispersive medium for material '{material}' from CSV data with max {max_num_poles} poles")
    try:
        # Use the max_num_poles parameter when calling get_medium_from_csv
        return get_medium_from_csv(dispersion_csv, max_num_poles=max_num_poles)
    except Exception as e:
        print(f"Error creating dispersive medium: {str(e)}")
        print(f"Please check that the CSV file format is correct and contains columns for wavelength (nm), n, and optionally k")
        print(f"First row should be a header row with column names")
        raise

def create_monitors(l_pre, l_grat, l_post, boxt, toxt, wghb, wg_w, tair, freqs, include_2d_monitor=False, 
                  is_dual_waveguide=False, wght=None, inth=None):
    """Create monitors for the simulation
    
    Args:
        l_pre: Pre-grating length
        l_grat: Grating length
        l_post: Post-grating length
        boxt: Bottom oxide thickness
        toxt: Top oxide thickness
        wghb: Bottom waveguide height
        wg_w: Waveguide width
        tair: Air thickness
        freqs: List of frequencies to monitor
        include_2d_monitor: Whether to include a 2D XZ field monitor (can increase memory usage)
        is_dual_waveguide: Whether this is a dual waveguide configuration
        wght: Top waveguide height (for dual waveguide)
        inth: Intermediate layer height (for dual waveguide)
    
    Returns:
        List of monitor objects
    """
    monitors = []
    
    # Calculate total waveguide height for monitor positioning
    if is_dual_waveguide:
        if wght is None or inth is None:
            raise ValueError("For dual waveguide configuration, both wght and inth must be specified")
        total_wg_height = wghb + inth + wght
    else:
        total_wg_height = wghb
    
    # Add a 2D field monitor for the entire X-Z plane (optional)
    if include_2d_monitor:
        field_monitor_2d = td.FieldMonitor(
            center=[l_grat/2, 0, 0],
            size=[l_grat + l_pre + l_post, 0, boxt + toxt + total_wg_height + 2],
            freqs=freqs,
            name="field_xz",
            fields=['E']
        )
        monitors.append(field_monitor_2d)
    
    # Flux monitor above the grating to capture upward emission
    flux_air = td.FluxMonitor(
        center=[l_grat/2, 0, toxt + total_wg_height + tair/2],
        size=[l_grat + l_pre + l_post, 2*wg_w, 0],
        freqs=freqs,
        name="flux_air"
    )
    monitors.append(flux_air)
    
    # Flux monitor below the grating to capture downward emission
    flux_si = td.FluxMonitor(
        center=[l_grat/2, 0, -boxt/4],
        size=[l_grat + l_pre + l_post, 2*wg_w, 0],
        freqs=freqs,
        name="flux_si"
    )
    monitors.append(flux_si)
    
    # Flux monitor at the start of the waveguide to measure input power
    flux_start = td.FluxMonitor(
        center=[-l_pre/2, 0, total_wg_height/2],
        size=[0, 2*wg_w, total_wg_height*4],
        freqs=freqs,
        name="flux_start"
    )
    monitors.append(flux_start)
    
    # Flux monitor at the end of the waveguide to measure transmitted power
    flux_end = td.FluxMonitor(
        center=[l_grat + l_post/2, 0, total_wg_height/2],
        size=[0, 2*wg_w, total_wg_height*4],
        freqs=freqs,
        name="flux_end"
    )
    monitors.append(flux_end)
    
    # Field monitor in the air above the grating
    field_top = td.FieldMonitor(
        center=[l_grat/2, 0, toxt + total_wg_height + tair/2],
        size=[l_grat + l_pre + l_post, 0, 0],
        freqs=freqs,
        name="field_top"
    )
    monitors.append(field_top)
    
    # Field monitor in the waveguide
    field_wg = td.FieldMonitor(
        center=[l_grat/2, 0, total_wg_height/2],
        size=[l_grat + l_pre + l_post, 0, 0],
        freqs=freqs,
        name="field_wg"
    )
    monitors.append(field_wg)
    
    # Field monitor for effective index calculation
    field_wg_neff = td.FieldMonitor(
        center=[0.4*l_grat, 0, total_wg_height/2],
        size=[0.8*l_grat, 0, 0],
        freqs=freqs,
        name="field_wg_neff"
    )
    monitors.append(field_wg_neff)
    
    return monitors

def create_farfield_monitor(l_grat, l_pre, l_post, toxt, tair, wg_w, freqs, 
                           is_dual_waveguide=False, wghb=None, wght=None, inth=None):
    """Create a far-field monitor for the simulation"""
    # Calculate total waveguide height for monitor positioning
    if is_dual_waveguide:
        if wght is None or inth is None or wghb is None:
            raise ValueError("For dual waveguide configuration, wghb, wght, and inth must be specified")
        total_wg_height = wghb + inth + wght
    else:
        total_wg_height = wghb if wghb is not None else 0.13  # Default if not specified

    theta_angles = np.linspace(-np.pi/2, np.pi/2, 1001)
    proj_z = toxt + total_wg_height + tair/2
    
    # Create far-field monitor
    angle_monitor = td.FieldProjectionAngleMonitor(
        center=[l_grat/2, 0, toxt + total_wg_height + tair/2],
        size=[l_grat + l_pre + l_post, 50*wg_w, 0],
        freqs=freqs,
        name="far_field",
        theta=list(theta_angles),
        phi=[0],
        custom_origin=(l_grat/2, 0, proj_z),
        proj_distance=1e6,  # microns
        far_field_approx=True
    )
    return angle_monitor

def create_mode_source(sim, l_pre, wghb, boxt, toxt, wg_w, freqs, 
                      is_dual_waveguide=False, wght=None, inth=None):
    """Create a mode source for the simulation"""
    # Calculate total waveguide height and mode source position
    if is_dual_waveguide:
        if wght is None or inth is None:
            raise ValueError("For dual waveguide configuration, both wght and inth must be specified")
        total_wg_height = wghb + inth + wght
        # Place mode source at the center of the bottom waveguide since that's typically the input
        mode_center_z = wghb/2
    else:
        total_wg_height = wghb
        mode_center_z = wghb/2
    
    # Create a box for the mode source
    src_box = td.Box(
        center=[-l_pre*0.9, 0, mode_center_z],
        size=[0, 10*wg_w, boxt+toxt/2]
    )
    
    # Calculate the mean frequency for the source
    mean_freq = np.mean(freqs)
    
    # Create mode solver
    mode_spec = td.ModeSpec(num_modes=1)
    from tidy3d.plugins.mode import ModeSolver
    mode_solver = ModeSolver(
        simulation=sim,
        plane=src_box,
        mode_spec=mode_spec,
        freqs=[mean_freq]  # Use the mean frequency
    )
    
    return mode_solver, src_box

def create_simulation(structures, monitors, l_grat, l_pre, l_post, boxt, toxt, wghb, tair, tSi, run_time, 
                     grid_dl=0.01, use_pml=True, absorber_layers=60, 
                     is_dual_waveguide=False, wght=None, inth=None, use_pec_bottom=False):
    """Create the simulation object with all components
    
    Args:
        structures: List of structure objects
        monitors: List of monitor objects
        l_grat: Grating length
        l_pre: Pre-grating length
        l_post: Post-grating length
        boxt: Bottom oxide thickness
        toxt: Top oxide thickness
        wghb: Bottom waveguide height
        tair: Air thickness
        tSi: Silicon thickness
        run_time: Simulation run time
        grid_dl: Grid spacing for simulation mesh (default: 0.01)
        use_pml: Whether to use PML boundary conditions for X direction (default: True)
                If False, uses absorber boundary conditions instead
        absorber_layers: Number of absorber layers if use_pml is False (default: 60)
        is_dual_waveguide: Whether this is a dual waveguide configuration (default: False)
        wght: Top waveguide height (for dual waveguide)
        inth: Intermediate layer height (for dual waveguide)
        use_pec_bottom: Whether to use PEC boundary condition at bottom instead of silicon substrate (default: False)
        
    Returns:
        Simulation object
    """
    # Calculate total waveguide height
    if is_dual_waveguide:
        if wght is None or inth is None:
            raise ValueError("For dual waveguide configuration, both wght and inth must be specified")
        total_wg_height = wghb + inth + wght
    else:
        total_wg_height = wghb
    
    # Calculate simulation size and bounds
    sim_size_x = l_grat + l_pre + l_post
    sim_size_y = 0  # 2D simulation along X-Z
    if use_pec_bottom:
        sim_size_z = boxt + toxt + total_wg_height + tair
        sim_center_z = (toxt + total_wg_height + tair - boxt) / 2
    else:
        sim_size_z = boxt + toxt + total_wg_height + tair + tSi
        sim_center_z = (toxt + total_wg_height + tair - boxt - tSi) / 2
    
    # Center of simulation
    sim_center_x = l_grat / 2
    sim_center_y = 0
    
    # Create boundary spec based on use_pml parameter and PEC bottom option
    if use_pec_bottom:
        print("Using PEC boundary condition at bottom surface")
        if use_pml:
            print("Using PML boundary conditions for X direction")
            boundary_spec = td.BoundarySpec(
                x = td.Boundary.pml(),
                y = td.Boundary.periodic(),
                z = td.Boundary(minus=td.PECBoundary(), plus=td.PML())
            )
        else:
            print(f"Using absorber boundary conditions with {absorber_layers} layers for X direction")
            print("This may reduce simulation divergence issues but might cause higher reflections")
            boundary_spec = td.BoundarySpec(
                x = td.Boundary(plus=td.Absorber(num_layers=absorber_layers), 
                               minus=td.Absorber(num_layers=absorber_layers)), 
                y = td.Boundary.periodic(), 
                z = td.Boundary(minus=td.PECBoundary(), plus=td.PML())
            )
    else:
        if use_pml:
            print("Using PML boundary conditions for X and Z directions")
            boundary_spec = td.BoundarySpec.pml(x=True, z=True)
        else:
            print(f"Using absorber boundary conditions with {absorber_layers} layers for X direction")
            print("This may reduce simulation divergence issues but might cause higher reflections")
            boundary_spec = td.BoundarySpec(
                x = td.Boundary(plus=td.Absorber(num_layers=absorber_layers), 
                               minus=td.Absorber(num_layers=absorber_layers)), 
                y = td.Boundary.periodic(), 
                z = td.Boundary.pml()
            )

    # Create simulation with calculated parameters
    sim = td.Simulation(
        size=[sim_size_x, sim_size_y, sim_size_z],
        center=[sim_center_x, sim_center_y, sim_center_z],
        structures=structures,
        monitors=monitors,
        grid_spec=td.GridSpec.uniform(dl=grid_dl),
        boundary_spec=boundary_spec,
        run_time=run_time
    )
    
    return sim

def add_source_to_simulation(sim, mode_solver, freqs):
    """Add a source to an existing simulation"""
    # Calculate the mean frequency for the source
    mean_freq = np.mean(freqs)
    
    # Create mode source using mean frequency
    source_time = td.GaussianPulse(freq0=mean_freq, fwidth=0.1*mean_freq)
    mode_source = mode_solver.to_source(mode_index=0, direction="+", source_time=source_time)
    
    # Update simulation with source
    return sim.copy(update=dict(sources=[mode_source]))

def setup_simulation(material, material_file, gratper, gratDC, N, l_pre, l_post, 
                    wg_w, boxt, toxt, tSi, tair, freqs, run_time, include_2d_monitor=False, 
                    grid_dl=0.01, csv_file=None, oxide_csv_file=None, oxide_index=None, max_num_poles=1,
                    use_pml=True, absorber_layers=60, wghb=None, wght=None, inth=None, use_pec_bottom=False):
    """Complete setup function that creates a simulation with all components
    
    Args:
        material: Material name for the waveguide
        material_file: Path to material file
        gratper: Grating period
        gratDC: Grating duty cycle
        N: Number of grating periods
        l_pre: Pre-grating length
        l_post: Post-grating length
        wg_w: Waveguide width
        boxt: Bottom oxide thickness
        toxt: Top oxide thickness
        tSi: Silicon thickness
        tair: Air thickness
        freqs: List of frequencies
        run_time: Simulation run time
        include_2d_monitor: Whether to include a 2D XZ field monitor (default: False)
        grid_dl: Grid spacing for simulation (default: 0.01)
        csv_file: Path to CSV file with waveguide refractive index data (default: None)
        oxide_csv_file: Path to CSV file with oxide refractive index data (default: None)
        oxide_index: Fixed refractive index for oxide if CSV file not provided (default: None)
        max_num_poles: Maximum number of poles to use in dispersion fitting (default: 1)
        use_pml: Whether to use PML boundary conditions for X direction (default: True)
               If False, uses absorber boundary conditions instead
        absorber_layers: Number of absorber layers if use_pml is False (default: 60)
        wghb: Bottom waveguide height (required)
        wght: Top waveguide height (for dual waveguide, default: None)
        inth: Intermediate layer height (for dual waveguide, default: None)
        use_pec_bottom: Whether to use PEC boundary condition at bottom instead of silicon substrate (default: False)
    
    Returns:
        Simulation object with source, grating length, mode data, mode solver, and oxide source info
        
    Note:
        This function creates structures with dispersive media that model the wavelength-dependent 
        refractive index of the materials. The create_grating_structure function uses get_material_medium 
        which returns a dispersive medium with the full wavelength dependence of the material.
    """
    # Check if this is a dual waveguide configuration
    is_dual_waveguide = wght is not None and inth is not None and wght > 0 and inth > 0
    
    if wghb is None:
        raise ValueError("Bottom waveguide height (wghb) must be specified")
    
    if is_dual_waveguide:
        print(f"Setting up dual waveguide simulation with:")
        print(f"  Bottom waveguide height (wghb): {wghb} μm")
        print(f"  Top waveguide height (wght): {wght} μm")
        print(f"  Intermediate layer height (inth): {inth} μm")
    else:
        print(f"Setting up single waveguide simulation with height (wghb): {wghb} μm")
    
    # Create structures
    structures, l_grat, oxide_source = create_grating_structure(
        gratper, gratDC, N, l_pre, l_post, material, wg_w, wghb, boxt, toxt, tSi, 
        material_file, csv_file, oxide_csv_file, oxide_index, max_num_poles,
        is_dual_waveguide=is_dual_waveguide, wght=wght, inth=inth, use_pec_bottom=use_pec_bottom
    )
    
    # Create monitors
    monitors = create_monitors(
        l_pre, l_grat, l_post, boxt, toxt, wghb, wg_w, tair, freqs, include_2d_monitor,
        is_dual_waveguide=is_dual_waveguide, wght=wght, inth=inth
    )
    
    # Create far-field monitor
    farfield_monitor = create_farfield_monitor(
        l_grat, l_pre, l_post, toxt, tair, wg_w, freqs,
        is_dual_waveguide=is_dual_waveguide, wghb=wghb, wght=wght, inth=inth
    )
    monitors.append(farfield_monitor)
    
    # Create simulation with the specified grid spacing and boundary conditions
    sim = create_simulation(
        structures, monitors, l_grat, l_pre, l_post, boxt, toxt, wghb, tair, tSi, run_time, 
        grid_dl=grid_dl, use_pml=use_pml, absorber_layers=absorber_layers,
        is_dual_waveguide=is_dual_waveguide, wght=wght, inth=inth, use_pec_bottom=use_pec_bottom
    )
    
    # Create mode source
    mode_solver, src_box = create_mode_source(
        sim, l_pre, wghb, boxt, toxt, wg_w, freqs,
        is_dual_waveguide=is_dual_waveguide, wght=wght, inth=inth
    )
    
    # Solve for modes
    modes = mode_solver.solve()
    
    # Add source to simulation
    sim_with_source = add_source_to_simulation(sim, mode_solver, freqs)
    
    return sim_with_source, l_grat, modes, mode_solver, oxide_source
