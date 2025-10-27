"""
Tidy3D Simulation Module for pqePDK-grat

This module provides functionality to create and run 2D Tidy3D simulations
based on the longitudinal design results from the grating design process.
"""
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from datetime import datetime

# Type aliases for clarity
Array = np.ndarray


def get_material_medium(material_name: str, 
                     material_index: float) -> td.Medium:
    """
    Get a material medium for the waveguide based on fixed indices provided in the config.
    
    Args:
        material_name: Name of the material (for identification only)
        wavelength: Operating wavelength in μm (not used for fixed indices)
        material_index: Fixed refractive index for the material (required)
        oxide_index: Fixed refractive index for oxide if material is "SiO2" or "oxide" (default: 1.45)
        
    Returns:
        Material medium for Tidy3D simulation
    """
    # Special handling for oxide materials
    print(f"Using fixed refractive index for {material_name}: n = {material_index}")
    return td.Medium(permittivity=material_index**2, name=material_name)


def create_apodized_grating(XwgCenter: Array, desDCxx: Array, desperxx: Array, 
                          wghb: float, wg_width: float, 
                          is_dual_waveguide: bool = False,
                          wght: float = 0.0, 
                          inth: float = 0.0) -> List[td.Structure]:
    """
    Create a list of Tidy3D structures representing the apodized grating
    based on the longitudinal design results. Each tooth will have the correct
    size based on the period and duty cycle at that position.
    
    Args:
        XwgCenter: Waveguide center positions along x-axis
        desDCxx: Duty cycles at each position
        desperxx: Grating periods at each position
        wghb: Bottom waveguide height in μm
        wg_width: Waveguide width in μm
        is_dual_waveguide: Whether this is a dual waveguide stack
        wght: Top waveguide height in μm (for dual waveguide only)
        inth: Intermediate layer height in μm (for dual waveguide only)
        
    Returns:
        List of Tidy3D structures representing the grating
    """
    structures = []
    
    # Filter out regions with zero DC or NaN periods 
    valid_indices = ~np.isnan(desperxx) & (desDCxx > 0)
    
    if not np.any(valid_indices):
        print("Warning: No valid grating regions found.")
        return structures
    
    # Extract valid positions, duty cycles, and periods for interpolation
    valid_positions = XwgCenter[valid_indices]
    valid_dc = desDCxx[valid_indices]
    valid_periods = desperxx[valid_indices]
    
    if len(valid_positions) < 2:
        print("Warning: Insufficient valid grating positions for interpolation.")
        return structures
    
    print(f"Creating grating using interpolation of {len(valid_positions)} control points")
    print(f"Position range: {valid_positions[0]:.2f} to {valid_positions[-1]:.2f} μm")
    print(f"Period range: {np.min(valid_periods):.3f} to {np.max(valid_periods):.3f} μm")
    print(f"Duty cycle range: {np.min(valid_dc):.3f} to {np.max(valid_dc):.3f}")
    
    # Get waveguide material from the simulation parameters
    # This is just a placeholder - will be replaced when added to simulation
    wg_medium = td.Medium(permittivity=2.0)
    
    current_pos = valid_positions[0]
    per_array = []
    dc_array = []
    tooth_lengths = []
    tooth_positions = []
    current_pos_array = []
    
    # Define start and end of grating region
    grating_start = valid_positions[0]
    grating_end = valid_positions[-1]
    
    # Create each individual tooth using interpolation
    x0 = grating_start
    min_tooth_size = 0.001  # Minimum tooth size in μm
    
    while x0 < grating_end:
        # Interpolate period and duty cycle at the current position
        per = np.interp(x0, valid_positions, valid_periods)
        dc = np.interp(x0, valid_positions, valid_dc)
        
        # Calculate tooth dimensions
        tooth_length = max(min_tooth_size, per * dc)
        
        # Position for the tooth center
        pos = x0 + tooth_length/2
        
        # Create teeth structure(s) based on waveguide configuration
        if is_dual_waveguide:
            # Create a box for bottom waveguide tooth
            bottom_tooth = td.Box(
                center=[pos, 0, wghb/2],  # Center of bottom waveguide
                size=[tooth_length, wg_width, wghb]
            )
            structures.append(td.Structure(geometry=bottom_tooth, medium=wg_medium))
            
            # Create a box for top waveguide tooth
            top_tooth = td.Box(
                center=[pos, 0, wghb + inth + wght/2],  # Center of top waveguide
                size=[tooth_length, wg_width, wght]
            )
            structures.append(td.Structure(geometry=top_tooth, medium=wg_medium))
        else:
            # Single waveguide tooth
            tooth = td.Box(
                center=[pos, 0, wghb/2],  # Center of waveguide
                size=[tooth_length, wg_width, wghb]
            )
            structures.append(td.Structure(geometry=tooth, medium=wg_medium))

        per_array.append(per)
        dc_array.append(dc)
        tooth_lengths.append(tooth_length)
        tooth_positions.append(pos)
        current_pos_array.append(current_pos)
        
        # Move to next tooth
        x0 = x0 + per
        current_pos = x0

    print(f"Period array: {[f'{p:.3f}' for p in per_array]}")
    print(f"Duty cycle array: {[f'{d:.3f}' for d in dc_array]}")
    print(f"Tooth lengths: {[f'{t:.3f}' for t in tooth_lengths]}")
    print(f"Tooth positions: {[f'{p:.3f}' for p in tooth_positions]}")
    print(f"Current position array: {[f'{p:.3f}' for p in current_pos_array]}")
    num_waveguides = 2 if is_dual_waveguide else 1
    print(f"Created {len(structures)} grating teeth structures across {num_waveguides} waveguide(s)")
    return structures


def create_simulation(design_results: Dict[str, Any], 
                    p: Dict[str, Any]) -> Tuple[td.Simulation, List[td.Structure]]:
    """
    Create a Tidy3D simulation for the grating coupler based on design results.
    
    Args:
        design_results: Dictionary containing longitudinal design results
        p: Parameters dictionary with simulation settings
        
    Returns:
        Tuple containing the Tidy3D simulation object and a list of structures
    """
    # Check for required parameters
    required_params = ['wavelength', 'material', 'material_index', 'oxide_index']
    missing_params = [param for param in required_params if param not in p]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # Access required parameters
    wavelength = p['wavelength']
    material_name = p['material']
    material_index = p['material_index']
    oxide_index = p['oxide_index']
    
    # Load material stack parameters from JSON file if provided
    material_json = p.get('material_json')
    material_stack = {}
    
    if material_json and os.path.exists(material_json):
        try:
            with open(material_json, 'r') as f:
                material_data = json.load(f)
                if 'layer_stack' in material_data:
                    material_stack = material_data['layer_stack']
                    print(f"Loaded material stack from {material_json}")
        except Exception as e:
            print(f"Error loading material JSON file: {e}")
            print("Using default material stack parameters")
    
    # Check if this is a dual waveguide configuration
    is_dual_waveguide = material_stack.get('is_single_waveguide', True) == False
    if is_dual_waveguide:
        print("Setting up DUAL waveguide configuration")
    else:
        print("Setting up SINGLE waveguide configuration")
        
    # Waveguide dimensions - using the exact names from AO_AIM.json
    wg_width = p.get('wg_width')  # No default
    
    # Use parameter names exactly as in AO_AIM.json
    wghb = material_stack.get('wghb', p.get('wghb'))  # Bottom waveguide height
    wght = material_stack.get('wght', p.get('wght', 0))  # Top waveguide height (0 if not dual)
    inth = material_stack.get('inth', p.get('inth', 0))  # Intermediate layer height
    
    # Structure parameters from material stack - using exact names from AO_AIM.json
    toxt = material_stack.get('toxt', p.get('toxt'))  # Top oxide thickness
    boxt = material_stack.get('boxt', p.get('boxt'))  # Bottom oxide thickness
    tSi = material_stack.get('tSi', p.get('tSi'))  # Silicon thickness
    tair = material_stack.get('tair', p.get('tair'))  # Air thickness
    
    # Check if we have required parameters
    missing_params = []
    if wg_width is None:
        missing_params.append('wg_width')
    if wghb is None:
        missing_params.append('wghb')
    if toxt is None:
        missing_params.append('toxt')
    if boxt is None:
        missing_params.append('boxt')
    if tSi is None:
        missing_params.append('tSi')
    if tair is None:
        missing_params.append('tair')
    
    # For dual waveguide configuration, check additional required parameters
    if is_dual_waveguide:
        if wght is None or wght <= 0:
            missing_params.append('wght')
        if inth is None or inth <= 0:
            missing_params.append('inth')
    
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}. Please provide in simulation config or material stack JSON.")
    
    # Calculate total waveguide height
    if is_dual_waveguide:
        total_wg_height = wghb + inth + wght
        print(f"Dual waveguide stack: bottom={wghb}μm + intermediate={inth}μm + top={wght}μm")
    else:
        total_wg_height = wghb
        print(f"Single waveguide height: {wghb}μm")
    
    print(f"Material stack parameters:")
    print(f"  Bottom waveguide height (wghb): {wghb} μm")
    if is_dual_waveguide:
        print(f"  Top waveguide height (wght): {wght} μm")
        print(f"  Intermediate layer (inth): {inth} μm")
    print(f"  Top oxide thickness (toxt): {toxt} μm")
    print(f"  Box thickness (boxt): {boxt} μm")
    print(f"  Silicon thickness (tSi): {tSi} μm")
    print(f"  Air thickness (tair): {tair} μm")
    
    # Get arrays from design results
    XwgCenter = design_results["XwgCenter"]  # Waveguide center positions
    desDCxx = design_results["desDCxx"]  # Duty cycles
    desperxx = design_results["desperxx"]  # Grating periods
    
    # Calculate simulation boundaries based on the design
    # Find grating extents
    valid_indices = ~np.isnan(desperxx) & (desDCxx > 0)
    start_x = XwgCenter[valid_indices][0] if np.any(valid_indices) else XwgCenter[0]
    end_x = XwgCenter[valid_indices][-1] if np.any(valid_indices) else XwgCenter[-1]
    
    # Add pre and post regions
    pre_length = p.get('l_pre', 5.0)  # μm
    post_length = p.get('l_post', 5.0)  # μm
    
    # Total design length
    design_length = end_x - start_x
    
    # Infer simulation dimensions from design
    sim_size_x = design_length + pre_length + post_length
    
    # Calculate simulation width based on waveguide width
    # Make it wider than the waveguide to avoid boundary effects
    sim_size_y = 0  # 2D simulation along X-Z
    
    # Calculate simulation height based on material stack
    total_stack_height = boxt + total_wg_height + toxt + tair
    sim_size_z = total_stack_height + tSi
    
    print(f"Calculated simulation dimensions:")
    print(f"  X size (length): {sim_size_x:.2f} μm")
    print(f"  Y size (width): {sim_size_y:.2f} μm")
    print(f"  Z size (height): {sim_size_z:.2f} μm")
    
    # Simulation center
    sim_center_x = (start_x + end_x) / 2
    sim_center_y = 0
    
    # Define the coordinate system (important for correct positioning)
    # Place z=0 at the bottom of the waveguide layer for clarity
    # Layers will extend in both the +z and -z directions from there
    
    # The zero coordinate (z=0) is at the top of the BOX layer
    # (i.e., at the bottom of the waveguide layer)
    
    # Calculate simulation center z 
    # This is measured from z=0 (top of BOX)
    sim_center_z = (total_wg_height + toxt + tair - boxt-tSi) / 2
    #sim_center_z = (total_stack_height - boxt-tSi) / 2
    
    print(f"Simulation center: ({sim_center_x:.2f}, {sim_center_y:.2f}, {sim_center_z:.2f})")
    print(f"Z-coordinate reference: z=0 is at the top of the BOX layer / bottom of waveguide")
    
    # Create the simulation region
    sim_size = [sim_size_x, sim_size_y, sim_size_z]
    sim_center = [sim_center_x, sim_center_y, sim_center_z]
    
    # Set up the boundary conditions
    boundary_type = p.get('boundary_type', 'PML')
    if boundary_type == "PML":
        boundary_spec = td.BoundarySpec.pml(x=True, y=False, z=True)
    else:
        # Use specified boundary type
        boundary_spec = td.BoundarySpec.all_sides(boundary_type)
    
    # Get the material mediums
    wg_medium = get_material_medium(
        material_name, 
        material_index
    )
    
    oxide_medium = get_material_medium(
        "SiO2", 
        oxide_index
    )
    
    # Use silicon with fixed index of 3.5 for substrate
    si_index = p.get('si_index', 3.5)
    substrate_medium = get_material_medium(
        "Si", 
        si_index
    )
    
    # Create all the structures with coordinates referenced from z=0 at the top of BOX
    structures = []
    
    # ==== Calculate structure positions ====
    # Substrate: centered at z = -boxt - tSi/2
    # BOX: centered at z = -boxt/2
    # Bottom WG: centered at z = wghb/2
    # Intermediate (if dual): centered at z = wghb + inth/2
    # Top WG (if dual): centered at z = wghb + inth + wght/2
    # Top oxide: centered at z = total_wg_height + toxt/2
    # Air: centered at z = total_wg_height + toxt + tair/2
    
    extend_pml = 5.0  # Extend structures into PML by 5 um to avoid artifacts
    
    # Create substrate (at the bottom)
    substrate = td.Box(
        center=[sim_center_x, 0, -boxt - tSi/2-extend_pml/2],  # Centered in the substrate
        size=[sim_size_x + extend_pml, sim_size_y+extend_pml, tSi+extend_pml]
    )
    substrate_structure = td.Structure(geometry=substrate, medium=substrate_medium)
    structures.append(substrate_structure)
    
    # Create the background oxide region (includes box and top oxide)
    # First, create the BOX layer
    ##box_oxide = td.Box(
    ##    center=[sim_center_x, 0, -boxt/2],  # Centered in the BOX
    ##    size=[sim_size_x + extend_pml, sim_size_y, boxt]
    ##)
    ##box_structure = td.Structure(geometry=box_oxide, medium=oxide_medium)
    ##structures.append(box_structure)
    
    # Create the top oxide layer
    oxide = td.Box(
        center=[sim_center_x, 0, (total_wg_height + toxt-boxt)/2],  # Centered in the top oxide
        size=[sim_size_x + extend_pml, sim_size_y+extend_pml, toxt+total_wg_height+boxt]
    )
    top_oxide_structure = td.Structure(geometry=oxide, medium=oxide_medium)
    structures.append(top_oxide_structure)
    
    # Add pre-grating waveguide(s)
    if is_dual_waveguide:
        # Bottom waveguide
        pre_wg_bottom = td.Box(
            center=[start_x - pre_length/2, 0, wghb/2],  # Centered in bottom WG
            size=[pre_length, wg_width, wghb]
        )
        structures.append(td.Structure(geometry=pre_wg_bottom, medium=wg_medium))
        
        # Top waveguide
        pre_wg_top = td.Box(
            center=[start_x - pre_length/2, 0, wghb + inth + wght/2],  # Centered in top WG
            size=[pre_length, wg_width, wght]
        )
        structures.append(td.Structure(geometry=pre_wg_top, medium=wg_medium))
    else:
        # Single waveguide
        pre_wg = td.Box(
            center=[start_x - pre_length/2, 0, wghb/2],  # Centered in WG
            size=[pre_length, wg_width, wghb]
        )
        structures.append(td.Structure(geometry=pre_wg, medium=wg_medium))
    
    # Create and add grating structures with correct z-position
    print(f"Creating grating structures in {'dual' if is_dual_waveguide else 'single'} waveguide configuration")
    grating_structures = create_apodized_grating(
        XwgCenter, desDCxx, desperxx, 
        wghb=wghb, wg_width=wg_width,
        is_dual_waveguide=is_dual_waveguide, wght=wght, inth=inth
    )
    
    # Add the created grating structures with the correct material
    for structure in grating_structures:
        geometry = structure.geometry
        new_structure = td.Structure(geometry=geometry, medium=wg_medium)
        structures.append(new_structure)
    
    # Add post-grating waveguide(s)
    if is_dual_waveguide:
        # Bottom waveguide
        post_wg_bottom = td.Box(
            center=[end_x + post_length/2, 0, wghb/2],  # Centered in bottom WG
            size=[post_length, wg_width, wghb]
        )
        structures.append(td.Structure(geometry=post_wg_bottom, medium=wg_medium))
        
        # Top waveguide
        post_wg_top = td.Box(
            center=[end_x + post_length/2, 0, wghb + inth + wght/2],  # Centered in top WG
            size=[post_length, wg_width, wght]
        )
        structures.append(td.Structure(geometry=post_wg_top, medium=wg_medium))
    else:
        # Single waveguide
        post_wg = td.Box(
            center=[end_x + post_length/2+extend_pml/2, 0, wghb/2],  # Centered in WG
            size=[post_length+extend_pml, wg_width, wghb]
        )
        structures.append(td.Structure(geometry=post_wg, medium=wg_medium))
    
    # Create monitors
    monitors = []
    
    # Mode frequency
    freq0 = td.C_0 / wavelength
    
    # Frequency settings
    fwidth = p.get('fwidth', 0.1 * freq0)  # Default 10% bandwidth
    num_freqs = p.get('num_freqs', 1)  # Default single frequency
    
    # Create frequency array
    if num_freqs > 1:
        freqs = np.linspace(freq0 - fwidth/2, freq0 + fwidth/2, num_freqs)
    else:
        freqs = [freq0]
    
    # Field monitor for the entire domain (XZ plane)
    field_monitor = td.FieldMonitor(
        center=[sim_center_x, 0, sim_center_z],
        size=[sim_size_x, 0, sim_size_z],  # XZ plane monitor
        freqs=freqs,
        name="field_xz",
        fields=['Ey']  # Main component for TE mode
    )
    monitors.append(field_monitor)
    
    # 1D field monitor in the waveguide
    if is_dual_waveguide:
        # For dual waveguide, monitor both waveguides
        wg_bottom_z = wghb/2
        wg_top_z = wghb + inth + wght/2
        
        # Bottom waveguide field monitor
        field_wg_bottom = td.FieldMonitor(
            center=[sim_center_x, 0, wg_bottom_z],
            size=[sim_size_x, 2*wg_width, 0],  # 1D line along X-axis in bottom waveguide
            freqs=freqs,
            name="field_wg_bottom",
            fields=['Ey']
        )
        monitors.append(field_wg_bottom)
        
        # Top waveguide field monitor
        field_wg_top = td.FieldMonitor(
            center=[sim_center_x, 0, wg_top_z],
            size=[sim_size_x, 2*wg_width, 0],  # 1D line along X-axis in top waveguide
            freqs=freqs,
            name="field_wg_top",
            fields=['Ey']
        )
        monitors.append(field_wg_top)
    else:
        # Single waveguide field monitor
        field_wg = td.FieldMonitor(
            center=[sim_center_x, 0, wghb/2],
            size=[sim_size_x, 2*wg_width, 0],  # 1D line along X-axis in waveguide
            freqs=freqs,
            name="field_wg",
            fields=['Ey']
        )
        monitors.append(field_wg)
    
    # 1D field monitor in the air above the structure
    field_air = td.FieldMonitor(
        center=[sim_center_x, 0, total_wg_height + toxt + tair/2],
        size=[sim_size_x, 2*wg_width, 0],  # 1D line along X-axis in air
        freqs=freqs,
        name="field_air",
        fields=['Ey']
    )
    monitors.append(field_air)
    
    # Flux monitors
    # Input flux (position at the start of the pre-grating waveguide)
    if is_dual_waveguide:
        # For dual waveguide, position the flux monitor to capture both waveguides
        input_z = (wghb + inth + wght) / 2  # Center between top and bottom WGs
        input_height = wghb + inth + wght + 2  # Add padding
    else:
        input_z = wghb / 2  # Center of the waveguide
        input_height = wghb * 2  # Add padding
    
    flux_input = td.FluxMonitor(
        center=[start_x - pre_length/4, 0, input_z],
        size=[0, wg_width*10, input_height],
        freqs=freqs,
        name="flux_input"
    )
    monitors.append(flux_input)
    
    # Output flux (transmitted at the end of post-grating waveguide)
    flux_output = td.FluxMonitor(
        center=[end_x + post_length/4, 0, input_z],
        size=[0, wg_width*10, input_height],
        freqs=freqs,
        name="flux_output"
    )
    monitors.append(flux_output)
    
    # Top flux (upward emission)
    flux_up = td.FluxMonitor(
        center=[sim_center_x, 0, total_wg_height + toxt + tair/2],
        size=[sim_size_x, wg_width*2, 0],
        freqs=freqs,
        name="flux_up"
    )
    monitors.append(flux_up)
    
    # Bottom flux (downward emission) - in the substrate
    flux_down = td.FluxMonitor(
        center=[sim_center_x, 0, -boxt - tSi/4],  # Quarter way into the substrate
        size=[sim_size_x, wg_width*2, 0],
        freqs=freqs,
        name="flux_down"
    )
    monitors.append(flux_down)
    
    # Far-field monitor
    theta_angles = np.linspace(-np.pi/2, np.pi/2, 1801)  # 1-degree resolution
    far_field = td.FieldProjectionAngleMonitor(
        center=[sim_center_x, 0, total_wg_height + toxt + tair/2],
        size=[sim_size_x, wg_width*5, 0],
        freqs=freqs,
        name="far_field",
        theta=list(theta_angles),
        phi=[0],
        custom_origin=(sim_center_x, 0, total_wg_height + toxt + tair/2),
        proj_distance=1e6,  # Far field (μm)
        far_field_approx=True
    )
    monitors.append(far_field)
    
    # Create source
    source_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    
    # Mode source parameters - place at appropriate position based on waveguide configuration
    if is_dual_waveguide:
        # For dual waveguide, position at bottom waveguide by default
        source_z = wghb / 2
        source_height = wghb * 5
    else:
        source_z = wghb / 2
        source_height = wghb * 5
    
    source_size = [0, wg_width*2, source_height]
    source_center = [start_x - pre_length*0.75, 0, source_z]
    
    # Mode source
    mode_spec = td.ModeSpec(num_modes=1)
    source = td.ModeSource(
        center=source_center,
        size=source_size,
        source_time=source_time,
        direction='+',
        mode_spec=mode_spec,
        mode_index=0
    )
    
    # Grid resolution and run time
    grid_resolution = p.get('grid_resolution', 0.01)  # Default resolution is 0.01 μm
    
    # Set explicit run time if provided
    run_time = p.get('run_time', 1e-12)  # Default: 1 picosecond
    
    # Create the grid specification
    grid_spec = td.GridSpec.uniform(dl=grid_resolution)
    
    # Create the Tidy3D simulation object
    simulation = td.Simulation(
        size=sim_size,
        center=sim_center,
        structures=structures,
        sources=[source],
        monitors=monitors,
        boundary_spec=boundary_spec,
        grid_spec=grid_spec,
        run_time=run_time
    )
    
    # Calculate the number of grating periods based on the design
    valid_indices = ~np.isnan(desperxx) & (desDCxx > 0)
    if np.any(valid_indices):
        # Count approximate number of periods
        grating_length = end_x - start_x
        avg_period = np.mean(desperxx[valid_indices])
        num_periods = grating_length / avg_period
        print(f"Estimated number of grating periods: {num_periods:.1f}")
        print(f"Average grating period: {avg_period:.3f} μm")
        print(f"Total grating length: {grating_length:.2f} μm")
    else:
        print("Warning: No valid grating periods found in design_results")
    
    print(f"Created simulation with {len(structures)} total structures")
    
    return simulation, structures


def visualize_simulation(sim: td.Simulation, structures: List[td.Structure], 
                       design_results: Dict[str, Any], p: Dict[str, Any]) -> List[plt.Figure]:
    """
    Visualize the Tidy3D simulation setup with streamlined output.
    
    Args:
        sim: Tidy3D simulation object
        structures: List of structures in the simulation
        design_results: Dictionary containing longitudinal design results
        p: Parameters dictionary
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    
    # Plot the simulation geometry using Tidy3D built-in plotting
    sim.plot(y=0)  # plot in the xz plane
    
    # Extract parameters
    wavelength = p.get('wavelength', 0)
    material_name = p.get('material', 'Unknown')
    
    # Extract material stack info
    material_json = p.get('material_json')
    material_stack = {}
    
    if material_json and os.path.exists(material_json):
        try:
            with open(material_json, 'r') as f:
                material_data = json.load(f)
                if 'layer_stack' in material_data:
                    material_stack = material_data['layer_stack']
        except Exception as e:
            print(f"Error loading material JSON file: {e}")
    
    # Determine if this is a dual waveguide configuration
    is_dual_waveguide = material_stack.get('is_single_waveguide', True) == False
    
    # Get stack parameters
    wghb = material_stack.get('wghb', p.get('wghb', 0))
    wght = material_stack.get('wght', p.get('wght', 0)) if is_dual_waveguide else 0
    inth = material_stack.get('inth', p.get('inth', 0)) if is_dual_waveguide else 0
    toxt = material_stack.get('toxt', p.get('toxt', 0))
    boxt = material_stack.get('boxt', p.get('boxt', 0))
    tSi = material_stack.get('tSi', p.get('tSi', 0))
    tair = material_stack.get('tair', p.get('tair', 0))
    wg_width = p.get('wg_width', 0)
    
    # Extract valid positions for grating region
    valid_indices = ~np.isnan(design_results["desperxx"]) & (design_results["desDCxx"] > 0)
    if np.any(valid_indices):
        grating_start = design_results["XwgCenter"][valid_indices][0]
        grating_end = design_results["XwgCenter"][valid_indices][-1]
        grating_length = grating_end - grating_start
        avg_period = np.mean(design_results["desperxx"][valid_indices])
        num_periods = grating_length / avg_period
    else:
        grating_start = sim.center[0] - sim.size[0]/4
        grating_end = sim.center[0] + sim.size[0]/4
        avg_period = 0
        grating_length = 0
        num_periods = 0
    
    # Get simulation bounds
    bbox = sim.geometry.bounds
    xmin, xmax = bbox[0][0], bbox[1][0]
    zmin, zmax = bbox[0][2], bbox[1][2]
    
    # Define the current axes for annotations
    ax = plt.gca()
    
    # Calculate the total waveguide height
    total_wg_height = wghb + inth + wght if is_dual_waveguide else wghb
    
    # Highlight regions with semi-transparent rectangles
    
    # Waveguide region(s)
    waveguide_color = 'orange'
    alpha_value = 0.1
    
    if is_dual_waveguide:
        # Bottom waveguide highlight
        rect_bottom = plt.Rectangle(
            (xmin, 0), 
            xmax - xmin, 
            wghb,
            linewidth=0, facecolor=waveguide_color, alpha=alpha_value
        )
        ax.add_patch(rect_bottom)
        
        # Top waveguide highlight
        rect_top = plt.Rectangle(
            (xmin, wghb + inth), 
            xmax - xmin, 
            wght,
            linewidth=0, facecolor=waveguide_color, alpha=alpha_value
        )
        ax.add_patch(rect_top)
        
        # Label waveguides
        ax.text(grating_start - 2, wghb/2, "Bottom WG", 
               color='darkblue', fontsize=8, ha='right', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        ax.text(grating_start - 2, wghb + inth + wght/2, "Top WG", 
               color='darkblue', fontsize=8, ha='right', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    else:
        # Single waveguide highlight
        rect_wg = plt.Rectangle(
            (xmin, 0), 
            xmax - xmin, 
            wghb,
            linewidth=0, facecolor=waveguide_color, alpha=alpha_value
        )
        ax.add_patch(rect_wg)
        
        # Label waveguide
        ax.text(grating_start - 2, wghb/2, "Waveguide", 
               color='darkblue', fontsize=8, ha='right', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Highlight grating region
    if np.any(valid_indices):
        grat_rect = plt.Rectangle(
            (grating_start, 0), 
            grating_length, 
            total_wg_height,
            linewidth=1, edgecolor='red', facecolor='none', linestyle='--', alpha=0.7
        )
        ax.add_patch(grat_rect)
        
        # Label grating region
        ax.text((grating_start + grating_end)/2, total_wg_height + 0.2, 
               f"Grating Region\n~{num_periods:.1f} periods, {avg_period:.3f}μm avg. period", 
               color='red', fontsize=8, ha='center', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add dimension indicators on the right side
    x_label_pos = xmax - 1.0
    
    # Add vertical indicators for material regions
    if toxt > 0:
        # Top oxide
        ax.plot([x_label_pos, x_label_pos], [total_wg_height, total_wg_height + toxt], 'b-', lw=1)
        ax.plot([x_label_pos, x_label_pos-0.2], [total_wg_height, total_wg_height+0.2], 'b-', lw=1)  # Bottom arrow
        ax.plot([x_label_pos, x_label_pos+0.2], [total_wg_height, total_wg_height+0.2], 'b-', lw=1)
        ax.plot([x_label_pos, x_label_pos-0.2], [total_wg_height+toxt, total_wg_height+toxt-0.2], 'b-', lw=1)  # Top arrow
        ax.plot([x_label_pos, x_label_pos+0.2], [total_wg_height+toxt, total_wg_height+toxt-0.2], 'b-', lw=1)
        
        # Label for top oxide
        ax.text(x_label_pos + 0.3, total_wg_height + toxt/2, f"toxt={toxt}μm", 
               ha='left', va='center', fontsize=8, color='blue')
        
        # Material name
        ax.text(x_label_pos - 0.5, total_wg_height + toxt/2, "SiO2", 
               ha='right', va='center', fontsize=8)
    
    if boxt > 0:
        # Bottom oxide
        ax.plot([x_label_pos, x_label_pos], [-boxt, 0], 'b-', lw=1)
        ax.plot([x_label_pos, x_label_pos-0.2], [-boxt, -boxt+0.2], 'b-', lw=1)  # Bottom arrow
        ax.plot([x_label_pos, x_label_pos+0.2], [-boxt, -boxt+0.2], 'b-', lw=1)
        ax.plot([x_label_pos, x_label_pos-0.2], [0, -0.2], 'b-', lw=1)  # Top arrow
        ax.plot([x_label_pos, x_label_pos+0.2], [0, -0.2], 'b-', lw=1)
        
        # Label for bottom oxide
        ax.text(x_label_pos + 0.3, -boxt/2, f"boxt={boxt}μm", 
               ha='left', va='center', fontsize=8, color='blue')
        
        # Material name
        ax.text(x_label_pos - 0.5, -boxt/2, "SiO2", 
               ha='right', va='center', fontsize=8)
    
    if tair > 0:
        # Air layer
        air_start = total_wg_height + toxt
        ax.plot([x_label_pos, x_label_pos], [air_start, air_start + tair], 'lightblue', lw=1)
        ax.plot([x_label_pos, x_label_pos-0.2], [air_start, air_start+0.2], 'lightblue', lw=1)  # Bottom arrow
        ax.plot([x_label_pos, x_label_pos+0.2], [air_start, air_start+0.2], 'lightblue', lw=1)
        ax.plot([x_label_pos, x_label_pos-0.2], [air_start+tair, air_start+tair-0.2], 'lightblue', lw=1)  # Top arrow
        ax.plot([x_label_pos, x_label_pos+0.2], [air_start+tair, air_start+tair-0.2], 'lightblue', lw=1)
        
        # Label for air
        ax.text(x_label_pos + 0.3, air_start + tair/2, f"tair={tair}μm", 
               ha='left', va='center', fontsize=8, color='lightblue')
        
        # Material name
        ax.text(x_label_pos - 0.5, air_start + tair/2, "Air", 
               ha='right', va='center', fontsize=8)
    
    # Add waveguide dimension indicators
    x_dim_pos = x_label_pos - 2.0
    
    if is_dual_waveguide:
        # Bottom waveguide height
        ax.plot([x_dim_pos, x_dim_pos], [0, wghb], 'orange', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [0, 0.1], 'orange', lw=1)  # Bottom arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [0, 0.1], 'orange', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [wghb, wghb-0.1], 'orange', lw=1)  # Top arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [wghb, wghb-0.1], 'orange', lw=1)
        
        # Label for bottom waveguide
        ax.text(x_dim_pos + 0.2, wghb/2, f"wghb={wghb}μm", 
               ha='left', va='center', fontsize=8, color='orange')
        
        # Intermediate layer
        ax.plot([x_dim_pos, x_dim_pos], [wghb, wghb+inth], 'gray', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [wghb, wghb+0.1], 'gray', lw=1)  # Bottom arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [wghb, wghb+0.1], 'gray', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [wghb+inth, wghb+inth-0.1], 'gray', lw=1)  # Top arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [wghb+inth, wghb+inth-0.1], 'gray', lw=1)
        
        # Label for intermediate layer
        ax.text(x_dim_pos + 0.2, wghb+inth/2, f"inth={inth}μm", 
               ha='left', va='center', fontsize=8, color='gray')
        
        # Top waveguide height
        ax.plot([x_dim_pos, x_dim_pos], [wghb+inth, wghb+inth+wght], 'orange', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [wghb+inth, wghb+inth+0.1], 'orange', lw=1)  # Bottom arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [wghb+inth, wghb+inth+0.1], 'orange', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [wghb+inth+wght, wghb+inth+wght-0.1], 'orange', lw=1)  # Top arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [wghb+inth+wght, wghb+inth+wght-0.1], 'orange', lw=1)
        
        # Label for top waveguide
        ax.text(x_dim_pos + 0.2, wghb+inth+wght/2, f"wght={wght}μm", 
               ha='left', va='center', fontsize=8, color='orange')
    else:
        # Single waveguide height
        ax.plot([x_dim_pos, x_dim_pos], [0, wghb], 'orange', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [0, 0.1], 'orange', lw=1)  # Bottom arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [0, 0.1], 'orange', lw=1)
        ax.plot([x_dim_pos, x_dim_pos-0.1], [wghb, wghb-0.1], 'orange', lw=1)  # Top arrow
        ax.plot([x_dim_pos, x_dim_pos+0.1], [wghb, wghb-0.1], 'orange', lw=1)
        
        # Label for waveguide height
        ax.text(x_dim_pos + 0.2, wghb/2, f"wghb={wghb}μm", 
               ha='left', va='center', fontsize=8, color='orange')
    
    # Add source and monitor markers
    for i, source in enumerate(sim.sources):
        if hasattr(source, 'center'):
            src_center = source.center
            # Draw source marker
            ax.plot([src_center[0]], [src_center[2]], 'r*', markersize=10)
            ax.text(src_center[0], src_center[2] - 0.5, "Source", 
                   color='red', fontsize=8, ha='center', va='top')
    
    # Find and label flux monitors
    for i, monitor in enumerate(sim.monitors):
        if hasattr(monitor, 'center') and hasattr(monitor, 'name') and 'flux' in monitor.name.lower():
            mon_center = monitor.center
            if mon_center[0] < grating_start:  # Input flux
                ax.plot([mon_center[0]], [mon_center[2]], 'go', markersize=6)
                ax.text(mon_center[0], mon_center[2] - 0.3, "Input Flux", 
                       color='green', fontsize=7, ha='center', va='top')
            elif mon_center[0] > grating_end:  # Output flux
                ax.plot([mon_center[0]], [mon_center[2]], 'go', markersize=6)
                ax.text(mon_center[0], mon_center[2] - 0.3, "Output Flux", 
                       color='green', fontsize=7, ha='center', va='top')
            elif mon_center[2] > total_wg_height:  # Upward flux
                ax.plot([mon_center[0]], [mon_center[2]], 'go', markersize=6)
                ax.text(mon_center[0], mon_center[2] + 0.3, "Up Flux", 
                       color='green', fontsize=7, ha='center', va='bottom')
    
    # Add title with grating information
    if is_dual_waveguide:
        plt.title(f"{material_name} Dual Waveguide Grating at λ={wavelength*1000:.0f}nm\n"
                f"Avg. Period: {avg_period:.3f}μm, Length: {grating_length:.2f}μm, ~{num_periods:.1f} periods")
    else:
        plt.title(f"{material_name} Grating at λ={wavelength*1000:.0f}nm\n"
                f"Avg. Period: {avg_period:.3f}μm, Length: {grating_length:.2f}μm, ~{num_periods:.1f} periods")
    
    plt.tight_layout()
    figures.append(plt.gcf())
    
    return figures


def run_simulation(sim: td.Simulation, task_name: str, output_dir: str = "output") -> str:
    """
    Run a Tidy3D simulation.
    
    Args:
        sim: Tidy3D simulation object
        task_name: Name for the simulation task
        output_dir: Directory to save the simulation results
        
    Returns:
        Path to the simulation results file
    """
    import tidy3d.web as web
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a Tidy3D web job
    job = web.Job(simulation=sim, task_name=task_name)
    
    # Estimate simulation cost
    estimated_cost = web.estimate_cost(job.task_id)
    print(f"Estimated simulation cost: {estimated_cost:.3f} Flex Credits")
    
    # Run the simulation
    print(f"Running simulation '{task_name}'...")
    results_path = os.path.join(output_dir, f"{task_name}_results.hdf5")
    sim_data = job.run(path=results_path)
    
    print(f"Simulation completed successfully.")
    print(f"Results saved to: {results_path}")
    
    return results_path


def analyze_results(results_path: str, design_results: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the Tidy3D simulation results.
    
    Args:
        results_path: Path to the simulation results file
        design_results: Dictionary containing the longitudinal design results
        p: Parameters dictionary
        
    Returns:
        Dictionary with analysis results
    """
    # Check if results file exists
    if not os.path.exists(results_path):
        return {
            "error": f"Results file not found: {results_path}"
        }
    
    # Load the simulation data
    sim_data = td.SimulationData.from_file(results_path)
    
    # Check if required monitors exist
    required_monitors = ["flux_input", "far_field"]
    for mon in required_monitors:
        if mon not in sim_data.monitor_data:
            return {
                "error": f"Required monitor '{mon}' not found in simulation results",
                "sim_data": sim_data
            }
    
    # Get the frequency (use first frequency if multiple)
    freq0 = sim_data.monitor_data["flux_input"].flux.coords["f"].values[0]
    
    # Calculate efficiencies
    input_flux = sim_data.monitor_data["flux_input"].flux.sel(f=freq0).values
    
    # Ensure input_flux is a scalar
    if hasattr(input_flux, "__len__"):
        input_flux = input_flux.item(0) if input_flux.size == 1 else input_flux[0]
    else:
        input_flux = input_flux
    
    # Check for output flux monitor (optional)
    transmission_efficiency = None
    if "flux_output" in sim_data.monitor_data:
        output_flux = sim_data.monitor_data["flux_output"].flux.sel(f=freq0).values
        # Ensure output_flux is a scalar
        if hasattr(output_flux, "__len__"):
            output_flux = output_flux.item(0) if output_flux.size == 1 else output_flux[0]
        transmission_efficiency = float(output_flux / input_flux)
    
    # Check for upward flux monitor (optional)
    upward_efficiency = None
    if "flux_up" in sim_data.monitor_data:
        upward_flux = sim_data.monitor_data["flux_up"].flux.sel(f=freq0).values
        # Ensure upward_flux is a scalar
        if hasattr(upward_flux, "__len__"):
            upward_flux = upward_flux.item(0) if upward_flux.size == 1 else upward_flux[0]
        upward_efficiency = float(upward_flux / input_flux)
    
    # Far-field analysis
    ff_data = sim_data.monitor_data["far_field"]
    
    # Get theta values from far-field monitor
    theta_values = np.array(ff_data.theta)
    theta_degrees = np.degrees(theta_values)

    # Get intensity (for first frequency and phi=0)
    # First, let's print the available coordinates to debug
    print("\nDEBUG: Far-field data coordinates:")
    print(f"Available coordinates: {ff_data.coords}")
    print(f"Frequency values: {ff_data.f}")
    
    # Get the field components and ensure we're selecting the correct frequency
    Er = ff_data.Er.sel(f=freq0, phi=0).values
    Etheta = ff_data.Etheta.sel(f=freq0, phi=0).values
    Ephi = ff_data.Ephi.sel(f=freq0, phi=0).values
    
    # Print shapes to debug
    print(f"\nDEBUG: Field component shapes:")
    print(f"Er shape: {Er.shape}")
    print(f"Etheta shape: {Etheta.shape}")
    print(f"Ephi shape: {Ephi.shape}")
    
    # Compute intensity from field components
    intensity = np.abs(Er)**2 + np.abs(Etheta)**2 + np.abs(Ephi)**2
    
    # Ensure we have a 1D array by selecting the first frequency index
    # The array should be [freq, theta, phi] -> we want just the theta dimension
    if intensity.ndim > 1:
        print(f"\nDEBUG: Intensity shape before reduction: {intensity.shape}")
        # Take the first frequency index since we're only simulating one frequency
        intensity = intensity[0, :]
        #intensity = intensity[0, :, 0]
        print(f"Intensity shape after reduction: {intensity.shape}")
    
    # Ensure intensity is a 1D array
    intensity = np.squeeze(intensity)
    print(f"Final intensity shape: {intensity.shape}")
    
    # Print detailed debugging information
    print(f"\nDEBUG: Array shapes and sizes:")
    print(f"intensity shape: {intensity.shape}, size: {intensity.size}")
    print(f"theta_degrees shape: {theta_degrees.shape}, size: {theta_degrees.size}")
    print(f"intensity min/max: {np.min(intensity):.2e}/{np.max(intensity):.2e}")
    print(f"theta_degrees min/max: {np.min(theta_degrees):.2f}/{np.max(theta_degrees):.2f}")
   
    # Calculate peak emission angle
    peak_idx = np.argmax(intensity)
    print(f"\nDEBUG: Peak index: {peak_idx}, Array length: {len(theta_degrees)}")
    peak_angle = theta_degrees[peak_idx]
    
    # Calculate FWHM (Full Width at Half Maximum)
    half_max = intensity.max() / 2
    above_half_max = intensity >= half_max
    edges = np.where(np.diff(above_half_max.astype(int)))[0]
    
    if len(edges) >= 2:
        left_idx, right_idx = edges[0], edges[1]
        left_angle = theta_degrees[left_idx]
        right_angle = theta_degrees[right_idx]
        fwhm = right_angle - left_angle
    else:
        # Fall back if FWHM calculation fails
        fwhm = None
    
    # Return analysis results
    analysis_results = {
        "frequency": freq0,
        "wavelength": td.C_0 / freq0,
        "transmission_efficiency": transmission_efficiency,
        "upward_efficiency": upward_efficiency,
        "peak_angle": float(peak_angle),
        "fwhm": fwhm,
        "theta_degrees": theta_degrees,
        "intensity": intensity,
        "sim_data": sim_data
    }
    
    return analysis_results
        


def visualize_results(analysis_results: Dict[str, Any], design_results: Dict[str, Any], 
                    p: Dict[str, Any]) -> List[plt.Figure]:
    """
    Visualize the simulation results with streamlined output.
    
    Args:
        analysis_results: Results from analyze_results
        design_results: Dictionary containing the longitudinal design results
        p: Parameters dictionary
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Check if there's an error in the analysis_results
    if "error" in analysis_results:
        # Create error figure
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error in simulation results:\n{analysis_results['error']}", 
                ha='center', va='center', fontsize=12, color='red',
                bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round'))
        plt.axis('off')
        plt.title("Simulation Error")
        figures.append(plt.gcf())
        return figures
    
    sim_data = analysis_results.get("sim_data")
    theta_degrees = analysis_results.get("theta_degrees")
    intensity = analysis_results.get("intensity")
    peak_angle = analysis_results.get("peak_angle")
    fwhm = analysis_results.get("fwhm")
    freq0 = analysis_results.get("frequency")
    
    # Only create figures if we have valid data
    if theta_degrees is not None and intensity is not None and len(theta_degrees) > 0 and len(intensity) > 0:
        # Figure 1: Combined far-field emission pattern and metrics
        plt.figure(figsize=(10, 6))
        
        # Plot far-field pattern
        plt.plot(theta_degrees, intensity / np.max(intensity), 'b-', linewidth=2)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Normalized Intensity")
        plt.grid(True, alpha=0.3)
        plt.xlim(-90, 90)
        
        # Mark peak angle
        if peak_angle is not None:
            plt.axvline(x=peak_angle, color='r', linestyle='--', alpha=0.7, 
                      label=f"Peak: {peak_angle:.2f}°")
        
        # Add efficiency metrics as text annotation
        metrics = []
        if 'wavelength' in analysis_results:
            metrics.append(f"λ = {analysis_results['wavelength']*1000:.1f} nm")
        if 'transmission_efficiency' in analysis_results and analysis_results['transmission_efficiency'] is not None:
            trans_eff = analysis_results['transmission_efficiency'] * 100
            metrics.append(f"Trans: {trans_eff:.1f}%")
        if 'upward_efficiency' in analysis_results and analysis_results['upward_efficiency'] is not None:
            up_eff = analysis_results['upward_efficiency'] * 100
            metrics.append(f"Up: {up_eff:.1f}%")
        if fwhm is not None:
            metrics.append(f"FWHM: {fwhm:.2f}°")
            
        # Add metrics text box
        if metrics:
            plt.annotate('\n'.join(metrics), xy=(0.03, 0.97), xycoords='axes fraction',
                        va='top', ha='left', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.title("Far-Field Emission Pattern")
        plt.legend(loc='upper right')
        plt.tight_layout()
        figures.append(plt.gcf())
    
    # Figure 2: Electric field visualization (only if field data exists)
    if "field_xz" in sim_data.monitor_data:
        field_data = sim_data.monitor_data["field_xz"]
        has_field_data = False
        
        # Check for available field components
        for field_comp in ['Ey', 'Ex', 'Ez']:
            if hasattr(field_data, field_comp):
                field = getattr(field_data, field_comp)
                if hasattr(field, 'sel') and 'f' in field.coords:
                    # Get field data for the first frequency
                    print(field.shape)
                    field_slice = field.sel(f=freq0)
                    print(field_slice.shape)
                    field_mag = np.abs(field_slice.values)
                    #field_mag = field_mag[:,:,:,0]
                    
                    # Squeeze out singleton dimensions for proper plotting
                    if field_mag.ndim > 2:
                        field_mag = np.squeeze(field_mag)
                    
                    # Skip if field data is all zeros or NaNs
                    if np.all(np.isnan(field_mag)) or np.all(field_mag == 0):
                        continue
                        
                    has_field_data = True
                    
                    # Get spatial coordinates
                    x_coords = field_data.Ey.x.values
                    z_coords = field_data.Ey.z.values
                    
                    # Create field plot
                    plt.figure(figsize=(10, 4))
                    extent = [x_coords.min(), x_coords.max(), z_coords.min(), z_coords.max()]
                    plt.imshow(field_mag.T, origin='lower', aspect='auto', cmap='magma', extent=extent)
                    plt.colorbar(label=f"|{field_comp}|")
                    plt.xlabel("X (μm)")
                    plt.ylabel("Z (μm)")
                    plt.title(f"Electric Field Magnitude (|{field_comp}|) at λ={td.C_0/freq0:.3f} μm")
                    plt.tight_layout()
                    figures.append(plt.gcf())
                    break
        
    return figures


def plot_material_indices(sim: td.Simulation, freqs: List[float], figsize: Tuple[int, int] = (6, 8)) -> plt.Figure:
    """
    Plot refractive index vs Z (μm) for each frequency in freqs.
    Uses simulation bounds to create a uniform grid of z points and computes the refractive index at each point.
    
    Args:
        sim: Tidy3D simulation object
        freqs: List of frequencies to compute refractive indices at (Hz)
        figsize: Figure size (default: (6, 8))
    
    Returns:
        fig: Matplotlib figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert freqs to numpy array if not already
    freqs = np.array(freqs)
    
    # Get simulation bounds
    bounds = sim.geometry.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]
    
    # Create a uniform grid of z points
    z_points = np.linspace(z_min, z_max, 1000)
    
    # Dictionary to track unique mediums and their descriptive names
    medium_names = {}
    
    # Color map for multiple wavelengths
    cmap = plt.cm.viridis
    colors = [cmap(i/len(freqs)) for i in range(len(freqs))]
    
    for i, f in enumerate(freqs):
        # Convert to wavelength in μm for labeling
        wavelength = td.C_0 / f
        wl_label = f"{wavelength:.3f} μm"
        
        # Compute refractive index at each z point
        n_profile = []
        for z in z_points:
            # Find the medium at this z position
            med = None
            for struct in sim.structures:
                geom = getattr(struct, 'geometry', None)
                if geom and hasattr(geom, 'center') and hasattr(geom, 'size'):
                    z_min_struct = geom.center[2] - geom.size[2] / 2
                    z_max_struct = geom.center[2] + geom.size[2] / 2
                    if z_min_struct <= z <= z_max_struct:
                        med = struct.medium
                        # No break since structures defined after will override preceding ones
            
            # Compute refractive index
            if med:
                if hasattr(med, 'eps_model'):
                    eps = med.eps_model(f)
                    n_val = np.sqrt(np.real(eps))
                    source = "Fit"
                else:
                    n_val = np.sqrt(float(med.permittivity))
                    source = "Fixed Index"
                
                # Track medium name
                if med not in medium_names:
                    medium_names[med] = f"{med.name} ({source})" if hasattr(med, 'name') else f"{med.__class__.__name__} ({source})"
            else:
                n_val = 1.0  # Default to air if no medium found
            
            n_profile.append(n_val)
        
        # Plot refractive index profile for this frequency
        ax.plot(n_profile, z_points, label=wl_label, color=colors[i], linewidth=2)
    
    # Annotate material regions
    z_mid_vals = {}
    for struct in sim.structures:
        geom = getattr(struct, 'geometry', None)
        if geom and hasattr(geom, 'center') and hasattr(geom, 'size'):
            z_min_struct = geom.center[2] - geom.size[2] / 2
            z_max_struct = geom.center[2] + geom.size[2] / 2
            z_mid = (z_min_struct + z_max_struct) / 2
            
            # Skip if this z region has already been labeled
            if any(abs(z - z_mid) < 0.2 for z in z_mid_vals.values()):
                continue
            
            med = struct.medium
            med_name = getattr(med, 'name', med.__class__.__name__)
            
            if hasattr(med, 'permittivity'):
                n_val = np.sqrt(float(med.permittivity))
                ax.text(n_val + 0.05, z_mid, med_name, 
                       ha='left', va='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                z_mid_vals[med_name] = z_mid
    
    # Add horizontal lines at material interfaces
    material_boundaries = []
    for struct in sim.structures:
        geom = getattr(struct, 'geometry', None)
        if geom and hasattr(geom, 'center') and hasattr(geom, 'size'):
            z_min_struct = geom.center[2] - geom.size[2] / 2
            z_max_struct = geom.center[2] + geom.size[2] / 2
            
            # Add to boundaries if not already there
            if z_min_struct not in material_boundaries:
                material_boundaries.append(z_min_struct)
            if z_max_struct not in material_boundaries:
                material_boundaries.append(z_max_struct)
    
    # Sort boundaries and draw horizontal lines
    material_boundaries.sort()
    for z_boundary in material_boundaries:
        if z_min < z_boundary < z_max:  # Only draw if within plot limits
            ax.axhline(y=z_boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Add axis labels
    ax.set_xlabel('Refractive Index (n)', fontsize=12)
    ax.set_ylabel('Z Position (μm)', fontsize=12)
    ax.set_title('Material Index vs Z Position', fontsize=14)
    
    # Add legend
    ax.legend(loc='best', fontsize=9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Improve aspect ratio
    ax.set_box_aspect(1.5)
    
    plt.tight_layout()
    return fig 


def plot_grating_parameters(design_results: Dict[str, Any], grating_structures: List[td.Structure] = None, 
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the period and duty cycle versus x-position for the grating design.
    
    Args:
        design_results: Dictionary containing longitudinal design results
        grating_structures: Optional list of actual grating structures from simulation
        figsize: Figure size (default: (10, 6))
        
    Returns:
        fig: Matplotlib figure object
    """
    # Extract design parameters
    XwgCenter = design_results.get("XwgCenter", [])
    desDCxx = design_results.get("desDCxx", [])
    desperxx = design_results.get("desperxx", [])
    
    # Filter out invalid values (NaN or duty cycle <= 0)
    valid_indices = ~np.isnan(desperxx) & (desDCxx > 0)
    control_positions = np.array(XwgCenter)[valid_indices]
    control_dc_values = np.array(desDCxx)[valid_indices]
    control_period_values = np.array(desperxx)[valid_indices]
    
    if len(control_positions) < 2:
        # Create figure with error message if insufficient data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient valid data points for interpolation",
               ha='center', va='center', fontsize=12)
        return fig
    
    # Create a dense array of positions for smooth interpolation
    min_pos = control_positions[0]
    max_pos = control_positions[-1]
    num_points = 200  # Number of points for smooth curve
    x_positions = np.linspace(min_pos, max_pos, num_points)
    
    # Interpolate period and duty cycle at each position
    period_values = np.interp(x_positions, control_positions, control_period_values)
    dc_values = np.interp(x_positions, control_positions, control_dc_values)
    
    # Create the figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot period on the first y-axis
    ax1.set_xlabel('X Position (μm)', fontsize=12)
    ax1.set_ylabel('Period (μm)', fontsize=12, color='blue')
    ax1.plot(x_positions, period_values, 'b-', label='Period (Interpolated)', linewidth=2)
    # Plot the original control points
    ax1.scatter(control_positions, control_period_values, color='blue', s=30, marker='o',
               label='Period (Control Points)', zorder=4)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Create a second y-axis for duty cycle
    ax2 = ax1.twinx()
    ax2.set_ylabel('Duty Cycle', fontsize=12, color='red')
    ax2.plot(x_positions, dc_values, 'r-', label='Duty Cycle (Interpolated)', linewidth=2)
    # Plot the original control points
    ax2.scatter(control_positions, control_dc_values, color='red', s=30, marker='o',
               label='Duty Cycle (Control Points)', zorder=4)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Calculate and extract actual teeth from structures if provided
    actual_positions = []
    actual_periods = []
    actual_dc = []
    
    if grating_structures is not None and len(x_positions) > 0:
        # Define grating region bounds
        grating_start = min_pos - period_values[0]
        grating_end = max_pos + period_values[-1]
        
        # Extract teeth from structures
        teeth_positions = []
        teeth_lengths = []
        
        for struct in grating_structures:
            if hasattr(struct.geometry, 'center') and hasattr(struct.geometry, 'size'):
                center = struct.geometry.center
                size = struct.geometry.size
                
                # Only consider likely teeth - within grating region and reasonable size
                position = center[0]
                tooth_length = size[0]
                is_in_wg_layer = abs(center[2]) < 5
                
                if (grating_start <= position <= grating_end and 
                    0.001 < tooth_length < 2.0 and 
                    is_in_wg_layer):
                    teeth_positions.append(position)
                    teeth_lengths.append(tooth_length)
        
        # Sort teeth by position
        if teeth_positions:
            sorted_teeth = sorted(zip(teeth_positions, teeth_lengths))
            teeth_positions = [t[0] for t in sorted_teeth]
            teeth_lengths = [t[1] for t in sorted_teeth]
            
            # Calculate actual periods and duty cycles
            for i in range(1, len(teeth_positions)):
                period = abs(teeth_positions[i] - teeth_lengths[i]/2 - teeth_positions[i-1] + teeth_lengths[i-1]/2)
                dc = teeth_lengths[i] / period if period > 0 else 0
                
                if period > 0:
                    actual_positions.append(teeth_positions[i])
                    actual_periods.append(period)
                    actual_dc.append(dc)
            
            # Plot actual parameters
            if actual_positions:
                ax1.scatter(actual_positions, actual_periods, color='blue', s=40, marker='x',
                          label='Actual Period', zorder=4)
                ax2.scatter(actual_positions, actual_dc, color='red', s=40, marker='x',
                          label='Actual DC', zorder=4)
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Add title and stats
    avg_period = np.mean(period_values)
    avg_dc = np.mean(dc_values)
    plt.title('Grating Period and Duty Cycle vs Position', fontsize=14)
    
    stats_text = f"Average Period: {avg_period:.3f} μm\nAverage Duty Cycle: {avg_dc:.3f}"
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the combined legend
    
    return fig


def plot_grating_on_lut(design_results: Dict[str, Any], lut_file: str,
                      grating_structures: List[td.Structure] = None,
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot the grating parameters on a 2D contour LUT plot.
    
    Args:
        design_results: Dictionary containing longitudinal design results
        lut_file: Path to the LUT data file (.csv or .mat)
        grating_structures: Optional list of actual grating structures from simulation
        figsize: Figure size (default: (10, 8))
        
    Returns:
        fig: Matplotlib figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if LUT file exists
    if not os.path.exists(lut_file):
        ax.text(0.5, 0.5, f"LUT file not found: {lut_file}", ha='center', va='center', fontsize=12)
        return fig
    
    # Load data file (supports both .mat and .csv)
    if lut_file.endswith('.csv'):
        # Load CSV file using existing lut_handler function
        from . import lut_handler
        lut_data = lut_handler.load_csv_data(lut_file)
        
        # Extract data in the same format as .mat files
        per_grid = lut_data['gratpersSave'] / 1000.0  # Convert to μm
        dc_grid = lut_data['pertDCsSave']
        spec_2d = lut_data['thetasFFSave']
    else:
        # Load MATLAB .mat file
        from scipy import io
        mat_data = io.loadmat(lut_file)
        
        # Extract AO_AIM format data
        per_grid = mat_data['gratpersSave'] / 1000.0  # Convert to μm
        dc_grid = mat_data['pertDCsSave']
        spec_2d = mat_data['thetasFFSave']
    
    # Default metadata
    lut_title = 'Grating Coupler LUT'
    lut_x_label = 'Period (μm)'
    lut_y_label = 'Duty Cycle'
    lut_z_label = 'Diffraction Angle (deg)'
    target_angle = None
    
    # Try to extract target angle from filename
    import re
    angle_match = re.search(r'(?:angle|ang)[-_]?(\d+)', os.path.basename(lut_file), re.IGNORECASE)
    if angle_match:
        target_angle = float(angle_match.group(1))
    
    # Extract design parameters
    XwgCenter = design_results.get("XwgCenter", [])
    desDCxx = design_results.get("desDCxx", [])
    desperxx = design_results.get("desperxx", [])
    
    # Filter out invalid values (NaN or duty cycle <= 0)
    valid_indices = ~np.isnan(desperxx) & (desDCxx > 0)
    control_positions = np.array(XwgCenter)[valid_indices]
    control_dc_values = np.array(desDCxx)[valid_indices]
    control_period_values = np.array(desperxx)[valid_indices]
    
    # Create a dense array of positions for smooth interpolation
    if len(control_positions) >= 2:
        min_pos = control_positions[0]
        max_pos = control_positions[-1]
        num_points = 200  # Number of points for smooth curve
        interp_positions = np.linspace(min_pos, max_pos, num_points)
        
        # Interpolate period and duty cycle at each position
        interp_period_values = np.interp(interp_positions, control_positions, control_period_values)
        interp_dc_values = np.interp(interp_positions, control_positions, control_dc_values)
    else:
        # Insufficient data for interpolation
        interp_positions = []
        interp_period_values = []
        interp_dc_values = []
    
    # Make sure we have 1D arrays for per_grid and dc_grid for meshgrid creation
    per_grid = np.array(per_grid).squeeze()
    dc_grid = np.array(dc_grid).squeeze()
    spec_2d = np.array(spec_2d).squeeze()
    
    # Fix the dimensions for the LUT data - ensure we have unique 1D arrays for mesh creation
    if per_grid.ndim > 1:
        per_grid = np.unique(per_grid)
    if dc_grid.ndim > 1:
        dc_grid = np.unique(dc_grid)
    
    # If spec_2d is 1D, reshape it to match the expected grid dimensions
    if spec_2d.ndim == 1:
        # We assume per_grid and dc_grid contain the unique values
        # and spec_2d contains values for each combination
        if len(spec_2d) == len(per_grid) * len(dc_grid):
            spec_2d = spec_2d.reshape(len(dc_grid), len(per_grid))
    
    # Create proper 2D meshgrid for plotting - this is what pcolormesh needs
    X, Y = np.meshgrid(per_grid, dc_grid)
    
    # Simpler heatmap plot using imshow which is more forgiving with dimensions
    extent = [per_grid.min(), per_grid.max(), dc_grid.min(), dc_grid.max()]
    contour = ax.imshow(spec_2d, extent=extent, aspect='auto', origin='lower', cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(lut_z_label, fontsize=12)
    
    # Plot design parameters as a curve
    if len(interp_positions) > 0:
        # Plot the interpolated trajectory
        sc = ax.scatter(interp_period_values, interp_dc_values, c='white', s=2, alpha=0.5)
        
        # Plot the trajectory with gradient color based on position
        points = np.array([interp_period_values, interp_dc_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a line collection
        from matplotlib.collections import LineCollection
        norm = plt.Normalize(interp_positions.min(), interp_positions.max())
        lc = LineCollection(segments, cmap='cool', norm=norm)
        lc.set_array(interp_positions[:-1])
        lc.set_linewidth(3)
        line = ax.add_collection(lc)
        
        # Add colorbar for position
        position_cbar = plt.colorbar(line, ax=ax, location='right', pad=0.1)
        position_cbar.set_label('Position (μm)', fontsize=12)
        
        # Plot the control points
        ax.scatter(control_period_values, control_dc_values, color='white', s=60, marker='o',
                 label='Control Points', edgecolors='black', linewidth=1, zorder=5)
        
    # Calculate and extract actual teeth from structures if provided
    actual_periods = []
    actual_dc = []
    
    if grating_structures is not None and len(interp_positions) > 0:
        # Define grating region bounds
        grating_start = min_pos - interp_period_values[0]
        grating_end = max_pos + interp_period_values[-1]
        
        # Extract teeth from structures
        teeth_positions = []
        teeth_lengths = []
        
        for struct in grating_structures:
            if hasattr(struct.geometry, 'center') and hasattr(struct.geometry, 'size'):
                center = struct.geometry.center
                size = struct.geometry.size
                
                # Only consider likely teeth - within grating region and reasonable size
                position = center[0]
                tooth_length = size[0]
                is_in_wg_layer = abs(center[2]) < 5
                
                if (grating_start <= position <= grating_end and 
                    0.001 < tooth_length < 2.0 and 
                    is_in_wg_layer):
                    teeth_positions.append(position)
                    teeth_lengths.append(tooth_length)
        
        # Calculate actual periods and duty cycles
        if teeth_positions:
            sorted_teeth = sorted(zip(teeth_positions, teeth_lengths))
            teeth_positions = [t[0] for t in sorted_teeth]
            teeth_lengths = [t[1] for t in sorted_teeth]
            
            actual_periods = []
            actual_dc = []
            
            for i in range(1, len(teeth_positions)):
                period = abs(teeth_positions[i] - teeth_lengths[i]/2 - teeth_positions[i-1] + teeth_lengths[i-1]/2)
                dc = teeth_lengths[i] / period if period > 0 else 0
                
                if period > 0:
                    actual_periods.append(period)
                    actual_dc.append(dc)
            
            # Plot actual parameters
            if actual_periods:
                ax.scatter(actual_periods, actual_dc, color='magenta', s=50, marker='x', 
                         label='Actual', zorder=4, alpha=0.8)
                ax.plot(actual_periods, actual_dc, 'magenta', linestyle='--', 
                      linewidth=1, alpha=0.5)
    
    # Add target angle contour if available
    if target_angle is not None:
        try:
            # Use contour directly on the 2D meshgrid
            cs = ax.contour(X, Y, spec_2d, levels=[target_angle], 
                         colors='white', linewidths=2, linestyles='dashed')
            plt.clabel(cs, inline=True, fontsize=10, fmt='%1.1f°')
            
            # Add to legend
            from matplotlib.lines import Line2D
            target_line = Line2D([0], [0], color='white', linestyle='dashed', linewidth=2)
            handles, labels = ax.get_legend_handles_labels()
            handles.append(target_line)
            labels.append(f'Target Angle: {target_angle}°')
            ax.legend(handles, labels, loc='best')
        except Exception:
            # Fall back to just showing the legend without the target angle contour
            ax.legend(loc='best')
    else:
        ax.legend(loc='best')
    
    # Set axis labels and title
    ax.set_xlabel(lut_x_label, fontsize=12)
    ax.set_ylabel(lut_y_label, fontsize=12)
    ax.set_title(lut_title, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_grating_details(sim: td.Simulation, structures: List[td.Structure], 
                        design_results: Dict[str, Any], p: Dict[str, Any]) -> plt.Figure:
    """
    Create a detailed visualization of the grating structure with period and duty cycle annotations
    for each grating tooth. Shows both the design parameters and actual structure.
    
    Args:
        sim: Tidy3D simulation object
        structures: List of structures in the simulation
        design_results: Dictionary containing longitudinal design results
        p: Parameters dictionary
        
    Returns:
        matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract design parameters
    XwgCenter = design_results.get("XwgCenter", [])
    desDCxx = design_results.get("desDCxx", [])
    desperxx = design_results.get("desperxx", [])
    
    # Filter out invalid values (NaN or duty cycle <= 0)
    valid_indices = ~np.isnan(desperxx) & (desDCxx > 0)
    control_positions = np.array(XwgCenter)[valid_indices]
    control_dc_values = np.array(desDCxx)[valid_indices]
    control_period_values = np.array(desperxx)[valid_indices]
    
    # Get material stack parameters from p
    material_json = p.get('material_json')
    material_stack = {}
    
    if material_json and os.path.exists(material_json):
        try:
            with open(material_json, 'r') as f:
                material_data = json.load(f)
                if 'layer_stack' in material_data:
                    material_stack = material_data['layer_stack']
        except Exception as e:
            print(f"Error loading material JSON file: {e}")
    
    # Check if this is a dual waveguide configuration
    is_dual_waveguide = material_stack.get('is_single_waveguide', True) == False
    
    # Use parameter names exactly as in AO_AIM.json
    wghb = material_stack.get('wghb', p.get('wghb'))  # Bottom waveguide height
    wght = material_stack.get('wght', p.get('wght', 0))  # Top waveguide height
    inth = material_stack.get('inth', p.get('inth', 0))  # Intermediate layer height
    wg_width = p.get('wg_width')  # Waveguide width
    
    # Calculate total waveguide height
    if is_dual_waveguide:
        total_wg_height = wghb + inth + wght
    else:
        total_wg_height = wghb
    
    # Find grating region bounds based on interpolation
    if len(control_positions) >= 2:
        # Create a dense array for interpolation
        min_pos = control_positions[0] 
        max_pos = control_positions[-1]
        # Create a denser array for plotting
        num_points = 200
        interp_positions = np.linspace(min_pos, max_pos, num_points)
        # Interpolate periods and duty cycles
        interp_period_values = np.interp(interp_positions, control_positions, control_period_values)
        interp_dc_values = np.interp(interp_positions, control_positions, control_dc_values)
        
        grating_start = min_pos
        grating_end = max_pos
    else:
        # Fallback if no valid positions
        interp_positions = []
        interp_period_values = []
        interp_dc_values = []
        grating_start = sim.center[0] - sim.size[0]/4
        grating_end = sim.center[0] + sim.size[0]/4
    
    # Extract grating teeth from structures
    teeth_data = []
    for struct in structures:
        if hasattr(struct.geometry, 'center') and hasattr(struct.geometry, 'size'):
            center = struct.geometry.center
            size = struct.geometry.size
            
            # Only consider likely teeth - within grating region and reasonable size
            position = center[0]
            tooth_length = size[0]
            
            # Check if this is a waveguide layer based on z position
            is_bottom_wg = abs(center[2] - wghb/2) < 0.1
            is_top_wg = is_dual_waveguide and abs(center[2] - (wghb + inth + wght/2)) < 0.1
            
            if ((grating_start - 1 <= position <= grating_end + 1) and 
                (is_bottom_wg or is_top_wg) and
                (0.001 < tooth_length < 2.0)):
                
                # Store tooth data
                teeth_data.append({
                    'position': position,
                    'width': tooth_length,
                    'height': wghb if is_bottom_wg else wght,
                    'z_pos': center[2] - size[2]/2,  # Bottom of tooth
                    'is_top': is_top_wg,
                    'x_min': position - tooth_length/2,  # Left edge of tooth
                    'x_max': position + tooth_length/2   # Right edge of tooth
                })
    
    # Sort teeth by position
    teeth_data.sort(key=lambda x: x['position'])
    
    # Calculate actual periods and duty cycles
    if len(teeth_data) > 1:
        for i in range(1, len(teeth_data)):
            period = abs(teeth_data[i]['position']-teeth_data[i]['width']/2 - teeth_data[i-1]['position']+teeth_data[i-1]['width']/2)
            dc = teeth_data[i]['width'] / period if period > 0 else 0
            teeth_data[i]['period'] = period
            teeth_data[i]['duty_cycle'] = dc
        
        # For the first tooth, use the interpolated design value or extrapolate
        if len(interp_positions) > 0:
            # Find the nearest position in the interpolated array
            nearest_idx = np.abs(interp_positions - teeth_data[0]['position']).argmin()
            teeth_data[0]['period'] = interp_period_values[nearest_idx]
            teeth_data[0]['duty_cycle'] = interp_dc_values[nearest_idx]
        elif len(control_period_values) > 0:
            teeth_data[0]['period'] = control_period_values[0]
            teeth_data[0]['duty_cycle'] = control_dc_values[0]
        else:
            teeth_data[0]['period'] = teeth_data[1]['period']
            teeth_data[0]['duty_cycle'] = teeth_data[1]['duty_cycle']
    
    # Print detailed tooth information
    print("\n---------- GRATING TOOTH DETAILS ----------")
    print(f"{'Tooth #':^8} | {'Position (μm)':^15} | {'Length (μm)':^15} | {'Period (μm)':^15} | {'Duty Cycle':^15} | {'Layer':^8}")
    print("-" * 85)
    
    for i, tooth in enumerate(teeth_data):
        position = tooth['position']
        width = tooth['width']
        period = tooth.get('period', 0)
        duty_cycle = tooth.get('duty_cycle', 0)
        layer = "Top" if tooth['is_top'] else "Bottom"
        
        print(f"{i+1:^8} | {position:^15.4f} | {width:^15.4f} | {period:^15.4f} | {duty_cycle:^15.4f} | {layer:^8}")
    
    print("-" * 85)
   
    # Set up plot limits
    x_margin = 5.0  # 5μm margin
    ax.set_xlim(grating_start - x_margin, grating_end + x_margin)
    ax.set_ylim(-1, total_wg_height + 2)
    
    # Plot background layers
    # Bottom region (substrate/BOX)
    ax.add_patch(plt.Rectangle((grating_start - x_margin, -1), 
                              (grating_end - grating_start) + 2*x_margin, 1, 
                              facecolor='lightgray', alpha=0.3))
    
    # Oxide region
    ax.add_patch(plt.Rectangle((grating_start - x_margin, 0), 
                              (grating_end - grating_start) + 2*x_margin, total_wg_height, 
                              facecolor='lightskyblue', alpha=0.2))
    
    # Air region
    ax.add_patch(plt.Rectangle((grating_start - x_margin, total_wg_height), 
                              (grating_end - grating_start) + 2*x_margin, 2, 
                              facecolor='white', alpha=0.1))
    
    # Draw horizontal line at z=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    
    # Draw teeth with annotations
    for i, tooth in enumerate(teeth_data):
        # Draw tooth as rectangle
        if tooth['is_top'] and is_dual_waveguide:
            color = 'darkred'
            z_pos = wghb + inth
        else:
            color = 'darkblue'
            z_pos = 0
        
        # Draw tooth
        tooth_rect = plt.Rectangle(
            (tooth['position'] - tooth['width']/2, z_pos), 
            tooth['width'], tooth['height'],
            facecolor=color, alpha=0.7
        )
        ax.add_patch(tooth_rect)
        
        # Annotate period and duty cycle
        if i > 0:  # Skip first tooth as it doesn't have a well-defined period
            # Calculate position for annotation (midpoint between current and previous tooth)
            prev_tooth = teeth_data[i-1]
            mid_x = (tooth['position'] + prev_tooth['position']) / 2
            
            # Draw period annotation
            if is_dual_waveguide:
                # For dual waveguide, place annotation above both waveguides
                y_pos = total_wg_height + 0.5
            else:
                # For single waveguide, place annotation above the waveguide
                y_pos = wghb + 0.5
            
            # Annotate with period and duty cycle
            period_text = f"P: {tooth['period']:.3f}μm\nDC: {tooth['duty_cycle']:.2f}"
            ax.annotate(period_text, xy=(mid_x, y_pos), xytext=(0, 3),
                       textcoords="offset points", ha='center', va='bottom',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
            
            # Draw a horizontal line with vertical end markers to indicate the period
            # Use the end of previous tooth to the end of current tooth
            line_y = y_pos - 0.3
            
            # Draw the horizontal line from the right edge of previous tooth to the right edge of current tooth
            ax.plot([prev_tooth['x_max'], tooth['x_max']], [line_y, line_y], 
                  color='black', linestyle='-', linewidth=1, alpha=0.7)
            
            # Draw vertical end markers
            marker_height = 0.15
            ax.plot([prev_tooth['x_max'], prev_tooth['x_max']], 
                  [line_y - marker_height, line_y + marker_height], 
                  color='black', linestyle='-', linewidth=1, alpha=0.7)
            ax.plot([tooth['x_max'], tooth['x_max']], 
                  [line_y - marker_height, line_y + marker_height], 
                  color='black', linestyle='-', linewidth=1, alpha=0.7)
    
    # Plot interpolated design curve for reference if we have data
    if len(interp_positions) > 0:
        # Create a secondary axes to show the design parameters
        ax2 = ax.twinx()
        
        # Scale factors to fit the curves in the top part of the plot
        period_scale = 0.3  # Scale factor for period to fit in plot
        dc_scale = 0.8     # Scale factor for duty cycle to fit in plot
        
        y_offset = total_wg_height + 1.0  # Offset to place above the structure
        
        # Plot period curve
        ax2.plot(interp_positions, y_offset + period_scale * interp_period_values, 
               color='blue', linestyle='-', linewidth=2, alpha=0.7, label='Period (Interp.)')
        ax2.scatter(control_positions, y_offset + period_scale * control_period_values, 
                  color='blue', s=30, marker='o', alpha=0.8)
        
        # Plot duty cycle curve
        ax2.plot(interp_positions, y_offset + dc_scale * interp_dc_values, 
               color='red', linestyle='-', linewidth=2, alpha=0.7)
        ax2.scatter(control_positions, y_offset + dc_scale * control_dc_values, 
                  color='red', s=30, marker='o', alpha=0.8)
        
        # Add legend for the interpolated curves
        from matplotlib.lines import Line2D
        period_line = Line2D([0], [0], color='blue', linewidth=2)
        dc_line = Line2D([0], [0], color='red', linewidth=2)
        point_marker = Line2D([0], [0], color='black', marker='o', linestyle='none', markersize=6)
        
        ax2.legend([period_line, dc_line, point_marker], 
                 ['Period (Interp.)', 'Duty Cycle (Interp.)', 'Control Points'], 
                 loc='upper right', framealpha=0.7)
        
        # Hide y-axis for the secondary axis
        ax2.set_yticks([])
    
    # Get wavelength and material from parameters
    wavelength = p.get('wavelength', 0) * 1000  # Convert to nm
    material_name = p.get('material', 'Unknown')
    
    # Add a title with grating information
    title = f"{material_name} {'Dual' if is_dual_waveguide else 'Single'} Waveguide Grating at λ={wavelength:.1f}nm"
    if len(teeth_data) > 0:
        avg_period = np.mean([t.get('period', 0) for t in teeth_data[1:]])
        avg_dc = np.mean([t.get('duty_cycle', 0) for t in teeth_data[1:]])
        title += f"\nAvg. Period: {avg_period:.3f}μm, Avg. DC: {avg_dc:.2f}, Teeth: {len(teeth_data)}"
    
    ax.set_title(title, fontsize=14)
    
    # Add legend
    if is_dual_waveguide:
        bottom_patch = plt.Rectangle((0, 0), 1, 1, facecolor='darkblue', alpha=0.7)
        top_patch = plt.Rectangle((0, 0), 1, 1, facecolor='darkred', alpha=0.7)
        ax.legend([bottom_patch, top_patch], ['Bottom Waveguide', 'Top Waveguide'], 
                loc='upper left', framealpha=0.7)
    
    plt.tight_layout()
    return fig


def simulate_grating(design_results: Dict[str, Any], p: Dict[str, Any], 
                   output_dir: str = "output", task_name: str = None,
                   show_plots: bool = True, run_sim: bool = True,
                   material_index: float = None, material_csv: str = None,
                   oxide_index: float = None, oxide_csv: str = None,
                   max_num_poles: int = 1, material_json: str = None) -> Dict[str, Any]:
    """
    Run end-to-end grating simulation from design to analysis.
    
    Args:
        design_results: Design results from longitudinal_design module
        p: Simulation parameters
        output_dir: Directory to save results
        task_name: Task name for the simulation
        show_plots: Whether to show plots
        run_sim: Whether to run the simulation (or just visualize)
        material_index: Override material index
        material_csv: CSV file with material data
        oxide_index: Override oxide index
        oxide_csv: CSV file with oxide data
        max_num_poles: Max number of poles for fitting
        material_json: Path to JSON file containing material stack parameters
        
    Returns:
        Dictionary with simulation results
    """
    # Create a copy of parameters to avoid modifying the original
    p_sim = p.copy()
    
    # Update parameter dictionary with additional parameters
    if material_index is not None:
        p_sim['material_index'] = material_index
    
    if oxide_index is not None:
        p_sim['oxide_index'] = oxide_index
    
    # Add material and oxide CSV files if provided
    if material_csv is not None:
        p_sim['material_csv'] = material_csv
    
    if oxide_csv is not None:
        p_sim['oxide_csv'] = oxide_csv
    
    # Set the material JSON file if provided
    if material_json is not None:
        p_sim['material_json'] = material_json
        print(f"Using material stack configuration from: {material_json}")
        
        # Load the material JSON to check for dual waveguide configuration
        try:
            with open(material_json, 'r') as f:
                material_data = json.load(f)
                if 'layer_stack' in material_data:
                    if material_data['layer_stack'].get('is_single_waveguide', True) == False:
                        print("Detected DUAL waveguide configuration from JSON file")
                    else:
                        print("Detected SINGLE waveguide configuration from JSON file")
        except Exception as e:
            print(f"Error checking material JSON file: {e}")
    
    # Convert wavelength to frequency
    if 'wavelength' in p_sim:
        wavelength = p_sim['wavelength']
        p_sim['frequency'] = td.C_0 / wavelength
    
    # Generate a unique task name if not provided
    if task_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mat_name = p_sim.get('material', 'unknown')
        wl_nm = int(p_sim.get('wavelength', 0) * 1000)
        task_name = f"grating_{mat_name}_wl{wl_nm}nm_{timestamp}"
    
    # Add the task name to parameters
    p_sim['task_name'] = task_name
    
    # Add the max number of poles for fitting
    p_sim['max_num_poles'] = max_num_poles
    
    # Create the simulation directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the output directory for this task
    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Save the parameters
    params_file = os.path.join(task_dir, "parameters.json")
    with open(params_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        p_json = {}
        for k, v in p_sim.items():
            if isinstance(v, np.ndarray):
                p_json[k] = v.tolist()
            else:
                p_json[k] = v
        
        json.dump(p_json, f, indent=2)
    
    # Save the design results
    design_file = os.path.join(task_dir, "design_results.npz")
    np.savez(design_file, **design_results)
    
    # Create the simulation
    simulation, structures = create_simulation(design_results, p_sim)
    
    # Save the simulation
    sim_file = os.path.join(task_dir, "simulation.json")
    simulation.to_json(sim_file)
    
    # Visualize the simulation
    if show_plots:
        fig_sim = visualize_simulation(simulation, structures, design_results, p_sim)

        # Create the detailed grating visualization
        fig_grating_details = plot_grating_details(simulation, structures, design_results, p_sim)

        freqs = [td.C_0 / wavelength]
        fig_mat = plot_material_indices(simulation, freqs)

        fig_grating = plot_grating_parameters(design_results, structures)

        # Check if LUT file is provided in parameters
        lut_file = p_sim.get('lut_file_path')
        fig_lut = plot_grating_on_lut(design_results, lut_file, structures)
                
        # Save the visualization figures
        for i, fig in enumerate(fig_sim):
            fig_file = os.path.join(task_dir, f"simulation_vis_{i+1}.png")
            fig.savefig(fig_file, dpi=300)
        
        # Save the detailed grating visualization
        fig_file = os.path.join(task_dir, "grating_details.png")
        fig_grating_details.savefig(fig_file, dpi=300)
        
        # Show figures if interactive mode
        if show_plots == "show":
            plt.show()
    
    # Run the simulation if requested
    results = None
    if run_sim:
        results_path = run_simulation(simulation, task_name, output_dir)
        
        # Analyze the results
        analysis_results = analyze_results(results_path, design_results, p_sim)
        
        # Save the analysis results
        analysis_file = os.path.join(task_dir, "analysis_results.npz")
        np.savez(analysis_file, **analysis_results)
        
        # Visualize the results
        if show_plots:
            fig_res = visualize_results(analysis_results, design_results, p_sim)
            
            # Save the visualization figures
            for i, fig in enumerate(fig_res):
                fig_file = os.path.join(task_dir, f"results_vis_{i+1}.png")
                fig.savefig(fig_file, dpi=300)
            
            # Show figures if interactive mode
            if show_plots == "show":
                plt.show()
        
        # Return the results
        results = {
            'simulation': simulation,
            'structures': structures,
            'results_path': results_path,
            'analysis_results': analysis_results,
            'task_dir': task_dir
        }
    else:
        # Only return the simulation objects
        results = {
            'simulation': simulation,
            'structures': structures,
            'task_dir': task_dir
        }
    
    return results 