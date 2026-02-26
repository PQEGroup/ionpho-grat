import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import json
import gdstk

# Import modules
from simulation_setup import setup_simulation, add_mode_source, load_simulation_config
from load_material import load_material_json, get_material_params, get_optical_properties
from plot_helpers import plot_combined_view

def load_gds_file(file_path):
    """Load GDS file using gdstk."""
    try:
        gdsii = gdstk.read_gds(file_path)
        gdsii_cell = gdsii.cells
        print(f"Loaded GDS file with cells: {[cell.name for cell in gdsii_cell]}")
        return gdsii
    except Exception as e:
        print(f"Error loading GDS file: {e}")
        return None

def process_gds_file(gds_file_path, config_file, run_sim=False, material_file=None, gds_config_file=None):
    """Process GDS file and create simulation.
    
    Parameters:
    -----------
    gds_file_path : str
        Path to GDS file with grating geometry
    config_file : str
        Path to simulation configuration JSON file (required)
    run_sim : bool
        Whether to run the simulation
    material_file : str
        Path to material JSON file (required)
    gds_config_file : str
        Path to GDS configuration JSON file (required)
    """
    # Config files must be provided
    if not config_file:
        print("ERROR: Simulation configuration file must be provided")
        return None
        
    if not gds_config_file:
        print("ERROR: GDS configuration file must be provided")
        return None
        
    # Check if config files exist
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file not found: {config_file}")
        return None
        
    if not os.path.exists(gds_config_file):
        print(f"ERROR: GDS configuration file not found: {gds_config_file}")
        return None
        
    # Material file must be provided
    if not material_file:
        print("ERROR: Material file must be provided")
        return None
    
    # Check if material file exists
    if not os.path.exists(material_file):
        print(f"ERROR: Material file not found: {material_file}")
        return None
    
    # Load the GDS file
    gdsii = load_gds_file(gds_file_path)
    if gdsii is None:
        print(f"Failed to load GDS file: {gds_file_path}")
        return None
    
    # Load the GDS config file
    with open(gds_config_file, 'r') as f:
        gds_config = json.load(f)
    
    # Load the simulation config
    with open(config_file, 'r') as f:
        sim_config = json.load(f)
    
    # Extract GDS import parameters
    gds_layer = gds_config.get('gds_layer')
    gds_dtype = gds_config.get('gds_dtype')
    cell_name = gds_config.get('cell_name')
    
    # Get grating parameters from GDS config
    theta_inc_degrees = gds_config.get('thetaIncDegrees')  # Emission angle in degrees
    wavelength = gds_config.get('wavelength')  # Wavelength in um
    
    # Check wg_width consistency between configs
    wg_width_sim = sim_config["simulation"]["wg_width"]
    wg_width_gds = gds_config.get('wg_width')
    if wg_width_sim != wg_width_gds:
        print(f"ERROR: wg_width mismatch - simulation_config has {wg_width_sim}, gds_config has {wg_width_gds}")
        return None
    wg_width = wg_width_sim
    
    # Load the appropriate cell
    if cell_name:
        cell = next((c for c in gdsii.cells if c.name == cell_name), None)
        if not cell:
            print(f"Cell '{cell_name}' not found in GDS file")
            return None
    else:
        # Use the first cell if no name specified
        cell = gdsii.cells[0]
        print(f"Using first cell: {cell.name}")
    
    # Load material data
    material_data = load_material_json(material_file)
    if not material_data:
        print(f"Failed to load material file: {material_file}")
        return None
    
    # Get material parameters
    material_params = get_material_params(material_file)
    if not material_params:
        print(f"Failed to extract parameters from material file: {material_file}")
        return None
        
    wgh_low, wgh_high, inth, _ = material_params
    print(f"wgh_low: {wgh_low}, wgh_high: {wgh_high}, inth: {inth}")
    
    # Determine dual layer setting from material
    dual_layer = not material_data.get('layer_stack', {}).get('is_single_waveguide')
    print(f"Using {'dual' if dual_layer else 'single'} layer waveguide structure based on material file")
    
    # Create only the necessary fields in p and mat dictionaries
    # that are actually used by setup_simulation:
    
    # In setup_simulation, these fields from 'p' are used:
    # - 'lambda': for wavelength
    # - 'wg_width': for waveguide width
    # - 'thetaIncDegrees': for calculating far field monitor positions
    p = {
        'lambda': wavelength,
        'wg_width': wg_width,
        'thetaIncDegrees': theta_inc_degrees
    }
    
    # Get bounding box dimensions from the GDS cell
    gds_bbox = cell.bounding_box()
    print(f"GDS bounding box: {gds_bbox}, cell: {cell}")
    # Calculate dimensions from bounding box
    grat_length = gds_bbox[1][0] - gds_bbox[0][0]
    grat_width = gds_bbox[1][1] - gds_bbox[0][1]
    
    # Calculate taper angle (in radians) based on GDS dimensions
    # The taper angle is the angle between the grating edge and the propagation axis
    tapang = np.arctan2(grat_width/2, grat_length)
    print(f"Calculated taper angle from GDS dimensions: {np.degrees(tapang):.2f} degrees")
    
    # In setup_simulation, these fields from 'mat' are used:
    # - 'tapang': grating angle (in radians)
    # - 'tapl': taper length
    # - 'lgrat': grating length
    mat = {
        'tapang': tapang,  # Calculated from GDS dimensions
        'tapl': grat_length / 3,   # Estimate taper length as 1/3 of total length
        'lgrat': grat_length * 2/3  # Estimate grating length as 2/3 of total length
    }
    # What matters above is the sum of tapl and lgrat.
    
    # Get optical properties from material data based on wavelength
    _, _, wg_index = get_optical_properties(material_data, wavelength)
    print(f"Using wavelength {wavelength} um: wg_index={wg_index}")
    
    # Create waveguide material medium
    wg_material = td.Medium(permittivity=wg_index**2)
    
    # Create GDS-based structures as proper Tidy3D structures
    tidy3d_structures = []
    
    # Bottom layer geometry
    bottom_layer_geometry = td.Geometry.from_gds(
        cell,
        gds_layer=gds_layer,
        gds_dtype=gds_dtype,
        axis=2,
        slab_bounds=(0, wgh_low),
        reference_plane="bottom",
    )
    
    # Add bottom layer structure
    bottom_structure = td.Structure(geometry=bottom_layer_geometry, medium=wg_material)
    tidy3d_structures.append(bottom_structure)
    
    # If dual layer, add top layer geometry
    if dual_layer and wgh_high > 0:
        # Check if there's a specific top layer in the GDS
        top_layer = gds_config.get('top_layer')
        top_dtype = gds_config.get('top_dtype')
        
        if top_layer is not None:
            # If top layer specified, use it
            top_layer_geometry = td.Geometry.from_gds(
                cell,
                gds_layer=top_layer,
                gds_dtype=top_dtype if top_dtype is not None else gds_dtype,
                axis=2,
                slab_bounds=(wgh_low + inth, wgh_low + inth + wgh_high),
                reference_plane="bottom",
            )
            top_structure = td.Structure(geometry=top_layer_geometry, medium=wg_material)
            tidy3d_structures.append(top_structure)
            print(f"Added top layer from GDS using layer {top_layer}")
        else:
            # If no specific top layer, use same layer but shifted in z
            top_layer_geometry = td.Geometry.from_gds(
                cell,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
                axis=2,
                slab_bounds=(wgh_low + inth, wgh_low + inth + wgh_high),
                reference_plane="bottom",
            )
            top_structure = td.Structure(geometry=top_layer_geometry, medium=wg_material)
            tidy3d_structures.append(top_structure)
            print(f"Added top layer by shifting bottom layer in z")
    
    # Extract the geometries from the structures for setup_simulation
    gds_geometries = [s.geometry for s in tidy3d_structures]
    
    # Get PEC layer config if present
    pec_layer_config = gds_config.get('pec_layer')
    
    # Use the existing setup_simulation function with our GDS-based geometries
    sim = setup_simulation(
        p, mat, [], [],  # Empty list for tapers as they are included in GDS
        dual_layer=dual_layer, 
        wgh_low=wgh_low, 
        wgh_high=wgh_high, 
        inth=inth,
        material_data=material_data,
        config_file=config_file,
        gds_structures=tidy3d_structures,
        pec_layer_config=pec_layer_config
    )
    
    # Add mode source
    sim_with_source, ms, modes, source_time = add_mode_source(sim, wavelength)
    
    # Plot simulation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))
    ax1 = sim_with_source.plot(z=wgh_low / 2, lw=1, edgecolor="k", ax=ax1)
    ax2 = sim_with_source.plot(y=0, lw=1, edgecolor="k", ax=ax2)
    plt.show()
    
    # Calculate frequency from wavelength
    freq = td.C_0 / wavelength
    
    # Plot combined view with all information
    fig3 = plot_combined_view(sim_with_source, source_time, freq)
    plt.figure(fig3.number)
    plt.show()
    
    # Calculate simulation cost regardless of whether we run it
    try:
        import tidy3d.web as web
        job_name = os.path.basename(gds_file_path).split('.')[0]
        
        # Get the directory of the GDS file to save output there
        gds_file_dir = os.path.dirname(os.path.abspath(gds_file_path))
        
        job = web.Job(simulation=sim_with_source, task_name=job_name, verbose=True)
        
        # Estimate and display cost
        estimated_cost = web.estimate_cost(job.task_id)
        print(f"Estimated maximum cost: {estimated_cost:.3f} Flex Credits")
        
        # Run simulation if requested
        if run_sim:
            # Get confirmation before running
            user_input = input("Run simulation? (y/n): ")
            if user_input.lower() == 'y':
                # Save to GDS file directory instead of script directory
                output_path = os.path.join(gds_file_dir, f"{job_name}.hdf5")
                sim_data = job.run(path=output_path)
                print(f"Simulation completed and saved to {output_path}")
                return sim_data
            else:
                print("Simulation not run")
                return sim_with_source
    except Exception as e:
        print(f"Error estimating/running simulation: {e}")
        return sim_with_source
    
    return sim_with_source

def main():
    """Main function to parse command line arguments and run simulation."""
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python loadGdsFile.py <gds_file_path> <material_file> <config_file> <gds_config_file> [--run]")
        sys.exit(1)
        
    gds_file_path = sys.argv[1]
    material_file = sys.argv[2]
    config_file = sys.argv[3]
    gds_config_file = sys.argv[4]
    
    run_sim = "--run" in sys.argv
    
    process_gds_file(
        gds_file_path, 
        config_file=config_file,
        run_sim=run_sim,
        material_file=material_file,
        gds_config_file=gds_config_file
    )

if __name__ == "__main__":
    main()