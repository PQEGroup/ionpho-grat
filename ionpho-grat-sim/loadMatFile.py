import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import json
import mat73
import scipy.io

# Import modules
from grating_structures import create_grating_var_interp_quartic, create_taper
from simulation_setup import setup_simulation, add_mode_source, load_simulation_config
from load_material import load_material_json, get_layer_parameters, get_optical_properties, get_simulation_parameters, get_material_params
from plot_helpers import plot_material_index_profile, plot_mode_source_spectrum, plot_combined_view
from gds_output import create_gds_with_separate_layers

def load_mat_file(file_path):
    """Load mat file using mat73 for v7.3 format."""
    try:
        # Load with mat73 for v7.3 format
        mat = mat73.loadmat(file_path)
        print("Loaded with mat73")
        return mat
    except Exception as e:
        print(f"Error loading mat file: {e}")
        return None

def process_mat_file(mat_file_path, config_file, run_sim=False, material_file=None, create_gds=True):
    """Main function to process mat file and create simulation.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to mat file with grating geometry
    config_file : str
        Path to simulation configuration JSON file (required)
    run_sim : bool
        Whether to run the simulation
    material_file : str
        Path to material JSON file (required)
    create_gds : bool
        Whether to create GDS output file (default: True)
    """
    # Config file must be provided
    if not config_file:
        print("ERROR: Configuration file must be provided")
        return None
        
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file not found: {config_file}")
        return None
        
    # Material file must be provided
    if not material_file:
        print("ERROR: Material file must be provided")
        return None
    
    # Check if material file exists
    if not os.path.exists(material_file):
        print(f"ERROR: Material file not found: {material_file}")
        return None
    
    # Load the mat file
    mat = load_mat_file(mat_file_path)
    if mat is None:
        print(f"Failed to load mat file: {mat_file_path}")
        return None
    
    # Extract parameters
    p = mat['p']
    print("Parameters loaded:", p)
    
    # Load material data
    material_data = load_material_json(material_file)
    if not material_data:
        print(f"Failed to load material file: {material_file}")
        return None
    
    # Get material name from the mat file and check for mismatch
    mat_material = p.get('material')
    material_name = material_data.get('material_name')
    
    if mat_material and material_name and mat_material.lower() != material_name.lower():
        print(f"WARNING: Material in mat file ({mat_material}) doesn't match material file ({material_name})")
        # Ask for confirmation
        user_input = input("Continue with different materials? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting due to material mismatch")
            return None
    
    print(f"Loaded material: {material_data.get('process_name', 'Unknown')} - {material_name or 'Unknown'}")
    
    # Get material parameters
    material_params = get_material_params(material_file)
    if not material_params:
        print(f"Failed to extract parameters from material file: {material_file}")
        return None
        
    wgh_low, wgh_high, inth, _ = material_params
    
    # Determine dual layer setting from material
    dual_layer = not material_data.get('layer_stack', {}).get('is_single_waveguide', True)
    print(f"Using {'dual' if dual_layer else 'single'} layer waveguide structure based on material file")
        
    # Convert to um for tidy3d
    wgh_low_um = wgh_low 
    wgh_high_um = wgh_high 
    inth_um = inth 
    
    # Extract grating parameters
    minSize = p.get('minSize')
    tapStart = mat.get('tapStart')
    tapl_SI = mat.get('tapl')
    gratl = mat.get('lgrat')
    tapang_in = mat.get('tapang')
    r0_SI = mat.get('r0')
    dr = mat.get('dr')
    
    # Extract coordinate arrays - only supporting latest format
    xPtsDC = np.squeeze(mat['xPtsBothFull'])
    xPtsPer = np.squeeze(mat['xPtsBothFull'])
    yPtsPer = np.squeeze(mat['yPtsPerFull'])
    yPtsDC = np.squeeze(mat['yPtsDCFull'])
    
    # Quartic curvature parameters
    print(mat.keys())
    xPtsRadsQ = np.squeeze(mat['xPtsRadsQ'])
    yPtsRadsQ_R = np.squeeze(mat['yPtsRadsQ_R'])
    yPtsRadsQ_A = np.squeeze(mat['yPtsRadsQ_A'])
    
    # Calculate total waveguide height
    wg_max_z = wgh_low_um + (inth_um + wgh_high_um if dual_layer else 0)
    etch_depth = wgh_low_um  # Default etch depth is the bottom waveguide height
    
    # Create grating structures
    dim = 3  # 3D simulation
    etches = create_grating_var_interp_quartic(
        dim, tapl_SI, gratl, tapang_in, tapStart, r0_SI, dr, 
        xPtsRadsQ, yPtsRadsQ_R, yPtsRadsQ_A, xPtsPer, xPtsDC, 
        yPtsPer, yPtsDC, minSize, wg_max_z, etch_depth,
        wgh_low_um, wgh_high_um, inth_um, dual_layer
    )
    
    # Create taper(s)
    sim_config = load_simulation_config(config_file)
    wg_width = sim_config["simulation"]["wg_width"]  # Waveguide width in um
    wg_end = gratl + tapl_SI + 1  # End position with some margin
    
    if dual_layer:
        tapers = create_taper(
            wg_width, 0, wgh_low_um, tapStart, tapang_in, r0_SI, dr, wg_end+1,
            dual_layer=True, zmin_top=wgh_low_um+inth_um, zmax_top=wgh_low_um+inth_um+wgh_high_um
        )
    else:
        tapers = create_taper(
            wg_width, 0, wgh_low_um, tapStart, tapang_in, r0_SI, dr, wg_end+1
        )
    
    # Create GDS file with tapers and etches on separate layers
    if create_gds:
        print("Creating GDS output file...")
        try:
            # Convert geometries to structures for GDS output
            taper_structures = [{'geometry': taper} for taper in tapers]
            etch_structures = [{'geometry': etch} for etch in etches]
            
            # Create GDS file
            gds_path = create_gds_with_separate_layers(
                taper_structures, 
                etch_structures, 
                mat_file_path,
                taper_layer=(1, 0),  # Layer 1 for tapers
                etch_layer=(2, 0)    # Layer 2 for etches
            )
            print(f"GDS file created successfully: {gds_path}")
        except Exception as e:
            print(f"Warning: Could not create GDS file: {e}")
            print(f"Note: You may need to install gdstk: pip install gdstk")
    
    # Setup simulation with material_data, using configuration parameters
    sim = setup_simulation(
        p, mat, etches, tapers,
        dual_layer=dual_layer, 
        wgh_low=wgh_low, 
        wgh_high=wgh_high, 
        inth=inth,
        material_data=material_data,
        config_file=config_file
    )
    
    # Add mode source
    wavelength = p.get('lambda') 
    sim_with_source, ms, modes, source_time = add_mode_source(sim, wavelength)
    
    # Plot simulation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))
    ax1 = sim_with_source.plot(z=wgh_low_um / 2, lw=1, edgecolor="k", ax=ax1)
    ax2 = sim_with_source.plot(y=0, lw=1, edgecolor="k", ax=ax2)
    plt.show()
    
    # Calculate frequency from wavelength
    freq = td.C_0 / (wavelength)
    
    ## Plot refractive index profile
    #fig1, ax1 = plot_material_index_profile(sim_with_source, freq)
    #plt.figure(fig1.number)
    #plt.show()
    
    ## Plot mode source spectrum
    #fig2, ax2 = plot_mode_source_spectrum(source_time, freq)
    #plt.figure(fig2.number)
    #plt.show()
    
    # Plot combined view with all information
    fig3 = plot_combined_view(sim_with_source, source_time, freq)
    plt.figure(fig3.number)
    plt.show()
    
    # Calculate simulation cost regardless of whether we run it
    try:
        import tidy3d.web as web
        job_name = os.path.basename(mat_file_path).split('.')[0]
        
        # Get the directory of the mat file to save output there
        mat_file_dir = os.path.dirname(os.path.abspath(mat_file_path))
        
        job = web.Job(simulation=sim_with_source, task_name=job_name, verbose=True)
        
        # Estimate and display cost
        estimated_cost = web.estimate_cost(job.task_id)
        print(f"Estimated maximum cost: {estimated_cost:.3f} Flex Credits")
        
        # Run simulation if requested
        if run_sim:
            # Get confirmation before running
            user_input = input("Run simulation? (y/n): ")
            if user_input.lower() == 'y':
                # Save to mat file directory instead of script directory
                output_path = os.path.join(mat_file_dir, f"{job_name}.hdf5")
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
        print("Usage: python loadMatFile.py <mat_file_path> <material_file> <config_file> [--run]")
        sys.exit(1)
        
    mat_file_path = sys.argv[1]
    material_file = sys.argv[2]
    config_file = sys.argv[3]
    
    run_sim = "--run" in sys.argv
    
    process_mat_file(
        mat_file_path, 
        config_file=config_file,
        run_sim=run_sim,
        material_file=material_file
    )

if __name__ == "__main__":
    main()