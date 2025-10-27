import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import scipy.io as sio
import mat73
import math
import argparse

# Add path to the tidy3d module if needed
sys.path.append(os.path.dirname(__file__))
from material_helper import get_material_params, load_material_data, validate_material_name
from simulation_helper import (getfilename, create_grating_structure, create_monitors, 
                              create_farfield_monitor, create_mode_source, create_simulation,
                              add_source_to_simulation, setup_simulation, get_material_medium)
from plot_utils import (plot_simulation_geometry, plot_mode_fields, plot_source_spectrum, 
                      plot_mode_profile_and_spectrum, plot_material_dispersion_from_csv,
                      plot_material_indices_visualization)

# Main simulation function
def run_grating_simulation(material, output_dir, period, gratDC, toxt, boxt, wghb, tair, tSi, 
                       wg_w, l_pre, l_post, run_time, 
                       lambdas, freqs, material_file, csv_file, oxide_csv_file=None, oxide_index=None,
                       include_2d_monitor=False, show_plots=True,
                       N=60, grid_dl=0.01, addl_str='', run_sim=True, max_num_poles=1,
                       use_pml=True, absorber_layers=60, wght=None, inth=None, use_pec_bottom=False):
    """Run a single grating simulation with the provided parameters
    
    Args:
        material: Material name for the waveguide
        output_dir: Directory to save output files
        period: Grating period in microns
        gratDC: Grating duty cycle (0.0-1.0)
        toxt: Top oxide thickness
        boxt: Bottom oxide thickness
        wghb: Bottom waveguide height
        tair: Air thickness
        tSi: Silicon thickness
        wg_w: Waveguide width
        l_pre: Pre-grating length
        l_post: Post-grating length
        run_time: Simulation run time
        lambdas: List of wavelengths (nm)
        freqs: List of frequencies (Hz)
        material_file: Path to material JSON file
        csv_file: Path to CSV file with waveguide refractive index data
        oxide_csv_file: Path to CSV file with oxide refractive index data (default: None)
        oxide_index: Fixed refractive index value for oxide if CSV not provided (default: None)
        include_2d_monitor: Whether to include a 2D XZ field monitor (default: False)
        show_plots: Whether to display plots (default: True)
        N: Number of grating periods (default: 60)
        grid_dl: Grid spacing for simulation mesh (default: 0.01)
        addl_str: Additional string tag for file naming (default: '')
        run_sim: Whether to actually run the simulation (default: True)
               If False, sets up the simulation but does not run it
        max_num_poles: Maximum number of poles to use in dispersion fitting (default: 1)
        use_pml: Whether to use PML boundary conditions for X direction (default: True)
               If False, uses absorber boundary conditions instead, which may help with convergence
        absorber_layers: Number of absorber layers if use_pml is False (default: 60)
        wght: Top waveguide height (for dual waveguide, default: None)
        inth: Intermediate layer height between waveguides (for dual waveguide, default: None)
        use_pec_bottom: Whether to use PEC boundary condition at bottom instead of silicon substrate (default: False)
    
    Note:
        The simulation uses dispersive materials that model wavelength-dependent 
        refractive indices for each material. This ensures accurate simulation across
        the frequency spectrum of the input pulse.
    """
    # Check if we have a dual waveguide configuration
    is_dual_waveguide = wght is not None and inth is not None and wght > 0 and inth > 0
    
    # Print material file and CSV file paths
    print(f"Using material file in simulation: {material_file}")
    print(f"Using dispersion data from CSV file: {csv_file}")
    
    # Print oxide information
    if oxide_csv_file:
        print(f"Using oxide dispersion data from CSV file: {oxide_csv_file}")
    elif oxide_index is not None:
        print(f"Using fixed oxide refractive index: {oxide_index}")
    else:
        print("Using default Tidy3D SiO2 (Horiba) for oxide material")
    
    # Print boundary conditions
    if use_pml:
        print(f"Using PML boundary conditions")
    else:
        print(f"Using absorber boundary conditions with {absorber_layers} layers")
        print("Note: This may reduce simulation divergence but might increase reflections")
    
    if use_pec_bottom:
        print(f"Using PEC boundary condition at bottom surface (no silicon substrate)")
    else:
        print(f"Using silicon substrate at bottom surface")
    
    # Print waveguide configuration details
    if is_dual_waveguide:
        print(f"Using dual waveguide configuration:")
        print(f"  Bottom waveguide height (wghb): {wghb} μm")
        print(f"  Top waveguide height (wght): {wght} μm")
        print(f"  Intermediate layer height (inth): {inth} μm")
    else:
        print(f"Using single waveguide configuration:")
        print(f"  Waveguide height (wghb): {wghb} μm")
    
    pertDC = 1 - gratDC
    
    # Convert to strings for filename with integer rounding
    perStr = f"{int(round(period*1e3))}"  # Convert microns to nm and round to integer
    DCStr = f"{pertDC:.2f}"   # Perturbation duty cycle (1-gratDC)
    
    NStr = f"{N}"  # Number of periods as string

    # Set up the complete simulation using the helper function
    sim, l_grat, modes, mode_solver, oxide_source = setup_simulation(
        material, material_file, period, gratDC, N, l_pre, l_post,
        wg_w, boxt, toxt, tSi, tair, freqs, run_time, 
        include_2d_monitor=include_2d_monitor, grid_dl=grid_dl,
        csv_file=csv_file, oxide_csv_file=oxide_csv_file, oxide_index=oxide_index,
        max_num_poles=max_num_poles, use_pml=use_pml,
        absorber_layers=absorber_layers,
        wghb=wghb, wght=wght, inth=inth, use_pec_bottom=use_pec_bottom
    )
    
    # Use the plot_utils function to plot simulation geometry with material annotations
    if show_plots:
        # Plot material dispersion with fitting from the CSV file
        plot_material_dispersion_from_csv(csv_file, material, figsize=(10, 6), max_num_poles=max_num_poles)
        
        # Plot oxide dispersion if a CSV file is provided
        if oxide_csv_file:
            plot_material_dispersion_from_csv(oxide_csv_file, "Oxide", figsize=(10, 6), max_num_poles=max_num_poles)
        
        # Plot simulation geometry
        plot_simulation_geometry(
            sim, material, material_file, l_grat, period, gratDC, N,
            wghb=wghb, boxt=boxt, toxt=toxt, tair=tair, tSi=tSi, wg_w=wg_w, mode_solver=mode_solver,
            is_dual_waveguide=is_dual_waveguide, wght=wght, inth=inth
        )
        
        # Explicitly plot material indices in a separate figure
        plot_material_indices_visualization(
            sim, freqs,
            figsize=(10, 6),
            oxide_source=oxide_source
        )
        
        # Plot mode profile and source spectrum
        plot_mode_profile_and_spectrum(
            modes, freqs, sim.sources[0].source_time,
            sim=sim, 
            wghb=wghb, 
            wght=wght if is_dual_waveguide else None, 
            inth=inth if is_dual_waveguide else None,
            is_dual_waveguide=is_dual_waveguide
        )
        
        # Show all plots
        plt.show()
    
    # Generate filename based on waveguide configuration
    if is_dual_waveguide:
        fname = getfilename(output_dir, lambdas[0], material, perStr, DCStr, l_grat, boxt, wghb, wght, inth, 0, toxt, N)
    else:
        fname = getfilename(output_dir, lambdas[0], material, perStr, DCStr, l_grat, boxt, wghb, 0, 0, 0, toxt, N)
    
    # Run simulation
    import tidy3d.web as web
    task_name = f"grating_{perStr}nm_pertDC{DCStr}_N{NStr}"
    if is_dual_waveguide:
        task_name += f"_dual_wg"
    if addl_str:
        task_name += f"_{addl_str}"
    
    job = web.Job(simulation=sim, task_name=task_name)
    estimated_cost = web.estimate_cost(job.task_id)

    print(f"Simulation size: {sim.num_cells} cells")
    print(f"Time steps: {sim.num_time_steps}")
    print(f'The estimated maximum cost is {estimated_cost:.3f} Flex Credits.')
    
    # Print oxide source information
    print(f"Oxide medium source: {oxide_source}")
    
    # Only run the simulation if run_sim is True
    if run_sim:
        print(f"Running simulation...")
        sim_data = job.run(path=fname + "_results.hdf5")
        print(f"Simulation completed and saved to {fname}_results.hdf5")
    else:
        print(f"Simulation setup completed but not run (run_sim=False)")
        print(f"To run this simulation later, use the task ID: {job.task_id}")
        sim_data = None  # No simulation data when not running
    
    print(f"Created simulation for grating period {perStr}nm, grating DC={gratDC:.2f}, perturbation DC={pertDC:.2f}, N={N}")
    
    return fname, sim_data

# Main function with argparse
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a single grating coupler simulation with specific parameters')
    parser.add_argument('--material-file', type=str, required=True,
                        help='Full path to material JSON file (required).')
    parser.add_argument('--csv-file', type=str, required=True,
                        help='Full path to CSV file with wavelength and refractive index data (required).')
    parser.add_argument('--oxide-csv-file', type=str, default=None,
                        help='Full path to CSV file with oxide refractive index data (optional).')
    parser.add_argument('--oxide-index', type=float, default=None,
                        help='Fixed refractive index for oxide if CSV not provided (optional).')
    parser.add_argument('--max-num-poles', type=int, default=1,
                       help='Maximum number of poles to use in dispersion fitting (default: 1)')
    parser.add_argument('--use-absorber', action='store_true',
                       help='Use absorber boundary conditions instead of PML for X direction (useful if PML doesnt converge)')
    parser.add_argument('--absorber-layers', type=int, default=60,
                       help='Number of absorber layers if using absorber (default: 60)')
    parser.add_argument('--use-pec-bottom', action='store_true',
                       help='Use PEC boundary condition at bottom surface instead of silicon substrate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files. Defaults to process_name/material_name.')
    parser.add_argument('--include-2d-monitor', action='store_true',
                        help='Include a 2D field monitor in the XZ plane (increases memory usage)')
    parser.add_argument('--period', type=float, default=0.55,
                        help='Grating period in microns (default: 0.55)')
    parser.add_argument('--gratDC', type=float, default=0.5,
                        help='Grating duty cycle (default: 0.5)')
    # Add simulation parameters that were previously in material file
    parser.add_argument('--l-pre', type=float, default=5.0,
                        help='Pre-grating length in microns (default: 5.0)')
    parser.add_argument('--l-post', type=float, default=5.0,
                        help='Post-grating length in microns (default: 5.0)')
    parser.add_argument('--run-time', type=float, default=0.55e-12,
                        help='Simulation run time in seconds (default: 0.55e-12)')
    parser.add_argument('--N', type=int, default=60,
                        help='Number of grating periods (default: 60)')
    parser.add_argument('--grid-dl', type=float, default=0.01,
                        help='Grid spacing for simulation mesh (default: 0.01)')
    parser.add_argument('--addl-str', type=str, default='',
                        help='Additional string tag for file naming (default: "")')
    parser.add_argument('--wavelengths', type=float, nargs='+',
                        help='Custom wavelengths in um to simulate (default: use all wavelengths from CSV)')
    parser.add_argument('--no-run', action='store_true',
                       help='Set up the simulation but do not run it (for debugging)')
    
    args = parser.parse_args()

    # Get the material file path
    material_file = args.material_file
    csv_file = args.csv_file
    oxide_csv_file = args.oxide_csv_file
    oxide_index = args.oxide_index
    
    print(f"Using material file: {material_file}")
    print(f"Using CSV file with refractive index data: {csv_file}")
    
    # Handle oxide configuration
    if oxide_csv_file:
        print(f"Using oxide CSV file with refractive index data: {oxide_csv_file}")
    elif oxide_index is not None:
        print(f"Using fixed oxide refractive index: {oxide_index}")
    else:
        print("Using default Tidy3D SiO2 (Horiba) for oxide material")
    
    # Check if the files exist
    if not os.path.exists(material_file):
        print(f"Error: Material file '{material_file}' does not exist.")
        print("Please provide a valid material file path.")
        sys.exit(1)
        
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' does not exist.")
        print("Please provide a valid CSV file path with wavelength and refractive index data.")
        sys.exit(1)
        
    if oxide_csv_file and not os.path.exists(oxide_csv_file):
        print(f"Error: Oxide CSV file '{oxide_csv_file}' does not exist.")
        print("Please provide a valid CSV file path with oxide refractive index data or omit to use default.")
        sys.exit(1)
    
    # Validate that the material name matches the CSV filename
    try:
        if not validate_material_name(material_file, csv_file):
            print("Warning: Proceeding with mismatched material name and CSV filename.")
    except ValueError as e:
        print(f"Error validating material name: {str(e)}")
        print("Please ensure the material_name field is present in the material JSON file.")
        sys.exit(1)
    
    # Load the material data to extract needed parameters
    material_data = load_material_data(material_file)
    
    # Extract material name from material file
    if "material_name" not in material_data:
        print(f"Error: material_name not defined in material file: {material_file}")
        print("Please ensure the material file includes a material_name field.")
        sys.exit(1)
    
    material_name = material_data["material_name"]
    
    # Extract process name from the material file
    if "process_name" not in material_data or not material_data["process_name"]:
        print(f"Error: process_name not defined in material file: {material_file}")
        print("Please ensure the material file includes a process_name field.")
        sys.exit(1)
    
    process_name = material_data["process_name"]
        
    # Set output directory using process name
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use process name in directory structure
        output_dir = os.path.join(os.getcwd(), process_name, material_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get simulation parameters from command line arguments
    l_pre = args.l_pre
    l_post = args.l_post
    run_time = args.run_time
    N = args.N
    grid_dl = args.grid_dl
    addl_str = args.addl_str
    max_num_poles = args.max_num_poles
    use_pml = not args.use_absorber
    absorber_layers = args.absorber_layers
    use_pec_bottom = args.use_pec_bottom
    
    # Get wavelengths from command line arguments
    if not args.wavelengths:
        print("Error: --wavelengths argument is required")
        print("Please specify one or more wavelengths in nm, e.g., --wavelengths 1550 1310")
        sys.exit(1)
    
    # Use wavelengths specified in command line
    lambdas = args.wavelengths
    print(f"Using wavelengths: {lambdas} nm")
    
    # Calculate frequencies from wavelengths
    freqs = [td.C_0 / (wl) for wl in lambdas]  # Hz
    print(f"Frequencies: {freqs} Hz")
    
    # Waveguide width
    wg_w = 0.4  # microns
    
    # Handle the waveguide configuration from material file
    from material_helper import is_dual_waveguide_stack, get_material_params
    
    # Determine if material file has dual waveguide configuration
    is_dual_waveguide_stack_result = is_dual_waveguide_stack(material_file)
    
    # Print waveguide configuration type from material file
    if is_dual_waveguide_stack_result:
        print(f"Material file contains dual waveguide configuration")
    else:
        print(f"Material file contains single waveguide configuration")
    
    # Extract appropriate parameters based on waveguide configuration
    if is_dual_waveguide_stack_result:
        # Get parameters for dual waveguide from material file
        toxt, boxt, wghb, wght, inth, tair, tSi = get_material_params(material_file)
        print(f"Using dual waveguide parameters from material file:")
        print(f"  Bottom waveguide height (wghb): {wghb} μm")
        print(f"  Top waveguide height (wght): {wght} μm")
        print(f"  Intermediate layer height (inth): {inth} μm")
    else:
        # Get parameters for single waveguide from material file
        toxt, boxt, wghb, tair, tSi = get_material_params(material_file)
        # Set dual waveguide parameters to None for single waveguide
        wght, inth = None, None
        print(f"Using single waveguide configuration with height (wghb): {wghb} μm")
    
    # Print simulation configuration
    print(f"Running simulation with:")
    print(f"  Process: {process_name}")
    print(f"  Material: {material_name}")
    print(f"  Wavelengths: {lambdas} nm")
    print(f"  Period: {args.period} μm")
    print(f"  Grating duty cycle (gratDC): {args.gratDC}")
    print(f"  Number of periods (N): {N}")
    print(f"  Grid spacing (grid_dl): {grid_dl}")
    print(f"  Max number of poles: {max_num_poles}")
    print(f"  Boundary conditions: {'PML' if use_pml else 'Absorber'} with {absorber_layers if not use_pml else 'NA'} layers")
    print(f"  Bottom surface: {'PEC' if use_pec_bottom else 'Silicon substrate'}")
    if addl_str:
        print(f"  Additional tag: {addl_str}")
    
    # Run the single period/DC simulation
    fname, sim_data = run_grating_simulation(
        material_name, output_dir, args.period, args.gratDC, 
        toxt, boxt, wghb, tair, tSi, wg_w, l_pre, l_post, run_time, 
        lambdas, freqs, material_file, csv_file, oxide_csv_file, oxide_index,
        include_2d_monitor=args.include_2d_monitor,
        N=N, 
        grid_dl=grid_dl,
        addl_str=addl_str,
        run_sim=not args.no_run,  # Run the simulation if --no-run is not specified
        max_num_poles=max_num_poles,
        use_pml=use_pml,
        absorber_layers=absorber_layers,
        wght=wght,
        inth=inth,
        use_pec_bottom=use_pec_bottom
    )
    
    if not args.no_run:
        print(f"Simulation results stored in {fname}_results.hdf5")
        print(f"To post-process the results, run: python post_process_2D.py --file {fname}_results.hdf5")
    else:
        print(f"Simulation set up but not run. No results file generated.")

# Run the simulation
if __name__ == "__main__":
    main()


