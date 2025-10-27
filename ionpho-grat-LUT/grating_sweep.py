import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import concurrent.futures
import argparse
import time
import pandas as pd
import json
import datetime
import shutil
from tqdm import tqdm
from pathlib import Path

# Add path to the tidy3d module if needed
sys.path.append(os.path.dirname(__file__))
from material_helper import get_material_params, load_material_data
from single_grating_simulation import run_grating_simulation

# Add global variables to store sweep parameters
_sweep_params = {
    'material': None,
    'output_dir': None,
    'toxt': None,
    'boxt': None,
    'wg_h': None,
    'tair': None,
    'tSi': None,
    'wg_w': None,
    'l_pre': None,
    'l_post': None,
    'run_time': None,
    'lambdas': None,
    'freqs': None,
    'material_file': None,
    'include_2d_monitor': False,
    'sweep_log_path': None
}

def run_single_simulation(args):
    """Run a single grating simulation with specified parameters
    
    Args:
        args: Tuple containing (params, simulation_params) where:
            params: Tuple of (period, gratDC, show_plots)
            simulation_params: Dictionary with all simulation parameters
    
    Returns:
        Simulation results dictionary
    """
    # Unpack arguments
    (period, gratDC, show_plots), sim_params = args
    
    pertDC = 1 - gratDC
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Log simulation start
        with open(sim_params['sweep_log_path'], 'a') as f:
            f.write(f"{timestamp},{period:.4f},{gratDC:.2f},{pertDC:.2f},started,in_progress\n")
        
        # Run the simulation with the simulations subfolder
        sim_output_dir = sim_params['simulations_dir']
        
        # Get simulation parameters from sim_params
        N = sim_params.get('N', 60)  # Number of grating periods
        grid_dl = sim_params.get('grid_dl', 0.01)  # Grid spacing
        addl_str = sim_params.get('addl_str', '')  # Additional string tag
        run_sim = sim_params.get('run_sim', True)  # Whether to run the simulation
        max_num_poles = sim_params.get('max_num_poles', 1)  # Max poles for dispersion fitting
        use_pml = sim_params.get('use_pml', True)  # Use PML boundary conditions
        absorber_layers = sim_params.get('absorber_layers', 60)  # Number of absorber layers
        
        # Extract dispersion data paths
        csv_file = sim_params.get('csv_file')  # Path to CSV file with refractive index data
        oxide_csv_file = sim_params.get('oxide_csv_file')  # Path to oxide CSV file
        oxide_index = sim_params.get('oxide_index')  # Fixed oxide refractive index
        
        # Extract dual waveguide parameters (if applicable)
        wght = sim_params.get('wght')  # Top waveguide height
        inth = sim_params.get('inth')  # Intermediate layer height
        
        # Extract boundary condition parameters
        use_pec_bottom = sim_params.get('use_pec_bottom', False)  # PEC bottom boundary condition
        
        # Run the simulation
        fname, sim_data = run_grating_simulation(
            sim_params['material'], 
            sim_output_dir, 
            period, 
            gratDC, 
            sim_params['toxt'], 
            sim_params['boxt'], 
            sim_params['wg_h'], 
            sim_params['tair'], 
            sim_params['tSi'], 
            sim_params['wg_w'], 
            sim_params['l_pre'], 
            sim_params['l_post'], 
            sim_params['run_time'], 
            sim_params['lambdas'], 
            sim_params['freqs'], 
            sim_params['material_file'],
            csv_file,
            oxide_csv_file=oxide_csv_file,
            oxide_index=oxide_index,
            include_2d_monitor=sim_params['include_2d_monitor'], 
            show_plots=show_plots,
            N=N,
            grid_dl=grid_dl,
            addl_str=addl_str,
            run_sim=run_sim,
            max_num_poles=max_num_poles,
            use_pml=use_pml,
            absorber_layers=absorber_layers,
            wght=wght,
            inth=inth,
            use_pec_bottom=use_pec_bottom
        )
        
        # Log successful simulation
        status = "setup_completed" if not run_sim else "completed"
        with open(sim_params['sweep_log_path'], 'a') as f:
            f.write(f"{timestamp},{period:.4f},{gratDC:.2f},{pertDC:.2f},{fname},{status}\n")
        
        # Return simulation results
        return {
            'period': period,
            'gratDC': gratDC,
            'pertDC': pertDC,
            'filename': fname,
            'status': status
        }
        
    except Exception as e:
        # Log failed simulation
        with open(sim_params['sweep_log_path'], 'a') as f:
            f.write(f"{timestamp},{period:.4f},{gratDC:.2f},{pertDC:.2f},failed,{str(e)}\n")
        
        # Return error information
        return {
            'period': period, 
            'gratDC': gratDC,
            'pertDC': pertDC,
            'filename': None,
            'status': f'failed: {str(e)}'
        }

def run_sweep(material, output_dir, periods, gratDCs, toxt, boxt, wghb, tair, tSi, 
              wg_w, l_pre, l_post, run_time, lambdas, freqs, material_file, csv_file=None, 
              oxide_csv_file=None, oxide_index=None, include_2d_monitor=False, 
              simultaneous=True, max_workers=4, plot_first=True,
              process_name="AIM_SL1", date_str=None, N=60, grid_dl=0.01, addl_str='', run_sim=True,
              max_num_poles=1, use_pml=True, absorber_layers=60, wght=None, inth=None, use_pec_bottom=False):
    """
    Run a parameter sweep over grating periods and duty cycles
    
    Args:
        material: Material name for the waveguide
        output_dir: Directory to save output files
        periods: List of grating periods in microns to sweep
        gratDCs: List of grating duty cycles (0.0-1.0) to sweep
                 Note: gratDC is the duty cycle of the grating teeth
                 pertDC = 1-gratDC is the duty cycle of the perturbation (gap)
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
        material_file: Path to material file
        csv_file: Path to CSV file with wavelength and refractive index data
        oxide_csv_file: Path to CSV file with oxide refractive index data
        oxide_index: Fixed refractive index value for oxide
        include_2d_monitor: Whether to include a 2D XZ field monitor
        simultaneous: Whether to run simulations simultaneously
        max_workers: Maximum number of concurrent simulations
        plot_first: Whether to plot the geometry for the first simulation
        process_name: Name of the process (default: "AIM_SL1")
        date_str: Date string (default: current date in YYYYMMDD format)
        N: Number of grating periods (default: 60)
        grid_dl: Grid spacing for simulation (default: 0.01)
        addl_str: Additional string tag for naming files (default: '')
        run_sim: Whether to actually run the simulations (default: True)
               If False, sets up the simulations but does not run them
        max_num_poles: Maximum number of poles to use in dispersion fitting (default: 1)
        use_pml: Whether to use PML boundary conditions (default: True)
        absorber_layers: Number of absorber layers if not using PML (default: 60)
        wght: Top waveguide height for dual waveguide (default: None)
        inth: Intermediate layer height for dual waveguide (default: None)
    
    Returns:
        results_df: DataFrame with sweep results
        sweep_dir: Path to sweep directory
    """
    # Get the current date if not provided
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
    # Create the process directory structure
    process_dir = os.path.join(os.path.dirname(output_dir), process_name)
    os.makedirs(process_dir, exist_ok=True)
    
    # Create the sweep directory with date and additional string if provided
    sweep_name = f"{material}_sweep_{date_str}"
    if addl_str:
        sweep_name += f"_{addl_str}"
    sweep_dir = os.path.join(process_dir, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Create simulations subdirectory
    simulations_dir = os.path.join(sweep_dir, "simulations")
    os.makedirs(simulations_dir, exist_ok=True)
    
    # Copy the material file to the sweep directory
    if material_file and os.path.exists(material_file):
        material_filename = os.path.basename(material_file)
        material_copy_path = os.path.join(sweep_dir, material_filename)
        shutil.copy2(material_file, material_copy_path)
        print(f"Copied material file to {material_copy_path}")
        
        # Also copy the CSV file if referenced in the material file
        try:
            material_data = load_material_data(material_file)
            if "csv_file" in material_data and not csv_file:
                csv_path = material_data["csv_file"]
                if not os.path.isabs(csv_path):
                    # Relative path - resolve relative to material_file
                    material_dir = os.path.dirname(os.path.abspath(material_file))
                    csv_abs_path = os.path.join(material_dir, csv_path)
                else:
                    csv_abs_path = csv_path
                
                if os.path.exists(csv_abs_path):
                    csv_filename = os.path.basename(csv_abs_path)
                    csv_copy_path = os.path.join(sweep_dir, csv_filename)
                    shutil.copy2(csv_abs_path, csv_copy_path)
                    print(f"Copied CSV file to {csv_copy_path}")
        except Exception as e:
            print(f"Warning: Could not copy CSV file referenced in material file: {e}")
    
    # Copy the provided CSV file if specified
    if csv_file and os.path.exists(csv_file):
        csv_filename = os.path.basename(csv_file)
        csv_copy_path = os.path.join(sweep_dir, csv_filename)
        shutil.copy2(csv_file, csv_copy_path)
        print(f"Copied CSV file to {csv_copy_path}")
            
    # Copy the oxide CSV file if specified
    if oxide_csv_file and os.path.exists(oxide_csv_file):
        oxide_csv_filename = os.path.basename(oxide_csv_file)
        oxide_csv_copy_path = os.path.join(sweep_dir, oxide_csv_filename)
        shutil.copy2(oxide_csv_file, oxide_csv_copy_path)
        print(f"Copied oxide CSV file to {oxide_csv_copy_path}")
    
    # Check if we have a dual waveguide configuration
    is_dual_waveguide = wght is not None and inth is not None and wght > 0 and inth > 0
    if is_dual_waveguide:
        print(f"Using dual waveguide configuration:")
        print(f"  Bottom waveguide height (wghb): {wghb} μm")
        print(f"  Top waveguide height (wght): {wght} μm")
        print(f"  Intermediate layer height (inth): {inth} μm")
    else:
        print(f"Using single waveguide configuration:")
        print(f"  Waveguide height (wghb): {wghb} μm")
    
    # Create and save the sweep configuration
    sweep_config = {
        "sweep": {
            "date": date_str,
            "material": material,
            "process_name": process_name,
            "periods": periods.tolist(),
            "gratDCs": gratDCs.tolist(),
            "wavelengths": lambdas.tolist()
        },
        "simulation": {
            "material_file": material_file,
            "csv_file": csv_file,
            "oxide_csv_file": oxide_csv_file,
            "oxide_index": oxide_index,
            "toxt": toxt,
            "boxt": boxt,
            "wghb": wghb,
            "wght": wght if is_dual_waveguide else None,
            "inth": inth if is_dual_waveguide else None,
            "tair": tair,
            "tSi": tSi,
            "wg_w": wg_w,
            "l_pre": l_pre,
            "l_post": l_post,
            "run_time": run_time,
            "N": N,
            "grid_dl": grid_dl,
            "include_2d_monitor": include_2d_monitor,
            "max_num_poles": max_num_poles,
            "use_pml": use_pml,
            "absorber_layers": absorber_layers,
            "simultaneous": simultaneous,
            "max_workers": max_workers
        },
        "results": {
            "sweep_dir": sweep_dir
        }
    }
    
    # Save sweep config
    sweep_config_path = os.path.join(sweep_dir, "sweep_config.json")
    with open(sweep_config_path, 'w') as f:
        json.dump(sweep_config, f, indent=2)
    print(f"Saved sweep configuration to {sweep_config_path}")
    
    # Create a sweep log file to track progress
    sweep_log_path = os.path.join(sweep_dir, "sweep_log.csv")
    with open(sweep_log_path, 'w') as f:
        f.write("Timestamp,Period,GratDC,PertDC,Filename,Status\n")
    
    # Generate the parameter combinations
    param_combinations = []
    show_plots_first = plot_first  # Whether to show plots for the first simulation only
    
    for i, period in enumerate(periods):
        for j, gratDC in enumerate(gratDCs):
            # Only show plots for the first simulation if plot_first is True
            show_plots = show_plots_first and i == 0 and j == 0
            param_combinations.append(((period, gratDC, show_plots), None))
    
    # Store global parameters for run_single_simulation
    global _sweep_params
    _sweep_params = {
        'material': material,
        'output_dir': output_dir,
        'toxt': toxt,
        'boxt': boxt,
        'wg_h': wghb,
        'wght': wght,
        'inth': inth,
        'tair': tair,
        'tSi': tSi,
        'wg_w': wg_w,
        'l_pre': l_pre,
        'l_post': l_post,
        'run_time': run_time,
        'lambdas': lambdas,
        'freqs': freqs,
        'material_file': material_file,
        'csv_file': csv_file,
        'oxide_csv_file': oxide_csv_file,
        'oxide_index': oxide_index,
        'include_2d_monitor': include_2d_monitor,
        'sweep_log_path': sweep_log_path,
        'simulations_dir': simulations_dir,
        'N': N,
        'grid_dl': grid_dl,
        'addl_str': addl_str,
        'run_sim': run_sim,
        'max_num_poles': max_num_poles,
        'use_pml': use_pml,
        'absorber_layers': absorber_layers
    }
    
    # Prepare simulation parameters for each job
    param_combinations = [((period, gratDC, show_plots), _sweep_params.copy()) 
                         for (period, gratDC, show_plots), _ in param_combinations]
    
    # Run the simulations
    results = []
    
    # Show total number of simulations
    total_simulations = len(param_combinations)
    print(f"Running {total_simulations} simulations ({len(periods)} periods × {len(gratDCs)} duty cycles)")
    
    if simultaneous and total_simulations > 1:
        # Use concurrent.futures to run simulations in parallel
        print(f"Running simulations in parallel with {max_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Show progress with tqdm
            for result in tqdm(executor.map(run_single_simulation, param_combinations), 
                               total=total_simulations, desc="Simulations"):
                results.append(result)
    else:
        # Run simulations sequentially
        print("Running simulations sequentially")
        for args in tqdm(param_combinations, desc="Simulations"):
            result = run_single_simulation(args)
            results.append(result)
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    results_csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved sweep results to {results_csv_path}")
    
    # Generate a summary of the results
    total_completed = (results_df['status'] == 'completed').sum()
    total_setup = (results_df['status'] == 'setup_completed').sum()
    total_failed = results_df['status'].str.startswith('failed').sum() if 'status' in results_df else 0
    
    print(f"\nSweep Summary:")
    print(f"  Total simulations: {total_simulations}")
    print(f"  Completed: {total_completed}")
    print(f"  Setup only: {total_setup}")
    print(f"  Failed: {total_failed}")
    
    print(f"\nTo post-process the results, run:")
    print(f"python post_process_sweep.py --config {sweep_config_path}")
    
    return results_df, sweep_dir

def load_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate config structure
        if "simulation" not in config:
            print(f"Error: Missing 'simulation' section in config file: {config_file}")
            # Import here to avoid circular imports
            from material_helper import print_format_info
            print_format_info()
            raise KeyError(f"Missing 'simulation' section in config file: {config_file}")
        
        # Extract simulation config
        sim_config = config.get('simulation', {})
        
        # Check for required parameters
        required_params = ["material", "material_file", "process_name", "period", "gratDC", "simulation_params"]
        missing_params = [param for param in required_params if param not in sim_config]
        
        if missing_params:
            missing_str = ", ".join(missing_params)
            print(f"Error: Required parameters missing from simulation config: {missing_str}")
            from material_helper import print_format_info
            print_format_info()
            raise KeyError(f"Required parameters missing from simulation config: {missing_str}")
        
        # Check that process_name is not empty
        if not sim_config["process_name"]:
            print("Error: process_name in config file cannot be empty")
            from material_helper import print_format_info
            print_format_info()
            raise ValueError("process_name in config file cannot be empty")
            
        # Check if material_file is provided
        if sim_config["material_file"] is None:
            print(f"Error: 'material_file' must be specified in config file: {config_file}")
            from material_helper import print_format_info
            print_format_info()
            raise KeyError(f"'material_file' must be specified in config file: {config_file}")
            
        # Check simulation_params section
        sim_params = sim_config.get('simulation_params', {})
        required_sim_params = ["l_pre", "l_post", "run_time", "N", "grid_dl", "wavelengths"]
        missing_sim_params = [param for param in required_sim_params if param not in sim_params]
        
        if missing_sim_params:
            missing_str = ", ".join(missing_sim_params)
            print(f"Error: Required parameters missing from simulation_params: {missing_str}")
            from material_helper import print_format_info
            print_format_info()
            raise KeyError(f"Required parameters missing from simulation_params: {missing_str}")
        
        # Check if wavelengths are provided
        wavelengths = sim_params.get("wavelengths")
        if not wavelengths or not isinstance(wavelengths, list) or len(wavelengths) == 0:
            print(f"Error: 'wavelengths' must be specified in simulation_params and must be a non-empty list")
            print("Example: 'wavelengths': [1550, 1310]")
            from material_helper import print_format_info
            print_format_info()
            raise ValueError(f"'wavelengths' must be a non-empty list in simulation_params")
        
        # Check material_config section
        material_config = sim_config.get('material_config', {})
        
        # Added support for material parameters in config
        sim_config['oxide_index'] = material_config.get('oxide_index')
        sim_config['oxide_csv_file'] = material_config.get('oxide_csv_file')
        sim_config['max_num_poles'] = material_config.get('max_num_poles', 1)
        
        # Check if oxide_source is specified
        if 'oxide_source' in material_config:
            # Validate oxide_source value
            valid_oxide_sources = ['default', 'fixed_index', 'csv_file']
            if material_config['oxide_source'] not in valid_oxide_sources:
                print(f"Error: Invalid oxide_source value '{material_config['oxide_source']}' in config")
                print(f"Valid values are: {', '.join(valid_oxide_sources)}")
                raise ValueError(f"Invalid oxide_source: {material_config['oxide_source']}")
            sim_config['oxide_source'] = material_config['oxide_source']
        
        # Added support for simulation boundary conditions
        boundary_config = sim_config.get('boundary_conditions', {})
        sim_config['use_pml'] = boundary_config.get('use_pml', True)
        sim_config['absorber_layers'] = boundary_config.get('absorber_layers', 60)
        sim_config['use_pec_bottom'] = boundary_config.get('use_pec_bottom', False)
        
        return sim_config
        
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file}")
        from material_helper import print_format_info
        print_format_info()
        raise
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {config_file}")
        print(f"JSON error: {str(e)}")
        from material_helper import print_format_info
        print_format_info()
        raise

def main():
    """Main function for grating parameter sweep"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run grating coupler parameter sweep')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON configuration file')
    parser.add_argument('--no-run', action='store_true',
                        help='Set up the simulations but do not run them (for debugging)')
    # Add arguments to override material files
    parser.add_argument('--material-file', type=str, 
                        help='Override material JSON file path from config')
    parser.add_argument('--csv-file', type=str, 
                        help='Override CSV file path with wavelength and refractive index data')
    parser.add_argument('--oxide-csv-file', type=str, default=None,
                        help='Path to CSV file with oxide refractive index data (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    sim_config = load_config(args.config)
    
    # Extract parameters from config
    material_name = sim_config["material"]  # Required by load_config
    material_file_path = sim_config["material_file"]  # Required by load_config
    output_dir = sim_config.get('output_dir')
    include_2d_monitor = sim_config.get('include_2d_monitor', False)  # Optional with default
    
    # Get process name from config file (required)
    process_name = sim_config["process_name"]
    
    # Get period parameters
    period_config = sim_config["period"]  # Required by load_config
    period_min = period_config["min"]
    period_max = period_config["max"]
    period_steps = period_config["steps"]
    
    # Get gratDC parameters
    gratDC_config = sim_config["gratDC"]  # Required by load_config
    gratDC_min = gratDC_config["min"]
    gratDC_max = gratDC_config["max"]
    gratDC_steps = gratDC_config["steps"]
    
    # Get execution parameters (optional with defaults)
    exec_config = sim_config.get('execution', {})
    sequential = exec_config.get('sequential', False)
    max_workers = exec_config.get('max_workers', 4)
    plot_first = exec_config.get('plot_first', True)
    run_sim = exec_config.get('run_sim', True) and not args.no_run  # Honor both config and command line
    
    # Get simulation parameters from config (all required by load_config)
    sim_params_config = sim_config["simulation_params"]
    l_pre = sim_params_config["l_pre"]
    l_post = sim_params_config["l_post"]
    run_time = sim_params_config["run_time"]
    N = sim_params_config["N"]
    grid_dl = sim_params_config["grid_dl"]
    addl_str = sim_params_config.get('addl_str', '')  # Optional with default
    
    # Get material configuration parameters (from config only)
    max_num_poles = sim_config.get('max_num_poles', 1)
    oxide_index = sim_config.get('oxide_index')
    oxide_csv_file = args.oxide_csv_file if args.oxide_csv_file is not None else sim_config.get('oxide_csv_file')
    
    # Get boundary condition parameters (from config only)
    use_pml = sim_config.get('use_pml', True)
    absorber_layers = sim_config.get('absorber_layers', 60)
    use_pec_bottom = sim_config.get('use_pec_bottom', False)
    
    # Get oxide source configuration
    oxide_source = sim_config.get('oxide_source', 'default')
    
    # Handle oxide source based on configuration
    if oxide_source == 'fixed_index' and oxide_index is None:
        print("Warning: oxide_source is set to 'fixed_index' but no oxide_index provided")
        print("Falling back to default Tidy3D SiO2 (Horiba)")
        oxide_source = 'default'
        
    if oxide_source == 'csv_file' and oxide_csv_file is None:
        print("Warning: oxide_source is set to 'csv_file' but no oxide_csv_file provided")
        print("Falling back to default Tidy3D SiO2 (Horiba)")
        oxide_source = 'default'
    
    # Override with command-line arguments if provided
    if args.oxide_csv_file is not None:
        oxide_source = 'csv_file'
        print(f"Using oxide CSV file from command line: {args.oxide_csv_file}")
    
    # Display oxide source configuration
    print(f"Oxide material source: {oxide_source}")
    if oxide_source == 'default':
        print("Using default Tidy3D SiO2 (Horiba) for oxide material")
    elif oxide_source == 'fixed_index':
        print(f"Using fixed oxide refractive index: {oxide_index}")
    elif oxide_source == 'csv_file':
        print(f"Using oxide dispersion data from CSV file: {oxide_csv_file}")
    
    # Get wavelengths from config
    lambdas = np.array(sim_params_config["wavelengths"])
    print(f"Using wavelengths from config: {lambdas} nm")
    
    # Calculate frequencies from wavelengths
    freqs = np.array([td.C_0 / (wl) for wl in lambdas])  # THz
    
    # Override material file if provided via command line
    if args.material_file:
        print(f"Overriding material file from config with command line argument: {args.material_file}")
        material_file_path = args.material_file
    
    # Check if the material file path is a relative path and make it absolute
    if not os.path.isabs(material_file_path):
        # Try relative to the config file directory first
        config_dir = os.path.dirname(os.path.abspath(args.config))
        if os.path.exists(os.path.join(config_dir, material_file_path)):
            material_file = os.path.join(config_dir, material_file_path)
        # If not found, try relative to the current working directory
        elif os.path.exists(os.path.join(os.getcwd(), material_file_path)):
            material_file = os.path.join(os.getcwd(), material_file_path)
        else:
            material_file = material_file_path
    else:
        material_file = material_file_path
    
    # Check if the material file exists
    if not os.path.exists(material_file):
        print(f"Error: Material file '{material_file}' does not exist.")
        print("Please provide a valid material file path in the configuration or via command line.")
        sys.exit(1)
    
    # Get material stack and refractive index from material file
    print(f"Using material file: {material_file}")
    
    # Load the material data to extract the process name
    material_data = load_material_data(material_file)
    material_process_name = material_data.get("process_name")
    
    # Use command line CSV file if provided, otherwise use from material file
    if args.csv_file:
        print(f"Using CSV file from command line: {args.csv_file}")
        csv_path = args.csv_file
    elif "csv_file" in material_data:
        csv_path = material_data["csv_file"]
        print(f"Using CSV file from material file: {csv_path}")
    else:
        print(f"Error: 'csv_file' field is missing from material file: {material_file}")
        print("This field is required to specify the path to the CSV file with wavelength and index data.")
        print("Please provide a CSV file either in the material JSON or via the --csv-file argument.")
        print("Example material JSON format:")
        print('  {')
        print('    "process_name": "AIM_SL1",')
        print('    "material_name": "AO_AIM",')
        print('    "csv_file": "AO_AIM.csv",')
        print('    "layer_stack": { ... }')
        print('  }')
        print("\nThe CSV file should have columns for wavelength (nm), n, and optionally k with a header row.")
        from material_helper import print_format_info
        print_format_info()
        sys.exit(1)
    
    # If CSV path is relative, resolve it
    if not os.path.isabs(csv_path):
        # If provided from command line, resolve relative to current directory
        if args.csv_file:
            csv_abs_path = os.path.join(os.getcwd(), csv_path)
        else:
            # Otherwise resolve relative to material file directory
            material_dir = os.path.dirname(os.path.abspath(material_file))
            csv_abs_path = os.path.join(material_dir, csv_path)
    else:
        csv_abs_path = csv_path
    
    # Check that the CSV file exists
    if not os.path.exists(csv_abs_path):
        print(f"Error: CSV file '{csv_abs_path}' does not exist.")
        print("Please provide a valid CSV file path in the material file or via command line.")
        sys.exit(1)
    
    # Handle oxide CSV file or oxide index if provided
    oxide_csv_file = args.oxide_csv_file
    
    if oxide_csv_file:
        # Resolve relative path if needed
        if not os.path.isabs(oxide_csv_file):
            oxide_csv_abs_path = os.path.join(os.getcwd(), oxide_csv_file)
        else:
            oxide_csv_abs_path = oxide_csv_file
            
        # Check that the oxide CSV file exists
        if not os.path.exists(oxide_csv_abs_path):
            print(f"Error: Oxide CSV file '{oxide_csv_abs_path}' does not exist.")
            print("Please provide a valid oxide CSV file path.")
            sys.exit(1)
        
        oxide_csv_file = oxide_csv_abs_path
        print(f"Using oxide dispersion data from: {oxide_csv_file}")
    elif oxide_index is not None:
        print(f"Using fixed oxide refractive index: {oxide_index}")
    else:
        print("Using default Tidy3D SiO2 (Horiba) material for oxide")
    
    # Print boundary condition information
    if use_pml:
        print(f"Using PML boundary conditions")
    else:
        print(f"Using absorber boundary conditions with {absorber_layers} layers")
    
    if use_pec_bottom:
        print(f"Using PEC boundary condition at bottom surface (no silicon substrate)")
    else:
        print(f"Using silicon substrate at bottom surface")
    
    # Ensure the material process name is defined
    if not material_process_name:
        print(f"Error: process_name not defined in material file: {material_file}")
        print("Please ensure the material file includes a process_name field.")
        sys.exit(1)
    
    # Extract material name from the material file name (if not explicitly overridden)
    if args.material_file:
        # If material file was overridden, update material_name from the file
        material_name = material_data.get("material_name", 
            os.path.splitext(os.path.basename(material_file))[0])
    
    # Check that process names match
    if material_process_name != process_name:
        print(f"Warning: Process name mismatch between config ({process_name}) and material file ({material_process_name}).")
        print("Proceeding with the process name from the material file.")
        process_name = material_process_name
    
    # Set output directory
    if output_dir is None:
        # Use process name in the directory structure
        output_dir = os.path.join(os.getcwd(), process_name, f"{material_name}_sweep")
    
    # Get current date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Get material parameters from the JSON file
    if os.path.exists(material_file):
        try:
            from material_helper import is_dual_waveguide_stack
            if is_dual_waveguide_stack(material_file):
                # If it's a dual waveguide stack, get all parameters
                toxt, boxt, wghb, wght, inth, tair, tSi = get_material_params(material_file)
                print(f"Using dual waveguide parameters from material file:")
                print(f"  Bottom waveguide height (wghb): {wghb} μm")
                print(f"  Top waveguide height (wght): {wght} μm")
                print(f"  Intermediate layer height (inth): {inth} μm")
            else:
                # Otherwise get single waveguide parameters
                toxt, boxt, wghb, tair, tSi = get_material_params(material_file)
                wght, inth = None, None  # Ensure these are None for single waveguide
                print(f"Using single waveguide configuration with height (wghb): {wghb} μm")
        except Exception as e:
            print(f"Error extracting parameters from material file: {str(e)}")
            sys.exit(1)
    else:
        print(f"Error: Could not find material file at {material_file}")
        sys.exit(1)
    
    # Default waveguide width (this is still hardcoded)
    wg_w = 0.4  # microns
    
    # Generate sweep parameters
    periods = np.linspace(period_min, period_max, period_steps)
    gratDCs = np.linspace(gratDC_min, gratDC_max, gratDC_steps)
    
    print(f"Running parameter sweep with:")
    print(f"  Process: {process_name}")
    print(f"  Date: {date_str}")
    print(f"  Material: {material_name}")
    print(f"  Wavelengths: {lambdas} nm")
    print(f"  Periods: {periods}")
    print(f"  Grating Duty Cycles (gratDC): {gratDCs}")
    print(f"  Perturbation Duty Cycles (pertDC = 1-gratDC): {1-gratDCs}")
    print(f"  Number of periods (N): {N}")
    print(f"  Grid spacing (grid_dl): {grid_dl}")
    print(f"  Max fitting poles: {max_num_poles}")
    if addl_str:
        print(f"  Additional tag: {addl_str}")
    
    # Update run_single_simulation to support additional parameters
    # First, we need to modify the function signature in the implementation
    
    # Run sweep with the updated parameters
    results_df, sweep_dir = run_sweep(
        material_name, output_dir, periods, gratDCs,
        toxt, boxt, wghb, tair, tSi, wg_w, l_pre, l_post, run_time,
        lambdas, freqs, material_file, csv_abs_path, oxide_csv_file, oxide_index, 
        include_2d_monitor=include_2d_monitor,
        simultaneous=not sequential, max_workers=max_workers,
        plot_first=plot_first, process_name=process_name, date_str=date_str,
        N=N, grid_dl=grid_dl, addl_str=addl_str, run_sim=run_sim,
        max_num_poles=max_num_poles, use_pml=use_pml, absorber_layers=absorber_layers,
        wght=wght, inth=inth, use_pec_bottom=use_pec_bottom
    )
    
    # Copy the input config file to the sweep directory
    config_dest = os.path.join(sweep_dir, os.path.basename(args.config))
    if args.config != config_dest:
        shutil.copy2(args.config, config_dest)
        print(f"Copied original config file to {config_dest}")
    
    print(f"Simulation sweep complete. To post-process the results, run:")
    print(f"python post_process_sweep.py --config {os.path.join(sweep_dir, 'sweep_config.json')}")

if __name__ == "__main__":
    main() 