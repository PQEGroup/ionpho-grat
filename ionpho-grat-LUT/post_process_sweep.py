import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd
import json
from tqdm import tqdm

# Add path to the tidy3d module if needed
sys.path.append(os.path.dirname(__file__))
from material_helper import get_material_params, load_material_data

def post_process_sweep(sweep_dir, show_plots=False):
    """
    Post-process all simulation results from a parameter sweep
    
    Args:
        sweep_dir: Directory containing sweep results and the simulations subfolder
        show_plots: Whether to display plots for each simulation (default: False)
    
    Returns:
        metrics_df: DataFrame with extracted metrics from all simulations
    """
    # Import here to avoid circular import
    from post_process_2D import post_process_simulation
    
    # Get sweep results
    results_path = os.path.join(sweep_dir, "sweep_results.csv")
    if not os.path.exists(results_path):
        print(f"Sweep results file not found: {results_path}")
        return None
    
    # Get simulations directory
    simulations_dir = os.path.join(sweep_dir, "simulations")
    if not os.path.exists(simulations_dir):
        print(f"Simulations directory not found: {simulations_dir}")
        print(f"Assuming simulation files are directly in the sweep directory")
        simulations_dir = sweep_dir
    
    results_df = pd.read_csv(results_path)
    completed_results = results_df[results_df.status == 'completed']
    
    if len(completed_results) == 0:
        print("No completed simulations found")
        return None
    
    print(f"Post-processing {len(completed_results)} simulations")
    
    # Extract metrics from each simulation
    metrics = []
    for _, row in tqdm(completed_results.iterrows(), total=len(completed_results)):
        filename = row['filename']
        
        # Check if the result file exists in the simulations directory
        result_file = f"{filename}_results.hdf5"
        if not os.path.exists(result_file) and os.path.exists(os.path.join(simulations_dir, os.path.basename(result_file))):
            # Try in simulations directory if not found directly
            result_file = os.path.join(simulations_dir, os.path.basename(result_file))
        
        if not os.path.exists(result_file):
            print(f"Result file not found: {result_file}")
            continue
        
        try:
            # Process the simulation
            peaks_list, heights_list, power_fractions, avg_angles, wavelengths, fit_results, n_eff_values = (
                post_process_simulation(result_file, sort_by_angle=True, show_plots=show_plots)
            )
            
            # Extract metrics for each wavelength
            for wl_idx, wl in enumerate(wavelengths):
                # Get upward emission fraction
                up_vs_input = power_fractions['up_vs_input'][wl_idx]
                
                # Get maximum peak angle and height
                peaks = peaks_list[wl_idx]
                heights = heights_list[wl_idx]
                max_peak_angle = peaks[0] if len(peaks) > 0 else None
                max_peak_height = heights[0] if len(heights) > 0 else None
                
                # Get average angles
                avg_angle_P = avg_angles['poynting_vector'][wl_idx]
                avg_angle_H = avg_angles['h_field'][wl_idx]
                
                # Get decay rates if available
                alpha_air = fit_results['air'].get(wl_idx, {}).get('alpha', None)
                alpha_wg = fit_results['waveguide'].get(wl_idx, {}).get('alpha', None)
                
                # Get effective index
                n_eff = n_eff_values[wl_idx]
                
                # Store metrics
                metrics.append({
                    'period': row['period'],
                    'gratDC': row['gratDC'],
                    'pertDC': row['pertDC'],
                    'wavelength': wl,
                    'up_vs_input': up_vs_input,
                    'max_peak_angle': max_peak_angle,
                    'max_peak_height': max_peak_height,
                    'avg_angle_P': avg_angle_P,
                    'avg_angle_H': avg_angle_H,
                    'alpha_air': alpha_air,
                    'alpha_wg': alpha_wg,
                    'n_eff': n_eff,
                    'filename': filename
                })
        except Exception as e:
            print(f"Error processing {result_file}: {str(e)}")
    
    # Convert metrics to DataFrame and save
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(sweep_dir, "sweep_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"Post-processing completed. Metrics saved to {metrics_path}")
    return metrics_df

def plot_sweep_results(sweep_dir, metric='up_vs_input', wavelength=None, metrics_to_plot=None):
    """
    Plot the results of a parameter sweep as a grid of 2D heatmap subplots
    
    Args:
        sweep_dir: Directory containing sweep results
        metric: Primary metric to plot (for backwards compatibility)
        wavelength: Specific wavelength to plot (default: first wavelength)
        metrics_to_plot: List of metrics to plot (default: None, which plots all key metrics)
    
    Returns:
        fig: The matplotlib figure object
    """
    # Get metrics
    metrics_path = os.path.join(sweep_dir, "sweep_metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return None
    
    metrics_df = pd.read_csv(metrics_path)
    
    # Filter by wavelength if specified
    if wavelength is not None:
        metrics_df = metrics_df[metrics_df.wavelength.round(1) == round(wavelength, 1)]
    else:
        # Get first wavelength
        wavelength = metrics_df.wavelength.unique()[0]
        metrics_df = metrics_df[metrics_df.wavelength.round(1) == round(wavelength, 1)]
    
    if len(metrics_df) == 0:
        print(f"No data found for wavelength {wavelength}")
        return None
    
    # Check if we have enough data for a 2D plot
    if len(metrics_df.period.unique()) <= 1 or len(metrics_df.gratDC.unique()) <= 1:
        print("Not enough data for a 2D plot")
        return None
    
    # Define list of metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = [
            'up_vs_input',            # Upward emission fraction
            'max_peak_angle',         # Main peak angle
            'alpha_air',              # Decay rate in air
            'alpha_wg',               # Decay rate in waveguide
            'avg_angle_P',            # Average angle (Poynting)
            'n_eff'                   # Effective index
        ]
    
    # If only one metric requested, convert to list
    if isinstance(metrics_to_plot, str):
        metrics_to_plot = [metrics_to_plot]
    
    # Ensure the requested metric is in the list if not already
    if metric not in metrics_to_plot:
        metrics_to_plot.insert(0, metric)
    
    # Filter to metrics that exist in the data
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns 
                        and m not in ['filename', 'status', 'period', 'gratDC', 'pertDC', 'wavelength']]
    
    if not available_metrics:
        print(f"None of the requested metrics {metrics_to_plot} found in data")
        return None
    
    # Determine subplot grid layout
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)  # Max 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division for number of rows
    
    # Create figure with subplots
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    
    # Get sweep info from directory structure
    process_name = os.path.basename(os.path.dirname(sweep_dir))
    sweep_name = os.path.basename(sweep_dir)
    
    plt.suptitle(f'{process_name} - {sweep_name}\nλ={wavelength:.1f} nm', fontsize=16)
    
    # Create subplot for each metric
    for i, metric_name in enumerate(available_metrics):
        try:
            # Create a pivot table for the heatmap with period on x-axis and gratDC on y-axis
            pivot_table = metrics_df.pivot_table(
                values=metric_name, 
                index='gratDC',     # Now gratDC is on the y-axis 
                columns='period'    # Now period is on the x-axis
            )
            
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # Plot heatmap
            im = ax.imshow(
                pivot_table.values,
                origin='lower',
                aspect='auto',
                extent=[
                    metrics_df.period.min(),    # x-min: min period
                    metrics_df.period.max(),    # x-max: max period
                    metrics_df.gratDC.min(),    # y-min: min gratDC
                    metrics_df.gratDC.max()     # y-max: max gratDC
                ],
                cmap='viridis'
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_name)
            
            # Set labels
            ax.set_xlabel('Grating Period (μm)')       # Period now on x-axis
            ax.set_ylabel('Grating Duty Cycle (gratDC)')  # gratDC now on y-axis
            ax.set_title(f'{metric_name}')
                
        except Exception as e:
            print(f"Error plotting {metric_name}: {str(e)}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure
    fig_path = os.path.join(sweep_dir, f"sweep_all_metrics_wavelength_{int(wavelength)}.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure with all metrics saved to {fig_path}")
    
    plt.show()
    return fig

def load_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate config structure
        if "simulation" not in config:
            print(f"Error: Missing 'simulation' section in config file: {config_file}")
            raise KeyError(f"Missing 'simulation' section in config file: {config_file}")
        
        # Extract simulation config
        sim_config = config.get('simulation', {})
        
        # Extract output_dir (this is the only parameter we need)
        sweep_dir = sim_config.get('output_dir')
        
        # Check if we can find the sweep_dir in the results section (from sweep_config.json)
        if sweep_dir is None and "results" in config:
            sweep_dir = config["results"].get("sweep_dir")
        
        # Extract post-processing config
        pp_config = config.get('post_processing', {})
        
        return sweep_dir, pp_config
        
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file}")
        raise
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {config_file}")
        print(f"JSON error: {str(e)}")
        raise

def main():
    """Main function for post-processing grating parameter sweep results"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Post-process grating coupler parameter sweep results')
    
    # Input options
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to JSON configuration file')
    parser.add_argument('--metrics', type=str, nargs='+',
                        help='List of metrics to plot (default: plots all key metrics)')
    parser.add_argument('--separate-plots', action='store_true',
                        help='Generate separate plot for each metric instead of subplots')
    
    args = parser.parse_args()
    
    # Load configuration
    sweep_dir, pp_config = load_config(args.config)
    
    # If sweep_dir is not in config, use the directory where config file is located
    if sweep_dir is None:
        sweep_dir = os.path.dirname(os.path.abspath(args.config))
    
    # Extract post-processing parameters
    show_plots = pp_config.get('show_plots', False)
    plot_results = pp_config.get('plot_results', True)
    metric = pp_config.get('metric', 'up_vs_input')
    wavelength = pp_config.get('wavelength')
    metrics_to_plot = args.metrics
    
    # Post-process the results
    metrics_df = post_process_sweep(sweep_dir, show_plots=show_plots)
    
    # Print available metrics for the user
    if metrics_df is not None:
        print("\nAvailable metrics in the data:")
        for col in metrics_df.columns:
            if col not in ['filename', 'status', 'period', 'gratDC', 'pertDC', 'wavelength']:
                print(f"  - {col}")
    
    # Plot results if requested
    if plot_results and metrics_df is not None:
        if wavelength is not None:
            # Plot for a specific wavelength
            plot_sweep_results(sweep_dir, metric=metric, wavelength=wavelength, metrics_to_plot=metrics_to_plot)
        else:
            # Plot for all wavelengths
            for wl in metrics_df.wavelength.unique():
                plot_sweep_results(sweep_dir, metric=metric, wavelength=wl, metrics_to_plot=metrics_to_plot)

if __name__ == "__main__":
    main() 