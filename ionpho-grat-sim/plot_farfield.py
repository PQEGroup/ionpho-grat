import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import argparse
import re
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
from scipy.optimize import curve_fit

def plot_radiation_efficiency(sim_data, sim_data_file, output_dir=None, show_plots=True, save_plots=False):
    """Plot radiation efficiency metrics from flux monitors.
    
    Parameters:
    -----------
    sim_data : td.SimulationData
        The simulation data with flux monitors
    sim_data_file : str
        Path to the simulation data file
    output_dir : str, optional
        Directory to save plots. If None, no plots are saved
    show_plots : bool, optional
        Whether to display the plots interactively
    save_plots : bool, optional
        Whether to save the plots to disk (default: False)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Check if required flux monitors exist
    required_monitors = ['flux_air', 'flux_si', 'flux_end', 'flux_source']
    missing_monitors = [m for m in required_monitors if m not in sim_data.monitor_data]
    
    if missing_monitors:
        print(f"Warning: Missing flux monitors: {', '.join(missing_monitors)}")
        print("Cannot plot radiation efficiency without all required flux monitors.")
        return None, None
    
    # Get flux data from monitors
    flux_air = sim_data['flux_air']
    flux_si = sim_data['flux_si']
    flux_end = sim_data['flux_end']
    flux_source = sim_data['flux_source']
    
    # Sum flux over spatial axes to get power per frequency
    flux_air_arr = np.array(flux_air.flux)
    power_up = np.sum(flux_air_arr, axis=tuple(range(1, flux_air_arr.ndim)))
    
    flux_si_arr = np.array(flux_si.flux)
    power_down = np.sum(flux_si_arr, axis=tuple(range(1, flux_si_arr.ndim)))
    
    flux_end_arr = np.array(flux_end.flux)
    power_end = np.sum(flux_end_arr, axis=tuple(range(1, flux_end_arr.ndim)))
    
    flux_source_arr = np.array(flux_source.flux)
    power_source = np.sum(flux_source_arr, axis=tuple(range(1, flux_source_arr.ndim)))
    print(f"power_source: {power_source}, power_up: {power_up}, power_down: {power_down}, power_end: {power_end}")
    
    # Get frequencies from the simulation frequency array
    # FluxData objects don't have an 'f' attribute, but we can get the frequencies from the monitor
    freqs = np.array(flux_air.flux.f)
    
    # Convert frequencies to wavelengths in nm
    wavelengths = td.C_0 / freqs * 1e3
    
    # Create figure for power bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Number of wavelengths and monitors
    n_wavelengths = len(wavelengths)
    n_monitors = 4  # Source, Air, Si, End
    
    # Width of a group of bars and individual bars
    group_width = 0.8  # Width of all bars for a single wavelength
    bar_width = group_width / n_monitors  # Width of a single bar
    
    # Positions for groups of bars (centered at each integer x value)
    x = np.arange(n_wavelengths)
    
    # Plot absolute power values as grouped bars
    monitor_labels = ['Source', 'Up (Air)', 'Down (Si)', 'End']
    powers = [power_source, power_up, power_down, power_end]
    
    # Plot each power series with appropriate offsets
    offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, n_monitors)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    for i, (power, label, offset, color) in enumerate(zip(powers, monitor_labels, offsets, colors)):
        # Calculate absolute power (take absolute value as flux can be negative)
        abs_power = np.abs(power)
        ax.bar(x + offset, abs_power, width=bar_width, label=label, color=color)
    
    # Set labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absolute Power (arb. units)')
    ax.set_title('Power at Different Flux Monitors')
    
    # Add wavelength labels to x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"{wl:.1f}" for wl in wavelengths], rotation=45)
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to fit labels
    plt.tight_layout()
    
    # Save the plot if requested
    if save_plots and output_dir:
        # Create base filename from input data name
        base_name = os.path.splitext(os.path.basename(sim_data_file))[0]
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save the figure
        filename = f"{base_name}_flux_power.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Saved flux power plot to {save_path}")
    
    # Display the plot if requested
    if not show_plots:
        plt.close()
    
    return fig, ax

def plot_farfield_data(sim_data_file, output_dir=None, show_plots=True, save_plots=False):
    """Plot the far field monitor data from a simulation result file.
    
    Parameters:
    -----------
    sim_data_file : str
        Path to the simulation data file (.hdf5)
    output_dir : str, optional
        Directory to save plots. If None, saves in the same directory as the simulation file
    show_plots : bool, optional
        Whether to display the plots interactively
    save_plots : bool, optional
        Whether to save the plots to disk (default: False)
    """
    # Load the simulation data
    print(f"Loading simulation data from {sim_data_file}")
    sim_data = td.SimulationData.from_file(sim_data_file)
    
    # Extract simulation parameters
    sim = sim_data.simulation
    
    # Find regular far field monitors and the XZ monitor
    far_field_xy_monitors = []
    far_field_xz_monitors = []
    height_monitors = []
    far_field_angle_monitors = []
    
    for monitor in sim.monitors:
        if isinstance(monitor, td.FieldProjectionCartesianMonitor):
            if "far_field_xy_height" in monitor.name:
                height_monitors.append(monitor.name)
            elif "far_field_xy_size" in monitor.name or monitor.name == "far_field":
                far_field_xy_monitors.append(monitor.name)
            elif "far_field_xz" in monitor.name:
                far_field_xz_monitors.append(monitor.name)
        elif isinstance(monitor, td.FieldProjectionAngleMonitor) and "far_field_angle" in monitor.name:
            far_field_angle_monitors.append(monitor.name)
    
    # Sort XY monitors by size
    def get_monitor_size(mon_name):
        if mon_name == "far_field":
            return 30  # Default/base size is 30
        size_match = re.search(r'size(\d+)', mon_name)
        return int(size_match.group(1)) if size_match else 0
    
    # Sort angle monitors by projection distance
    def get_projection_distance(mon_name):
        dist_match = re.search(r'dist(\d+)', mon_name)
        return int(dist_match.group(1)) if dist_match else 50  # Default 50 if not found
    
    # Sort XZ monitors by y-projection distance
    def get_y_projection_distance(mon_name):
        dist_match = re.search(r'ydist(-?\d+)', mon_name)
        return int(dist_match.group(1)) if dist_match else 0  # Default 0 if not found
    
    far_field_xy_monitors.sort(key=get_monitor_size)
    far_field_angle_monitors.sort(key=get_projection_distance)
    far_field_xz_monitors.sort(key=get_y_projection_distance)  # Sort by y-distance
    height_monitors.sort(key=lambda x: int(re.search(r'height(\d+)', x).group(1)))
    
    if not far_field_xy_monitors and not height_monitors and not far_field_angle_monitors and not far_field_xz_monitors:
        print("No far field monitors found in simulation!")
        return
    
    print(f"Found {len(far_field_xy_monitors)} XY monitors, {len(height_monitors)} height-varying monitors, "
          f"{len(far_field_xz_monitors)} XZ monitors, and {len(far_field_angle_monitors)} angle monitors")
    
    # Create output directory if needed
    if save_plots and output_dir is None:
        output_dir = os.path.dirname(sim_data_file)
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Base output filename
    base_name = os.path.splitext(os.path.basename(sim_data_file))[0]
    
    # Process all XY far field monitors together
    if far_field_xy_monitors:
        print(f"Processing {len(far_field_xy_monitors)} XY far field monitors together")
        
        # Number of monitors determines the subplot layout
        num_monitors = len(far_field_xy_monitors)
        if num_monitors <= 2:
            fig_rows, fig_cols = 1, num_monitors
        else:
            fig_rows = 2
            fig_cols = (num_monitors + 1) // 2  # Ceiling division
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*6, fig_rows*5), squeeze=False)
        axes = axes.flatten()
        
        # Prepare intensity data list
        intensity_data = []
        
        # First pass: collect data
        for i, monitor_name in enumerate(far_field_xy_monitors):
            # Find original monitor to get center
            original_monitor = None
            for monitor in sim.monitors:
                if monitor.name == monitor_name:
                    original_monitor = monitor
                    break
                    
            if original_monitor is None:
                print(f"Warning: Could not find original monitor definition for {monitor_name}")
                monitor_center = [0, 0, 0]
            else:
                monitor_center = original_monitor.center
            
            # Extract monitor data
            monitor_data = sim_data[monitor_name]
            
            # Get frequency data 
            freqs = monitor_data.f
            if hasattr(freqs, 'values'):
                freqs = freqs.values
            
            wavelengths = td.C_0 / freqs
            
            # Use the first frequency
            freq = freqs[0]
            wavelength = wavelengths[0]
            
            # Extract field components in spherical coordinates
            Er = monitor_data.Er
            Etheta = monitor_data.Etheta
            Ephi = monitor_data.Ephi
            
            # Select by frequency if needed
            if hasattr(Er, 'sel'):
                Er = Er.sel(f=freq)
                Etheta = Etheta.sel(f=freq)
                Ephi = Ephi.sel(f=freq)
            
            # Extract values if needed
            if hasattr(Er, 'values'):
                Er = Er.values
                Etheta = Etheta.values
                Ephi = Ephi.values
            
            # Calculate intensity
            intensity = np.abs(Er)**2 + np.abs(Etheta)**2 + np.abs(Ephi)**2
            
            # Get coordinate axes
            x_vals = monitor_data.x
            if hasattr(x_vals, 'values'):
                x_vals = x_vals.values
                
            y_vals = monitor_data.y
            if hasattr(y_vals, 'values'):
                y_vals = y_vals.values
            
            # Shift coordinates to be centered at the monitor center
            x_vals_centered = x_vals + monitor_center[0]
            y_vals_centered = y_vals + monitor_center[1]
            
            # Store data for plotting
            intensity_data.append({
                'monitor_name': monitor_name,
                'intensity': intensity,
                'x_vals': x_vals_centered,
                'y_vals': y_vals_centered,
                'size': get_monitor_size(monitor_name),
                'wavelength': wavelength
            })
        
        # Second pass: plot with individual scales
        for i, data in enumerate(intensity_data):
            if i >= len(axes):
                print(f"Warning: Not enough subplots for all monitors. Skipping {data['monitor_name']}")
                continue
                
            ax = axes[i]
            
            # Create intensity plot
            # Transpose the intensity data to match the flipped coordinate order
            intensity_plot = data['intensity'].squeeze().T
            intensity_plot = intensity_plot / np.max(intensity_plot)
            im = ax.pcolormesh(
                data['x_vals'], data['y_vals'], 
                intensity_plot, 
                shading='auto', 
                cmap='inferno'
            )
            
            # Add title and labels
            size_text = f"{data['size']}μm" if data['monitor_name'] != "far_field" else "30μm"
            ax.set_title(f"Far Field - {size_text} window")
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
            ax.set_aspect('equal')
            
            # Add colorbar to each subplot
            plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
            
            # Fit 2D Gaussian to extract beam waists
            def gaussian_2d(coords, A, x0, y0, wx, wy, offset):
                x, y = coords
                return A * np.exp(-2 * ((x - x0)**2 / wx**2 + (y - y0)**2 / wy**2)) + offset
            
            X, Y = np.meshgrid(data['x_vals'], data['y_vals'])
            flat_coords = np.vstack([X.ravel(), Y.ravel()])
            flat_intensity = intensity_plot.ravel()
            
            peak_idx = np.argmax(flat_intensity)
            x0_guess = flat_coords[0, peak_idx]
            y0_guess = flat_coords[1, peak_idx]
            p0 = [1.0, x0_guess, y0_guess, 
                  (data['x_vals'][-1] - data['x_vals'][0]) / 4,
                  (data['y_vals'][-1] - data['y_vals'][0]) / 4, 0.0]
            
            try:
                popt, _ = curve_fit(gaussian_2d, flat_coords, flat_intensity, p0=p0,
                                    bounds=([0, data['x_vals'][0], data['y_vals'][0], 0, 0, -np.inf],
                                            [np.inf, data['x_vals'][-1], data['y_vals'][-1], np.inf, np.inf, np.inf]))
                A_fit, x0_fit, y0_fit, wx_fit, wy_fit, offset_fit = popt
                print(f"  {data['monitor_name']}: beam waist wx = {wx_fit:.2f} μm, wy = {wy_fit:.2f} μm "
                      f"(center: x0={x0_fit:.2f}, y0={y0_fit:.2f})")
                
                # Overlay 1/e^2 contour on the plot
                fit_2d = gaussian_2d(flat_coords, *popt).reshape(X.shape)
                #ax.contour(X, Y, fit_2d, levels=[A_fit * np.exp(-2) + offset_fit],
                #           colors='cyan', linewidths=1.5, linestyles='--')
                ax.set_title(f"Far Field - {size_text} window\n$w_x$={wx_fit:.2f}μm, $w_y$={wy_fit:.2f}μm")
            except RuntimeError:
                print(f"  {data['monitor_name']}: Gaussian fit did not converge")
        
        # Add a common title
        plt.suptitle(f"Far Field Intensity at Different Window Sizes - λ = {wavelength*1e3:.1f} nm", fontsize=16)
        plt.tight_layout()
        
        # Save the combined plot
        if save_plots:
            filename = f"{base_name}_xy_farfield_comparison.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300)
            print(f"Saved XY far field comparison to {save_path}")
        
        if not show_plots:
            plt.close()
    
    # Process XZ monitors if they exist
    if far_field_xz_monitors:
        print(f"Processing {len(far_field_xz_monitors)} XZ far field monitors")
        
        # Create a single figure for all XZ monitors
        fig, axes = plt.subplots(len(far_field_xz_monitors), 1, figsize=(10, len(far_field_xz_monitors)*4), squeeze=False)
        axes = axes.flatten()
        
        # Prepare intensity data list
        intensity_data = []
        
        # First pass: collect data
        for i, monitor_name in enumerate(far_field_xz_monitors):
            print(f"Collecting data from XZ monitor: {monitor_name}")
            
            # Get y-projection distance from monitor name
            y_dist = get_y_projection_distance(monitor_name)
            
            # Find original monitor to get center
            original_monitor = None
            for monitor in sim.monitors:
                if monitor.name == monitor_name:
                    original_monitor = monitor
                    break
                
            if original_monitor is None:
                print(f"Warning: Could not find original monitor definition for {monitor_name}")
                monitor_center = [0, 0, 0]
            else:
                monitor_center = original_monitor.center
            
            # Extract monitor data
            monitor_data = sim_data[monitor_name]
            
            # Get frequency data 
            freqs = monitor_data.f
            if hasattr(freqs, 'values'):
                freqs = freqs.values
            
            wavelengths = td.C_0 / freqs
            
            # Use the first frequency
            freq = freqs[0]
            wavelength = wavelengths[0]
            
            # Extract field components
            Er = monitor_data.Er
            Etheta = monitor_data.Etheta
            Ephi = monitor_data.Ephi
            
            # Select by frequency if needed
            if hasattr(Er, 'sel'):
                Er = Er.sel(f=freq)
                Etheta = Etheta.sel(f=freq)
                Ephi = Ephi.sel(f=freq)
            
            # Extract values if needed
            if hasattr(Er, 'values'):
                Er = Er.values
                Etheta = Etheta.values
                Ephi = Ephi.values
            
            # Calculate intensity
            intensity = np.abs(Er)**2 + np.abs(Etheta)**2 + np.abs(Ephi)**2
            
            # Get coordinate axes
            x_vals = monitor_data.x
            if hasattr(x_vals, 'values'):
                x_vals = x_vals.values
                
            z_vals = monitor_data.z  # y-coordinates are actually z-coordinates for XZ monitor
            if hasattr(z_vals, 'values'):
                z_vals = z_vals.values
            
            # Shift coordinates to be centered at the monitor center
            x_vals_centered = x_vals + monitor_center[0]
            z_vals_centered = z_vals  # No need to shift z values
            
            # Ensure intensity is properly squeezed
            intensity_squeezed = intensity.squeeze()
            
            # Store data for plotting
            intensity_data.append({
                'monitor_name': monitor_name,
                'y_distance': y_dist,
                'intensity': intensity_squeezed,
                'x_vals': x_vals_centered,
                'z_vals': z_vals_centered,
                'wavelength': wavelength
            })
        
        # Second pass: plot all monitors with individual scales
        for i, data in enumerate(intensity_data):
            if i >= len(axes):
                print(f"Warning: Not enough subplots for all monitors. Skipping {data['monitor_name']}")
                continue
            
            ax = axes[i]
            
            # Determine if we need to transpose based on shape
            int_shape = data['intensity'].shape
            x_len = len(data['x_vals'])
            z_len = len(data['z_vals'])

            print(f"Monitor {data['monitor_name']} - Intensity shape: {int_shape}, x_vals: {x_len}, z_vals: {z_len}")
            
            intensity_norm = data['intensity'] / np.max(data['intensity'])

            # Choose how to plot based on dimensions
            if int_shape[0] == x_len and int_shape[1] == z_len:
                im = ax.pcolormesh(
                    data['x_vals'], 
                    data['z_vals'], 
                    intensity_norm.T, 
                    shading='auto', 
                    cmap='inferno'
                )
            else:
                im = ax.pcolormesh(
                    data['x_vals'], 
                    data['z_vals'], 
                    intensity_norm.T, 
                    shading='auto', 
                    cmap='inferno'
                )
            
            # Add colorbar to each subplot
            plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
            
            # Add title and labels
            ax.set_title(f"Y Distance: {data['y_distance']} μm")
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Z (μm)")
        
        # Add overall title
        fig.suptitle(f"Far Field XZ Intensity at Different Y Distances - λ = {wavelength*1e3:.1f} nm", fontsize=16)
        plt.tight_layout()
        
        # Save the combined plot
        if save_plots:
            filename = f"{base_name}_xz_farfield_comparison.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300)
            print(f"Saved XZ far field comparison to {save_path}")
        
        if not show_plots:
            plt.close()
    
    # Process the far field angle monitors
    if far_field_angle_monitors:
        print(f"Processing {len(far_field_angle_monitors)} angle monitors")
        
        # We'll create a single figure with all monitors
        num_monitors = len(far_field_angle_monitors)
        if num_monitors <= 2:
            fig_rows, fig_cols = 1, num_monitors
        else:
            fig_rows = 2
            fig_cols = (num_monitors + 1) // 2  # Ceiling division
        
        # Create a single figure for all monitors
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*7, fig_rows*6), 
                                 subplot_kw={"projection": "polar"}, squeeze=False)
        axes = axes.flatten()
        
        # Prepare intensity data
        intensity_data = []
        
        # First pass: collect data
        for i, monitor_name in enumerate(far_field_angle_monitors):
            print(f"Collecting data from angle monitor: {monitor_name}")
            
            # Get projection distance from monitor name
            dist = get_projection_distance(monitor_name)
            
            # Extract monitor data
            angle_monitor_data = sim_data[monitor_name]
            
            # Get frequency data
            freqs = angle_monitor_data.f
            if hasattr(freqs, 'values'):
                freqs = freqs.values
            
            wavelengths = td.C_0 / freqs
            
            # Use first frequency
            freq = freqs[0]
            wavelength = wavelengths[0]
            
            # Extract field components in spherical coordinates
            Er = angle_monitor_data.Er
            Etheta = angle_monitor_data.Etheta
            Ephi = angle_monitor_data.Ephi
            
            # Select by frequency if needed
            if hasattr(Er, 'sel'):
                Er = Er.sel(f=freq)
                Etheta = Etheta.sel(f=freq)
                Ephi = Ephi.sel(f=freq)
            
            # Extract values if needed
            if hasattr(Er, 'values'):
                Er = Er.values
                Etheta = Etheta.values
                Ephi = Ephi.values
            
            # Calculate intensity
            intensity = np.abs(Er)**2 + np.abs(Etheta)**2 + np.abs(Ephi)**2
            
            # Get angular coordinates
            phi_vals = angle_monitor_data.phi
            if hasattr(phi_vals, 'values'):
                phi_vals = phi_vals.values
                
            theta_vals = angle_monitor_data.theta
            if hasattr(theta_vals, 'values'):
                theta_vals = theta_vals.values
            
            # Ensure intensity is properly squeezed
            if hasattr(intensity, 'shape') and len(intensity.shape) > 2:
                intensity = intensity.squeeze()
            
            # Store data for plotting
            intensity_data.append({
                'monitor_name': monitor_name,
                'projection_distance': dist,
                'intensity': intensity,
                'phi_vals': phi_vals,
                'theta_vals': theta_vals,
                'wavelength': wavelength
            })
        
        # Second pass: plot all monitors with individual scales
        for i, data in enumerate(intensity_data):
            if i >= len(axes):
                print(f"Warning: Not enough subplots for all monitors. Skipping {data['monitor_name']}")
                continue
            
            ax = axes[i]
            
            # Enable grid for better readability
            ax.grid(True, color='gray', alpha=0.5, linestyle='--')
            
            # Add grid lines at specific theta values
            theta_ticks = np.array([15, 30, 45, 60, 75, 90])  # Degrees
            ax.set_rticks(theta_ticks)
            ax.set_rlabel_position(45)  # Position of theta labels (in degrees)
            
            # Create the polar plot (phi as angle, theta as radius)
            im = ax.pcolormesh(
                data['phi_vals'],
                data['theta_vals'] * 180 / np.pi,  # Convert to degrees
                data['intensity'],
                cmap="RdBu",
                shading="auto"
            )
            
            # Add colorbar for this subplot
            fig.colorbar(im, ax=ax, label='Intensity (a.u.)')
            
            # Add title
            ax.set_title(f"Distance: {data['projection_distance']} μm")
            
            # Label the phi axis (only for bottom row)
            if i >= len(axes) - fig_cols:
                ax.set_xlabel(r"$\phi$ (deg)")
            
            # Position the theta label
            label_position = ax.get_rlabel_position()
            ax.text(
                np.radians(label_position - 8),
                ax.get_rmax() / 1.3,
                "$\\theta$ (deg)",
                rotation=label_position,
                ha="center",
                va="center",
            )
        
        # Add overall title
        fig.suptitle(f"Far Field Intensity at Different Projection Distances - λ = {wavelength*1e3:.1f} nm", fontsize=16)
        plt.tight_layout()
        
        # Save the combined plot
        if save_plots:
            filename = f"{base_name}_angle_farfield_comparison.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300)
            print(f"Saved angle far field comparison to {save_path}")
        
        if not show_plots:
            plt.close()
    
    # Process height-varying monitors to create an animation
    if height_monitors:
        print("Creating animation from height-varying monitors...")
        frame_files = []
        
        # Get data without saving individual frames
        for monitor_name in height_monitors:
            height = int(re.search(r'height(\d+)', monitor_name).group(1))
            
            # Find original monitor to get center
            original_monitor = None
            for monitor in sim.monitors:
                if monitor.name == monitor_name:
                    original_monitor = monitor
                    break
                    
            if original_monitor is None:
                print(f"Warning: Could not find original monitor definition for {monitor_name}")
                monitor_center = [0, 0, 0]  # Default if not found
            else:
                monitor_center = original_monitor.center
            
            monitor_data = sim_data[monitor_name]
            
            # Get frequency data 
            freqs = monitor_data.f
            if hasattr(freqs, 'values'):
                freqs = freqs.values
            
            wavelengths = td.C_0 / freqs
            
            # For first wavelength only
            freq = freqs[0]
            wavelength = wavelengths[0]
            
            # Extract field components in spherical coordinates
            Er = monitor_data.Er
            Etheta = monitor_data.Etheta
            Ephi = monitor_data.Ephi
            
            # Select by frequency if needed
            if hasattr(Er, 'sel'):
                Er = Er.sel(f=freq)
                Etheta = Etheta.sel(f=freq)
                Ephi = Ephi.sel(f=freq)
            
            # Extract values if needed
            if hasattr(Er, 'values'):
                Er = Er.values
                Etheta = Etheta.values
                Ephi = Ephi.values
            
            # Calculate intensity
            intensity = np.abs(Er)**2 + np.abs(Etheta)**2 + np.abs(Ephi)**2
            
            # Get coordinate axes
            x_vals = monitor_data.x
            if hasattr(x_vals, 'values'):
                x_vals = x_vals.values
                
            y_vals = monitor_data.y
            if hasattr(y_vals, 'values'):
                y_vals = y_vals.values
            
            # Shift coordinates to be centered at the monitor center
            x_vals_centered = x_vals + monitor_center[0]
            y_vals_centered = y_vals + monitor_center[1]
            
            # Create frame in memory (don't display or save individual frames)
            plt.figure(figsize=(8, 8))
            plt.pcolormesh(y_vals_centered, x_vals_centered, intensity.squeeze(), shading='auto', cmap='inferno')
            plt.colorbar(label='Intensity (a.u.)')
            plt.xlabel("Y (μm)")
            plt.ylabel("X (μm)")
            plt.title(f"Far Field at Height: {height} μm - λ = {wavelength*1e3:.1f} nm")
            plt.axis('equal')
            plt.tight_layout()
            
            # Store frame in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            frame_files.append(img)
            
            plt.close()
        
        # Create the GIF if saving plots
        if save_plots:
            try:
                # Save the GIF 
                gif_filename = f"{base_name}_height_animation.gif"
                gif_path = os.path.join(output_dir, gif_filename)
                
                # Save GIF with 200ms delay between frames
                frame_files[0].save(
                    gif_path, 
                    format='GIF',
                    append_images=frame_files[1:],
                    save_all=True,
                    duration=200,
                    loop=0,
                    optimize=False
                )
                print(f"Saved animation to {gif_path}")
            except Exception as e:
                print(f"Error creating GIF: {e}")
        
        # Display the animation if showing plots
        if show_plots and frame_files:
            # Create animation figure
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.suptitle(f"Far Field Height Animation - λ = {wavelengths[0]*1e3:.1f} nm")
            
            # Display the first frame
            frame = np.array(frame_files[0])
            im = ax.imshow(frame)
            ax.axis('off')
            
            # Create animation
            def animate(i):
                frame = np.array(frame_files[i])
                im.set_array(frame)
                return [im]
            
            ani = FuncAnimation(
                fig, 
                animate, 
                frames=len(frame_files),
                interval=200,
                blit=True
            )
            
            plt.tight_layout()
            print("Displaying height animation...")
    
    # Also plot radiation efficiency from flux monitors
    if 'flux_air' in sim_data.monitor_data and 'flux_si' in sim_data.monitor_data:
        print("Plotting radiation efficiency from flux monitors...")
        plot_radiation_efficiency(sim_data, sim_data_file, output_dir=output_dir, show_plots=show_plots, save_plots=save_plots)
    
    # Show plots if requested
    if show_plots:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot far field monitor data from Tidy3D simulation')
    parser.add_argument('sim_data_file', help='Path to the simulation data file (.hdf5)')
    parser.add_argument('--output-dir', help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true', help='Do not show plots interactively')
    parser.add_argument('--save', action='store_true', help='Save the plots to disk')
    
    args = parser.parse_args()
    
    plot_farfield_data(
        args.sim_data_file, 
        output_dir=args.output_dir, 
        show_plots=not args.no_show,
        save_plots=args.save
    )
