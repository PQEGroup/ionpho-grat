import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import os
import sys
import scipy.io as sio
import argparse
from scipy.signal import find_peaks
import time

# Import plotting utility and helpers
sys.path.append(os.path.dirname(__file__))
from plot_utils import quick_plot, plot_power_distribution, plot_farfield_intensity, plot_farfield_comparison, plot_decay_fits, plot_phase_evolution, plot_decay_fits
from helpers import extract_farfield_peaks, calculate_power_window_fractions, fit_exponential_decay

def calculate_avg_angles(monitor_data, freq_idx=0):
    """
    Calculate power-weighted average angles for Poynting vector and H-field components
    
    Args:
        monitor_data: Field monitor data containing E and H fields
        freq_idx: Frequency index to use for calculations
        
    Returns:
        avg_angle_P: Power-weighted average angle of the Poynting vector (in degrees)
        avg_angle_H: Power-weighted average angle of the H-field (in degrees)
    """
    # Extract E and H field components at the specified frequency
    Ex = np.abs(monitor_data.Ex.isel(f=freq_idx).values)
    Ez = np.abs(monitor_data.Ez.isel(f=freq_idx).values)
    Hx = np.abs(monitor_data.Hx.isel(f=freq_idx).values)
    Hz = np.abs(monitor_data.Hz.isel(f=freq_idx).values)
    
    # Calculate Poynting vector components (approximation based on E and H field magnitudes)
    # In 2D, Px ≈ Ey*Hz - Ez*Hy and Pz ≈ Ex*Hy - Ey*Hx
    # Since we have a TE mode primarily, Ey is dominant, so we approximate:
    Px = Ez * Hx  # Approximate x-component of Poynting vector
    Pz = Ex * Hz  # Approximate z-component of Poynting vector
    
    # Calculate total power magnitude
    P_mag = np.sqrt(Px**2 + Pz**2)
    
    # Calculate angles using arctan2 (returns angles in radians)
    # For Poynting vector - angle between power flow and x-axis
    angles_P = np.arctan2(Pz, Px)
    
    # For H-field - angle between H-field direction and x-axis
    angles_H = np.arctan2(Hz, Hx)
    
    # Calculate power-weighted average of angles
    # Normalize by total power to get weighted average
    # Convert from radians to degrees
    total_power = np.sum(P_mag)
    
    if total_power > 0:
        # Calculate power-weighted average angle for Poynting vector
        avg_angle_P = np.sum(angles_P * P_mag) / total_power * 180 / np.pi
        
        # Calculate power-weighted average angle for H-field
        avg_angle_H = np.sum(angles_H * P_mag) / total_power * 180 / np.pi
    else:
        avg_angle_P = 0
        avg_angle_H = 0
    
    return avg_angle_P, avg_angle_H

def post_process_simulation(simulation_file, sort_by_angle=False, show_plots=True):
    """Process a 2D simulation to extract far-field data and find peak angles
    
    Args:
        simulation_file: Path to the Tidy3D simulation data file
        sort_by_angle: If True, sort peaks by angle from positive to negative.
                       If False (default), sort peaks by height.
        show_plots: If True, display plots of the results (default: True)
    
    Returns:
        peaks_list: List of peak angles for each wavelength
        heights_list: List of peak heights for each wavelength
        power_fractions: Dictionary of power fraction metrics
        avg_angles: Dictionary of average angle metrics
        wavelengths: Array of wavelengths in nm
        fit_results: Dictionary of exponential fit parameters for air and waveguide
        n_eff_values: List of effective indices for each wavelength
    """
    print(f"Processing file: {simulation_file}")
    
    # Load the simulation data
    sim_data = td.SimulationData.from_file(simulation_file)
    
    # Get simulation parameters
    sim_params = sim_data.simulation
    center = sim_params.center
    size = sim_params.size
    structures = sim_params.structures
    freq0 = sim_params.sources[0].source_time.freq0
    # Infer grating length from simulation center (center_x = l_grat/2)
    l_grat = 2 * center[0]
    
    # Extract field data from the field_top monitor
    field_data = sim_data["field_top"]
    x_coords = np.array(field_data.Ex.x)
    z_coords = np.array(field_data.Ex.z)
    
    # Extract field components for all frequencies
    freqs = np.array(field_data.Ex.f)
    num_freqs = len(freqs)
    
    # Compute wavelengths in nanometers
    wavelengths = td.C_0 / freqs * 1e3
    
    # Calculate number of subplots needed
    num_rows = (num_freqs + 1) // 2  # Ceiling division to get enough rows

    # Prepare fit parameter dicts and data lists for decay fits
    fit_params_air = {}
    fit_params_wg = {}
    x_air_list, I_air_list, x_fit_air_list, y_fit_air_list = [], [], [], []
    x_wg_list, I_wg_list, x_fit_wg_list, y_fit_wg_list = [], [], [], []
    
    for freq_idx in range(num_freqs):
        wl = wavelengths[freq_idx]
        
        # Extract field components for air (field_top monitor)
        Ex = np.array(field_data.Ex.isel(f=freq_idx))
        Ey = np.array(field_data.Ey.isel(f=freq_idx))
        Ez = np.array(field_data.Ez.isel(f=freq_idx))
        
        # Calculate intensity for air
        I_air = np.abs(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
        I_air_line = np.squeeze(I_air[:, 0])
        
        # Fit exponential decay to air intensity
        params_air, x_fit_air, y_fit_air = fit_exponential_decay(x_coords, I_air_line)
        if params_air:
            fit_params_air[freq_idx] = params_air
            x_air_list.append(x_coords)
            I_air_list.append(I_air_line)
            x_fit_air_list.append(x_fit_air)
            y_fit_air_list.append(y_fit_air)

        # Extract field components for waveguide (field_wg monitor)
        wg_data = sim_data['field_wg']
        x_wg = np.array(wg_data.Ex.x)
        
        Ex_wg = np.array(wg_data.Ex.isel(f=freq_idx))
        Ey_wg = np.array(wg_data.Ey.isel(f=freq_idx))
        Ez_wg = np.array(wg_data.Ez.isel(f=freq_idx))
        
        # Calculate intensity for waveguide
        I_wg = np.abs(Ex_wg * np.conj(Ex_wg) + Ey_wg * np.conj(Ey_wg) + Ez_wg * np.conj(Ez_wg))
        I_wg_line = np.squeeze(I_wg[:, 0])
        
        # Fit exponential decay starting from 1μm after the grating start, exclude post-grating region
        grating_start_plus_1um = 1.0  # 1μm after grating start (grating starts at x=0)
        params_wg, x_fit_wg, y_fit_wg = fit_exponential_decay(x_wg, I_wg_line, 
                                                               start_from=grating_start_plus_1um, 
                                                               exclude_after=l_grat)
        if params_wg:
            fit_params_wg[freq_idx] = params_wg
            x_wg_list.append(x_wg)
            I_wg_list.append(I_wg_line)
            x_fit_wg_list.append(x_fit_wg)
            y_fit_wg_list.append(y_fit_wg)
    
    # Extract effective index from field_wg_neff monitor for each frequency
    wg_neff_data = sim_data['field_wg_neff']
    x_neff = np.array(wg_neff_data.Ey.x)
    n_eff_values = []
    
    # Compute phase and effective index (plotting via plot_phase_evolution)
    for freq_idx in range(num_freqs):
        wl = wavelengths[freq_idx]
        
        # Extract Ey field for this frequency ## TODO: Check if this is correct - why is it Ey? and why not Ez? It was Ez in the original code. Don't think it matters too much though.
        Ey = np.array(wg_neff_data.Ey.isel(f=freq_idx))
        Ey_line = np.squeeze(Ey[:, 0])
        
        # Calculate phase and unwrap
        phase = np.angle(Ey_line)
        phase_unwrap = np.unwrap(phase)
        
        # Compute k0 from wavelength (in μm^-1)
        wavelength_um = wl * 1e-3  # nm to μm
        k0 = 2 * np.pi / wavelength_um
        
        # Effective index calculation
        n_eff = (phase_unwrap[-1] - phase_unwrap[0]) / (k0 * (x_neff[-1] - x_neff[0]))
        n_eff_values.append(n_eff)
    
    # Continue with original functionality (except for the single-frequency plots)
    
    # Calculate power ratios
    # Get flux data from monitors
    flux_air = sim_data['flux_air']
    flux_si = sim_data['flux_si']
    flux_end = sim_data['flux_end']
    flux_start = sim_data['flux_start']
    # Sum flux over spatial axes to get power per frequency
    flux_air_arr = np.array(flux_air.flux)
    power_up = np.sum(flux_air_arr, axis=tuple(range(1, flux_air_arr.ndim)))
    flux_si_arr = np.array(flux_si.flux)
    power_down = np.sum(flux_si_arr, axis=tuple(range(1, flux_si_arr.ndim)))
    flux_end_arr = np.array(flux_end.flux)
    power_end = np.sum(flux_end_arr, axis=tuple(range(1, flux_end_arr.ndim)))
    flux_start_arr = np.array(flux_start.flux)
    power_start = np.sum(flux_start_arr, axis=tuple(range(1, flux_start_arr.ndim)))

    # Use built-in far-field data from the 'far_field' monitor
    ff = sim_data['far_field']  # FieldProjectionAngleData
    theta = np.array(ff.theta)
    
    # Prepare far-field data for plotting
    ff_data = {
        'Etheta': [np.abs(ff.Etheta.isel(f=i)) for i in range(len(freqs))],
        'Ephi': [np.abs(ff.Ephi.isel(f=i)) for i in range(len(freqs))],
        'Er':    [np.abs(ff.Er.isel(f=i))    for i in range(len(freqs))]
    }
    
    # Extract actual far-field peaks and heights for each frequency
    peaks_list, heights_list = extract_farfield_peaks(theta, ff_data,
                                                    height_threshold=0.2,
                                                    distance_min=10,
                                                    sort_by_angle=sort_by_angle)

    # Calculate average angles for each frequency
    num_freq = len(freqs)
    avg_P_list, avg_H_list = [], []
    for idx in range(num_freq):
        P_i, H_i = calculate_avg_angles(field_data, freq_idx=idx)
        avg_P_list.append(P_i)
        avg_H_list.append(H_i)

    avg_angles = {
        'poynting_vector': avg_P_list,
        'h_field':         avg_H_list
    }

    # Compute power fractions for each frequency
    # Avoid division by zero
    up_vs_input = np.where(power_start != 0, power_up / power_start, 0)
    down_vs_input = np.where(power_start != 0, power_down / power_start, 0)
    up_vs_down = np.where(power_down != 0, power_up / power_down, 0)
    transmitted_vs_input = np.where(power_start != 0, power_end / power_start, 0)
    # Create dictionaries with lists for each frequency
    power_fractions = {
        'up_vs_input': up_vs_input.tolist(),
        'down_vs_input': down_vs_input.tolist(),
        'up_vs_down': up_vs_down.tolist(),
        'transmitted_vs_input': transmitted_vs_input.tolist()
    }
    
    # Add fit parameters to the output
    fit_results = {
        'air': fit_params_air,
        'waveguide': fit_params_wg
    }

    # Only generate plots if show_plots is True
    if show_plots:
        # Use plot_utils for exponential decay fits
        plot_decay_fits(wavelengths, x_air_list, I_air_list, fit_params_air, x_fit_air_list, y_fit_air_list,
                        'Electric Field Intensity in Air with Exponential Fits')
        plot_decay_fits(wavelengths, x_wg_list, I_wg_list, fit_params_wg, x_fit_wg_list, y_fit_wg_list,
                        'Electric Field Intensity in Waveguide with Exponential Fits')
        
        # Use plot_utils for phase evolution
        # Prepare phase data lists
        phase_list = []
        phase_unwrap_list = []
        for freq_idx in range(num_freqs):
            Ey = np.array(wg_neff_data.Ey.isel(f=freq_idx))
            Ey_line = np.squeeze(Ey[:, 0])
            phase = np.angle(Ey_line)
            phase_unwrap = np.unwrap(phase)
            phase_list.append(phase)
            phase_unwrap_list.append(phase_unwrap)
        plot_phase_evolution(wavelengths, x_neff, phase_list, phase_unwrap_list, n_eff_values)
        
        # Now show power distribution and far-field comparison
        plot_power_distribution(power_up, power_down, power_end, power_start, freqs)
        plot_farfield_comparison(theta, ff_data, freqs, upward_frac=power_up / power_start if power_start.size > 1 else None)
    
    return peaks_list, heights_list, power_fractions, avg_angles, wavelengths, fit_results, n_eff_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process Tidy3D 2D grating simulation results')
    parser.add_argument('--file', type=str, help='Single simulation file to process')
    parser.add_argument('--dir', type=str, help='Directory containing simulation files to process')
    parser.add_argument('--sort-by-angle', action='store_true',
                        help='Sort peaks by angle (positive to negative) instead of height')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plotting of results')
    
    args = parser.parse_args()
    
    if args.file:
        peaks_list, heights_list, power_fractions, avg_angles, wavelengths, fit_results, n_eff_values = post_process_simulation(
            args.file,
            sort_by_angle=args.sort_by_angle,
            show_plots=not args.no_plots
        )
        
        print("\nSummary:")
        print(f"Peaks sorted by: {'angle (+ to -)' if args.sort_by_angle else 'height (descending)'}")
        
        # Display far-field peaks by wavelength
        for wl, pks, hts in zip(wavelengths, peaks_list, heights_list):
            print(f"\nWavelength {wl:.1f} nm Far-field Peaks:")
            if len(pks) > 0:
                for j, (pk, ht) in enumerate(zip(pks, hts)):
                    print(f"  Peak {j+1}: Angle = {pk:.2f}°, Height = {ht:.4f}")
            else:
                print("  No far-field peaks detected")
        
        # Display average angles by wavelength
        print("\nAverage Angles per Wavelength:")
        for wl, P_i, H_i in zip(wavelengths, avg_angles['poynting_vector'], avg_angles['h_field']):
            print(f"  Wavelength {wl:.1f} nm: Poynting = {P_i:.2f}°, H-field = {H_i:.2f}°")
        
        # Display power fractions per wavelength
        print("\nPower Fractions per Wavelength:")
        for i, wl in enumerate(wavelengths):
            print(f"\nWavelength {wl:.1f} nm:")
            for fraction, arr in power_fractions.items():
                print(f"  {fraction}: {arr[i]:.4f}")
            
        # Display exponential fit results
        print("\nExponential Decay Fit Results:")
        print("Air intensity decay rates:")
        for freq_idx, params in fit_results['air'].items():
            wl = wavelengths[freq_idx]
            print(f"  Wavelength {wl:.1f} nm: Alpha = {params['alpha']:.4f} μm⁻¹, A = {params['A']:.4e}, Offset = {params['offset']:.4e}")
            
        print("\nWaveguide intensity decay rates:")
        for freq_idx, params in fit_results['waveguide'].items():
            wl = wavelengths[freq_idx]
            print(f"  Wavelength {wl:.1f} nm: Alpha = {params['alpha']:.4f} μm⁻¹, A = {params['A']:.4e}, Offset = {params['offset']:.4e}")
            
        # Display effective indices
        print("\nEffective Indices:")
        for wl, n_eff in zip(wavelengths, n_eff_values):
            print(f"  Wavelength {wl:.1f} nm: n_eff = {n_eff:.4f}")
            
    elif args.dir:
        print("Batch processing is not supported in the new version. Please specify a single file (--file) to process.")
        parser.print_help()
    else:
        print("Please specify either a file (--file) or directory (--dir) to process")
        parser.print_help() 