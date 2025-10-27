"""
Look-Up Table (LUT) handling module for AIM Grating Design

This module contains functions for loading, processing, and transforming
lookup tables used in grating design.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import Dict, Tuple, List, Any, Union, Optional

# Import from our modules
from . import helper_functions

# Type aliases
Array = np.ndarray

def load_and_process_lut(p: Dict[str, Any]) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Load and process lookup tables based on parameters.
    
    Args:
        p: Dictionary of parameters including wavelength, material, etc.
        
    Returns:
        Tuple of (grating periods, duty cycles, alpha values, theta values, 
        effective index values)
    """
    wavelength_nm = round(p["lambda"] * 1e3)
    print(f"Loading LUT for wavelength: {wavelength_nm} nm")
    
    # Get alpha calculation parameters
    useAlphasAir = p.get("useAlphasAir", True)
    useGeometricMeanAlpha = p.get("useGeometricMeanAlpha", True)
    
    # Set up LUT parameters
    sampFac = 5  # For LUT interpolation
    maxDC = 0.6  # Maximum duty cycle
    
    # Create default parameters if not present
    if "LUTparams" not in p:
        p["LUTparams"] = {"toxt": 1000}
    
    p["LUTparams"]["forward_em"] = p.get("forward_em", False)
    
    # Load LUT data
    LUT = helper_functions.loadLUT_AIM(p["lambda"], maxDC, p)
    
    # Get interpolated LUT
    LUT2 = helper_functions.getInterpLUT_gen(sampFac, LUT)
    
    # Select alpha values based on configuration
    if not useAlphasAir:
        LUTalph1 = LUT2["alphasWG"]
        if wavelength_nm == 732:
            LUTalph1 = LUT2["alphasInWG"]
    else:
        LUTalph1 = LUT2["alphasAir"]
    
    # Use geometric mean of alpha if requested
    if useGeometricMeanAlpha:
        LUTalph1 = np.sqrt(LUT2["alphasWG"] * LUT2["alphasAir"])
    
    # Select theta values
    if 'thetasFF' in LUT2:
        LUTthet1 = -1 * LUT2["thetasFF"]
    else:
        LUTthet1 = LUT2["thetasH"]
    
    # Select effective index values
    LUTnEffs1 = -1 * LUT2["nEffs"]
    
    # Get grating parameters
    gp = LUT2["gratpers"]
    dc = LUT2["pertDCs"]
    
    # Apply mask
    LUTalph, LUTthet, _ = helper_functions.maskLUT_AIM(gp, dc, LUTalph1, LUTthet1, p["minSize"], p["lambda"], 0, p["LUTparams"])
    LUTnEffs, _, _ = helper_functions.maskLUT_AIM(gp, dc, LUTnEffs1, LUTnEffs1, p["minSize"], p["lambda"], 0, p["LUTparams"])
    
    return gp, dc, LUTalph, LUTthet, LUTnEffs


def load_csv_data(csv_file_path: str, wavelength_filter: Optional[float] = None) -> Dict[str, Array]:
    """
    Load sweep metrics data from CSV file and convert to LUT format.
    
    Args:
        csv_file_path: Path to the CSV file containing sweep metrics
        wavelength_filter: Optional wavelength to filter data (in nm)
        
    Returns:
        Dictionary containing LUT data compatible with existing processing functions
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Read CSV data
    df = pd.read_csv(csv_file_path)
    
    # Filter by wavelength if specified
    if wavelength_filter is not None:
        df = df[df['wavelength'] == wavelength_filter]
        if len(df) == 0:
            raise ValueError(f"No data found for wavelength {wavelength_filter} nm")
    
    print(f"Loaded {len(df)} rows from CSV file")
    if wavelength_filter:
        print(f"Filtered to {len(df)} rows for wavelength {wavelength_filter} nm")
    
    # Extract unique periods and duty cycles to create grid structure
    unique_periods = np.sort(df['period'].unique())
    unique_grat_dcs = np.sort(df['gratDC'].unique()) 
    unique_pert_dcs = np.sort(df['pertDC'].unique())
    
    print(f"Periods: {len(unique_periods)} values from {unique_periods[0]:.3f} to {unique_periods[-1]:.3f}")
    print(f"Grating DCs: {len(unique_grat_dcs)} values from {unique_grat_dcs[0]:.3f} to {unique_grat_dcs[-1]:.3f}")
    print(f"Perturbation DCs: {len(unique_pert_dcs)} values from {unique_pert_dcs[0]:.3f} to {unique_pert_dcs[-1]:.3f}")
    
    # Convert periods to nanometers (as expected by existing code)
    gratpers_nm = unique_periods * 1000  # Convert μm to nm
    
    # Create output dictionary matching expected LUT structure
    LUT = {
        'gratpersSave': gratpers_nm,
        'pertDCsSave': unique_pert_dcs,
        'alphasWGSave': np.zeros((len(unique_pert_dcs), len(unique_periods))),
        'alphasAirSave': np.zeros((len(unique_pert_dcs), len(unique_periods))),
        'thetasHSave': np.zeros((len(unique_pert_dcs), len(unique_periods))),
        'thetasPoyntingSave': np.zeros((len(unique_pert_dcs), len(unique_periods))),
        'nEffsSave': np.zeros((len(unique_pert_dcs), len(unique_periods))),
        'thetasFFSave': np.zeros((len(unique_pert_dcs), len(unique_periods)))
    }
    
    # Fill in the data arrays
    for _, row in df.iterrows():
        # Find indices
        period_idx = np.where(unique_periods == row['period'])[0]
        pert_dc_idx = np.where(unique_pert_dcs == row['pertDC'])[0]
        
        if len(period_idx) > 0 and len(pert_dc_idx) > 0:
            p_idx = period_idx[0]
            dc_idx = pert_dc_idx[0]
            
            # Fill data arrays (note: arrays are indexed as [dc, period])
            LUT['alphasWGSave'][dc_idx, p_idx] = row['alpha_wg']
            LUT['alphasAirSave'][dc_idx, p_idx] = row['alpha_air']
            LUT['thetasHSave'][dc_idx, p_idx] = row['avg_angle_H']
            LUT['thetasPoyntingSave'][dc_idx, p_idx] = row['avg_angle_P']
            LUT['nEffsSave'][dc_idx, p_idx] = row['n_eff']
            
            # Use max_peak_angle for far-field theta if available
            if 'max_peak_angle' in row and not pd.isna(row['max_peak_angle']):
                LUT['thetasFFSave'][dc_idx, p_idx] = row['max_peak_angle']
            else:
                LUT['thetasFFSave'][dc_idx, p_idx] = row['avg_angle_H']
    
    return LUT


def load_and_process_csv_lut(csv_file_path: str, p: Dict[str, Any]) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Load and process CSV-based lookup tables.
    
    Args:
        csv_file_path: Path to the CSV file containing sweep metrics
        p: Dictionary of parameters including wavelength, material, etc.
        
    Returns:
        Tuple of (grating periods, duty cycles, alpha values, theta values, 
        effective index values)
    """
    wavelength_nm = round(p["lambda"] * 1e3)
    print(f"Loading CSV LUT for wavelength: {wavelength_nm} nm")
    
    # Get alpha calculation parameters
    useAlphasAir = p.get("useAlphasAir", True)
    useGeometricMeanAlpha = p.get("useGeometricMeanAlpha", True)
    
    # Set up LUT parameters
    sampFac = 5  # For LUT interpolation
    maxDC = 0.6  # Maximum duty cycle
    
    # Create default parameters if not present
    if "LUTparams" not in p:
        p["LUTparams"] = {"toxt": 1000}
    
    p["LUTparams"]["forward_em"] = p.get("forward_em", False)
    
    # Load CSV data
    LUT = load_csv_data(csv_file_path, wavelength_nm)
    
    # Get interpolated LUT using existing function
    LUT2 = helper_functions.getInterpLUT_gen(sampFac, LUT)
    
    # Select alpha values based on configuration
    if not useAlphasAir:
        LUTalph1 = LUT2["alphasWG"]
        if wavelength_nm == 732:
            LUTalph1 = LUT2["alphasInWG"] if "alphasInWG" in LUT2 else LUT2["alphasWG"]
    else:
        LUTalph1 = LUT2["alphasAir"]
    
    # Use geometric mean of alpha if requested
    if useGeometricMeanAlpha:
        LUTalph1 = np.sqrt(LUT2["alphasWG"] * LUT2["alphasAir"])
    
    # Select theta values
    if 'thetasFF' in LUT2:
        LUTthet1 = -1 * LUT2["thetasFF"]
    else:
        LUTthet1 = LUT2["thetasH"]
    
    # Select effective index values
    LUTnEffs1 = -1 * LUT2["nEffs"]
    
    # Get grating parameters
    gp = LUT2["gratpers"]
    dc = LUT2["pertDCs"]
    
    # Apply mask using existing function
    LUTalph, LUTthet, _ = helper_functions.maskLUT_AIM(gp, dc, LUTalph1, LUTthet1, p["minSize"], p["lambda"], 0, p["LUTparams"])
    LUTnEffs, _, _ = helper_functions.maskLUT_AIM(gp, dc, LUTnEffs1, LUTnEffs1, p["minSize"], p["lambda"], 0, p["LUTparams"])
    
    return gp, dc, LUTalph, LUTthet, LUTnEffs


def smooth_data(data, window, isnan=None):
    """
    Smooth data with a moving average, handling NaN values
    
    Args:
        data: Data to smooth
        window: Window size for smoothing
        isnan: Boolean array indicating NaN locations (optional)
        
    Returns:
        Smoothed data
    """
    if isnan is None:
        isnan = np.isnan(data)
    
    data_clean = np.copy(data)
    if np.sum(~isnan) >= window:
        # Use uniform filter for smoothing
        smoothed = uniform_filter1d(np.where(isnan, 0, data), size=window, mode='nearest')
        count = uniform_filter1d(~isnan * 1.0, size=window, mode='nearest')
        smoothed_valid = smoothed / (count + (count == 0))  # Avoid division by zero
        data_clean[~isnan] = smoothed_valid[~isnan]
        
    return data_clean


def plot_lut_2d(gp: Array, dc: Array, LUTalph: Array, LUTthet: Array, LUTnEffs: Array, 
                p: Dict[str, Any], show_plots: bool = False, combined_only: bool = True,
                design_data: Optional[Dict[str, Any]] = None) -> List[plt.Figure]:
    """
    Create 2D plots of the LUT data showing alpha, theta, and effective index.
    
    Args:
        gp: Grating periods array (μm)
        dc: Duty cycles array 
        LUTalph: Alpha values (waveguide coupling strength) 
        LUTthet: Theta values (far field angles) in degrees
        LUTnEffs: Effective index values
        p: Parameters dictionary
        show_plots: Whether to display plots immediately
        combined_only: If True, only create the combined subplot figure (default: True)
        design_data: Optional dictionary containing design results with 'desperxx' and 'desDCxx'
        
    Returns:
        List of figure handles for the 2D LUT plots
    """
    # Grating periods are already in μm, no conversion needed
    gp_um = gp
    
    # Create meshgrid for contour plots
    DC_mesh, GP_mesh = np.meshgrid(dc, gp_um, indexing='ij')
    
    figures = []
    wavelength_nm = round(p["lambda"] * 1e3)
    
    # Calculate levels for consistent plotting
    levels_alpha = np.linspace(np.nanmin(LUTalph), np.nanmax(LUTalph), 20)
    levels_theta = np.linspace(np.nanmin(LUTthet), np.nanmax(LUTthet), 20)
    levels_neff = np.linspace(np.nanmin(LUTnEffs), np.nanmax(LUTnEffs), 20)
    
    # Extract design points if provided
    design_periods = None
    design_dcs = None
    if design_data is not None:
        if 'desperxx' in design_data and 'desDCxx' in design_data:
            design_periods = design_data['desperxx']
            design_dcs = design_data['desDCxx']
            # Filter out invalid points (NaN or zero)
            valid_mask = ~(np.isnan(design_periods) | np.isnan(design_dcs) | (design_dcs == 0))
            design_periods = design_periods[valid_mask]
            design_dcs = design_dcs[valid_mask]
            print(f"Found {len(design_periods)} valid design points to overlay on LUT plots")
    
    if not combined_only:
        # Figure 1: Alpha (waveguide coupling strength) vs period and duty cycle
        fig_alpha = plt.figure(figsize=(12, 8))
        
        # Create contour plot
        cs_alpha = plt.contourf(GP_mesh, DC_mesh, LUTalph, levels=levels_alpha, cmap='viridis')
        
        # Add contour lines
        cs_lines_alpha = plt.contour(GP_mesh, DC_mesh, LUTalph, levels=levels_alpha[::2], colors='white', alpha=0.5, linewidths=0.5)
        plt.clabel(cs_lines_alpha, inline=True, fontsize=8, fmt='%.3f')
        
        # Colorbar and labels
        cbar_alpha = plt.colorbar(cs_alpha)
        cbar_alpha.set_label('Alpha (μm⁻¹)', rotation=270, labelpad=20)
        
        plt.xlabel('Grating Period (μm)')
        plt.ylabel('Duty Cycle')
        plt.title(f'Waveguide Alpha vs Period and Duty Cycle\n(λ = {wavelength_nm} nm)')
        plt.grid(True, alpha=0.3)
        
        # Overlay design points if available
        if design_periods is not None and design_dcs is not None:
            plt.scatter(design_periods, design_dcs, c='red', s=20, alpha=0.7, 
                       edgecolors='white', linewidths=0.5, label='Design Points')
            plt.legend()
        
        figures.append(fig_alpha)
        
        # Figure 2: Theta (far field angle) vs period and duty cycle
        fig_theta = plt.figure(figsize=(12, 8))
        
        # Create contour plot
        cs_theta = plt.contourf(GP_mesh, DC_mesh, LUTthet, levels=levels_theta, cmap='plasma')
        
        # Add contour lines
        cs_lines_theta = plt.contour(GP_mesh, DC_mesh, LUTthet, levels=levels_theta[::2], colors='white', alpha=0.5, linewidths=0.5)
        plt.clabel(cs_lines_theta, inline=True, fontsize=8, fmt='%.1f')
        
        # Colorbar and labels
        cbar_theta = plt.colorbar(cs_theta)
        cbar_theta.set_label('Theta (degrees)', rotation=270, labelpad=20)
        
        plt.xlabel('Grating Period (μm)')
        plt.ylabel('Duty Cycle')
        plt.title(f'Far Field Angle (Theta) vs Period and Duty Cycle\n(λ = {wavelength_nm} nm)')
        plt.grid(True, alpha=0.3)
        
        # Overlay design points if available
        if design_periods is not None and design_dcs is not None:
            plt.scatter(design_periods, design_dcs, c='red', s=20, alpha=0.7, 
                       edgecolors='white', linewidths=0.5, label='Design Points')
            plt.legend()
        
        figures.append(fig_theta)
        
        # Figure 3: Effective Index vs period and duty cycle
        fig_neff = plt.figure(figsize=(12, 8))
        
        # Create contour plot
        cs_neff = plt.contourf(GP_mesh, DC_mesh, LUTnEffs, levels=levels_neff, cmap='coolwarm')
        
        # Add contour lines
        cs_lines_neff = plt.contour(GP_mesh, DC_mesh, LUTnEffs, levels=levels_neff[::2], colors='black', alpha=0.5, linewidths=0.5)
        plt.clabel(cs_lines_neff, inline=True, fontsize=8, fmt='%.3f')
        
        # Colorbar and labels
        cbar_neff = plt.colorbar(cs_neff)
        cbar_neff.set_label('Effective Index', rotation=270, labelpad=20)
        
        plt.xlabel('Grating Period (μm)')
        plt.ylabel('Duty Cycle')
        plt.title(f'Effective Index vs Period and Duty Cycle\n(λ = {wavelength_nm} nm)')
        plt.grid(True, alpha=0.3)
        
        # Overlay design points if available
        if design_periods is not None and design_dcs is not None:
            plt.scatter(design_periods, design_dcs, c='red', s=20, alpha=0.7, 
                       edgecolors='white', linewidths=0.5, label='Design Points')
            plt.legend()
        
        figures.append(fig_neff)
    
    # Figure 4: Combined view with subplots
    fig_combined = plt.figure(figsize=(18, 6))
    
    # Alpha subplot
    ax1 = plt.subplot(1, 3, 1)
    cs1 = plt.contourf(GP_mesh, DC_mesh, LUTalph, levels=levels_alpha, cmap='viridis')
    plt.contour(GP_mesh, DC_mesh, LUTalph, levels=levels_alpha[::3], colors='white', alpha=0.5, linewidths=0.5)
    cbar1 = plt.colorbar(cs1, ax=ax1, shrink=0.8)
    cbar1.set_label('Alpha (μm⁻¹)', rotation=270, labelpad=15)
    plt.xlabel('Grating Period (μm)')
    plt.ylabel('Duty Cycle')
    plt.title('Alpha (Waveguide)')
    plt.grid(True, alpha=0.3)
    
    # Overlay design points if available
    if design_periods is not None and design_dcs is not None:
        plt.scatter(design_periods, design_dcs, c='red', s=15, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, label='Design Points')
    
    # Theta subplot
    ax2 = plt.subplot(1, 3, 2)
    cs2 = plt.contourf(GP_mesh, DC_mesh, LUTthet, levels=levels_theta, cmap='plasma')
    plt.contour(GP_mesh, DC_mesh, LUTthet, levels=levels_theta[::3], colors='white', alpha=0.5, linewidths=0.5)
    cbar2 = plt.colorbar(cs2, ax=ax2, shrink=0.8)
    cbar2.set_label('Theta (degrees)', rotation=270, labelpad=15)
    plt.xlabel('Grating Period (μm)')
    plt.ylabel('Duty Cycle')
    plt.title('Far Field Angle')
    plt.grid(True, alpha=0.3)
    
    # Overlay design points if available
    if design_periods is not None and design_dcs is not None:
        plt.scatter(design_periods, design_dcs, c='red', s=15, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, label='Design Points')
    
    # Effective Index subplot
    ax3 = plt.subplot(1, 3, 3)
    cs3 = plt.contourf(GP_mesh, DC_mesh, LUTnEffs, levels=levels_neff, cmap='coolwarm')
    plt.contour(GP_mesh, DC_mesh, LUTnEffs, levels=levels_neff[::3], colors='black', alpha=0.5, linewidths=0.5)
    cbar3 = plt.colorbar(cs3, ax=ax3, shrink=0.8)
    cbar3.set_label('Effective Index', rotation=270, labelpad=15)
    plt.xlabel('Grating Period (μm)')
    plt.ylabel('Duty Cycle')
    plt.title('Effective Index')
    plt.grid(True, alpha=0.3)
    
    # Overlay design points if available
    if design_periods is not None and design_dcs is not None:
        plt.scatter(design_periods, design_dcs, c='red', s=15, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, label='Design Points')
    
    # Add legend to the last subplot if design points are present
    if design_periods is not None and design_dcs is not None:
        plt.legend(loc='upper right', fontsize=10)
    
    title_text = f'LUT Data Overview (λ = {wavelength_nm} nm)'
    if design_periods is not None and design_dcs is not None:
        title_text += f' - Red dots show {len(design_periods)} design points'
    plt.suptitle(title_text, fontsize=16)
    plt.tight_layout()
    
    figures.append(fig_combined)
    
    if show_plots:
        plt.show()
    
    plot_type = "combined" if combined_only else "individual + combined"
    print(f"Generated {len(figures)} LUT 2D plot(s) ({plot_type})")
    print(f"  - Period range: {gp_um.min():.3f} to {gp_um.max():.3f} μm")
    print(f"  - Duty cycle range: {dc.min():.3f} to {dc.max():.3f}")
    print(f"  - Alpha range: {np.nanmin(LUTalph):.3f} to {np.nanmax(LUTalph):.3f} μm⁻¹")
    print(f"  - Theta range: {np.nanmin(LUTthet):.1f} to {np.nanmax(LUTthet):.1f} degrees")
    print(f"  - Effective index range: {np.nanmin(LUTnEffs):.3f} to {np.nanmax(LUTnEffs):.3f}")
    
    return figures 