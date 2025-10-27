"""
Longitudinal grating design module

This module handles the longitudinal design of gratings, including
calculating coupling strength (alpha), duty cycles, and periods.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from typing import Dict, List, Tuple, Any, Optional, Union

# Import from our modules
from . import helper_functions

# Type aliases
Array = np.ndarray

def create_longitudinal_design(p: Dict[str, Any], phi_fs: Array, EfieldAmp: Array, 
                             theta0: Array, Xwg: Array, Ywg: Array, 
                             waist_info: Dict[str, Any], gp: Array, dc: Array, 
                             LUTalph: Array, LUTthet: Array) -> Dict[str, Any]:
    """
    Create the longitudinal grating design based on beam parameters and LUTs.
    
    Args:
        p: Parameters dictionary
        phi_fs: Phase of the beam
        EfieldAmp: E-field amplitude
        theta0: Beam angle
        Xwg, Ywg: Waveguide coordinates
        waist_info: Beam waist information
        gp: Grating periods array
        dc: Duty cycles array
        LUTalph: Alpha values from LUT
        LUTthet: Theta values from LUT
        
    Returns:
        Dictionary containing the longitudinal design results
    """
    # Extract parameters
    nPts = p["nPts"]
    XwgWaists = waist_info["XwgWaists"]
    clipBeginning = p.get("clipBeginning", 0.85)
    clipEnd = p.get("clipEnd", 1.52)
    hard_clip = p.get("hard_clip", False)
    nu = p.get("nu", 1.0)
    minSize = p.get("minSize", 0.075)
    
    # Get centerline coordinates
    XwgCenter = Xwg[int(nPts // 2), :]
    
    # Calculate differential steps along centerline
    dxwg = np.diff(XwgCenter, prepend=XwgCenter[0])
    
    # Get centerline E-field and normalize
    Ecenter = EfieldAmp[int(nPts // 2), :]
    normfac = trapezoid(dxwg * Ecenter**2)
    Ecenter = Ecenter / np.sqrt(normfac)
    
    # Optional top-hat intensity
    if p.get("step_intensity", False):
        Ecenter = np.zeros_like(Ecenter)
        insideRegion = (XwgCenter > XwgWaists[2] - abs(XwgWaists[2] - XwgWaists[0]) * clipBeginning) & \
                       (XwgCenter < XwgWaists[2] + abs(XwgWaists[1] - XwgWaists[2]) * clipEnd)
        Ecenter[insideRegion] = 1
        normfac = trapezoid(dxwg * Ecenter**2)
        Ecenter = Ecenter / np.sqrt(normfac)
    
    # Determine smoothing window
    wavelength_nm = round(p["lambda"] * 1e3)
    smoothwindow = 200 if wavelength_nm in [397, 405, 854, 422, 729, 866, 375] else 0
    if smoothwindow == 0:
        print('NO SMOOTHING DONE -- need to add wavelength to condition list')
    print(f"Smoothing window: {(smoothwindow / nPts) * (XwgCenter[-1] - XwgCenter[0])} um")
    
    # Initialize arrays
    alpha_ideal = np.ones(nPts)      # Ideal alpha
    alpha_limited = np.ones(nPts)    # Actual alpha (limited by fabrication constraints)
    maxAlphas = np.ones(nPts)        # Maximum possible alpha at each point
    minAlphas = np.zeros(nPts)       # Minimum possible alpha at each point
    desDCxx = np.zeros(nPts)         # Desired duty cycle
    desperxx = np.zeros(nPts)        # Desired period
    DCzeros = np.zeros(nPts)         # Flags for regions with DC=0
    
    # For every point along the center line, find the set of (period, DC) pairs
    # that gives the theta we want, then find the alpha closest to the target
    for k in range(nPts):
        # Calculate ideal alpha based on coupling equation
        alpha_ideal[k] = 0.5 * nu * Ecenter[k]**2 / (1 - nu * trapezoid(dxwg[:k+1] * Ecenter[:k+1]**2))
        alpha_ideal[k] = min(alpha_ideal[k], 1)  # Cap at 1
        #print(alpha_ideal[k], nu, nu*Ecenter[k]**2, 1 - nu * trapezoid(dxwg[:k+1] * Ecenter[:k+1]**2))
        
        # Get angle at current point
        theta_current_point = theta0[k] * 180 / np.pi
        
        # Calculate available alphas for this angle
        alphasAllowed, desper, desDC = helper_functions.calcAlphasAvail(
            theta_current_point, gp, dc, LUTthet, LUTalph, False)
        #print(alphasAllowed)
        
        # Handle case with no valid alphas
        if len(alphasAllowed) <= 2:
            alpha_limited[k], desperxx[k], desDCxx[k] = 0, np.nan, 0
            DCzeros[k] = 1
            maxAlphas[k], minAlphas[k] = np.nan, np.nan
        else:
            # Find min/max alphas and the closest to ideal
            maxAlphas[k], minAlphas[k] = np.nanmax(alphasAllowed), np.nanmin(alphasAllowed)
            alpha_limited[k] = alphasAllowed[np.nanargmin(np.abs(alphasAllowed - alpha_ideal[k]))]
            #alpha_limited[k] = np.nanmin(abs(alphasAllowed - alpha_ideal[k]))

            #print("hi", alpha_limited[k], min(alphasAllowed), np.nanmin(abs(alphasAllowed - alpha_ideal[k])))
            idx = np.argmin(np.abs(alphasAllowed - alpha_limited[k]))
            desperxx[k], desDCxx[k] = desper[idx], desDC[idx]
        #print(alpha_limited[k], alpha_ideal[k], desperxx[k], desDCxx[k])
    
    # Apply smoothing if required
    if smoothwindow > 0:
        desDCxx_nans = desDCxx.copy()
        desDCxx_nans[np.array(DCzeros, dtype=bool)] = np.nan
        desperxx_nans = desperxx.copy()
        desperxx_nans[np.array(DCzeros, dtype=bool)] = np.nan
        
        # Define smoothing function with NaN handling
        def smoothdata(data, window, method='savgol'):
            isnan = np.isnan(data)
            data_clean = np.copy(data)
            if np.sum(~isnan) >= window:
                if method == 'savgol':
                    win_length = window if window % 2 else window+1
                    data_clean[~isnan] = savgol_filter(data[~isnan], window_length=win_length, polyorder=2)
            return data_clean
        
        desDCxx = smoothdata(desDCxx_nans, smoothwindow)
        desperxx = smoothdata(desperxx_nans, smoothwindow)
    
    # Find the initial min DC spot to cut off first bit set by fab limits
    minAlph = np.min(alpha_limited[:nPts//2])
    idxMinAlph = np.argmin(alpha_limited[:nPts//2])
    minDC = desDCxx[idxMinAlph]
    
    # Initialize E-field arrays
    E2 = np.zeros(nPts)
    E2ideal = np.zeros(nPts)
    
    # Apply clipping and calculate resulting fields
    for k in range(nPts):
        first_12 = k < (nPts * 1 // 2)
        last_34 = k > (nPts * 3 // 4)
        below_min_alpha = alpha_ideal[k] < minAlphas[k]
        
        # Define clipping regions
        before_clip_factor = XwgCenter[k] < XwgWaists[2] - abs(XwgWaists[2] - XwgWaists[0]) * clipBeginning
        after_clip_factor = XwgCenter[k] > XwgWaists[2] + abs(XwgWaists[1] - XwgWaists[2]) * clipEnd
        
        # Apply clipping
        if hard_clip and before_clip_factor:
            alpha_limited[k] = 0
            desDCxx[k] = 0
            DCzeros[k] = 1
        elif first_12 and below_min_alpha and before_clip_factor:
            alpha_limited[k] = 0
            desDCxx[k] = 0
            DCzeros[k] = 1
        
        if after_clip_factor:
            alpha_limited[k] = 0
            desDCxx[k] = 0
            DCzeros[k] = 1
        
        # Calculate resulting field intensities
        E2[k] = 2 * alpha_limited[k] * np.exp(-2 * trapezoid(dxwg[:k+1] * alpha_limited[:k+1]))
        E2ideal[k] = 2 * alpha_ideal[k] * np.exp(-2 * trapezoid(dxwg[:k+1] * alpha_ideal[:k+1]))
    
    # Set last point to zero
    alpha_limited[-1] = 0
    desDCxx[-1] = 0
    DCzeros[-1] = 1
    
    # Calculate coupling efficiency
    nuLim = abs(trapezoid(dxwg * np.sqrt(E2) * Ecenter))**2
    
    # Calculate full fields
    EfieldFull = EfieldAmp * np.exp(-1j * phi_fs)
    EfieldFullNormed = Ecenter[1:] * np.exp(-1j * phi_fs[int(nPts//2), 1:])
    E2full = Ecenter**2
    E2lim = E2[1:]
    EfieldAmpLims = np.sqrt(E2lim)
    EfieldFullLims = EfieldAmpLims * np.exp(-1j * phi_fs[int(nPts//2), 1:])
    
    # Calculate weighted mean position
    normFacMean = sum(E2lim)
    xmean = sum(XwgCenter[1:] * (E2lim / normFacMean))
    #print(alpha_limited)
    #print(alpha_ideal)
    #print(minAlphas)
    #print(maxAlphas)
    #print(nuLim)

    #plt.figure()
    #plt.plot(XwgCenter[1:], E2lim)
    #plt.figure()
    #plt.plot(XwgCenter[1:], E2ideal[1:])
    #plt.show()
    
    # Return results dictionary
    return {
        "XwgCenter": XwgCenter,
        "alpha_ideal": alpha_ideal,
        "alpha_limited": alpha_limited,
        "maxAlphas": maxAlphas,
        "minAlphas": minAlphas,
        "desDCxx": desDCxx,
        "desperxx": desperxx,
        "E2": E2,
        "E2ideal": E2ideal,
        "E2full": E2full,
        "E2lim": E2lim,
        "EfieldFullNormed": EfieldFullNormed,
        "EfieldFullLims": EfieldFullLims,
        "nuLim": nuLim,
        "xmean": xmean,
        "theta0": theta0,
        "XwgWaists": XwgWaists
    }


def plot_design_results(p: Dict[str, Any], design: Dict[str, Any]) -> List[plt.Figure]:
    """
    Create plots of the longitudinal grating design results.
    
    Args:
        p: Parameters dictionary
        design: Design results from create_longitudinal_design
        
    Returns:
        List of figure handles
    """
    # Extract design parameters
    XwgCenter = design["XwgCenter"]
    alpha_ideal = design["alpha_ideal"]
    alpha_limited = design["alpha_limited"]
    maxAlphas = design["maxAlphas"]
    minAlphas = design["minAlphas"]
    theta0 = design["theta0"]
    E2full = design["E2full"]
    E2lim = design["E2lim"]
    XwgWaists = design["XwgWaists"]
    xmean = design["xmean"]
    desDCxx = design["desDCxx"]
    desperxx = design["desperxx"]
    EfieldFullNormed = design["EfieldFullNormed"]
    EfieldFullLims = design["EfieldFullLims"]
    nu = p.get("nu", 1.0)
    nuLim = design["nuLim"]
    minSize = p.get("minSize", 0.075)
    
    # Create figures list
    figures = []
    
    # Figure 1: Alpha values
    fig_alphas_2 = plt.figure(figsize=(10, 6))
    ax = fig_alphas_2.add_subplot(111)
    
    ax.plot(XwgCenter, alpha_ideal, 'k--', label='Ideal Alpha')
    ax.plot(XwgCenter, alpha_limited, 'b-', label='Limited Alpha')
    ax.plot(XwgCenter, maxAlphas, 'r:', label='Max Alpha')
    ax.plot(XwgCenter, minAlphas, 'g:', label='Min Alpha')
    
    for waist in XwgWaists:
        ax.axvline(x=waist, color='k', linestyle=':')
    
    ax.set_xlabel('X Position (μm)')
    ax.set_ylabel('Alpha (μm⁻¹)')
    ax.set_title('Alpha Values')
    ax.legend()
    ax.grid(True)
    
    figures.append(fig_alphas_2)
    
    # Figure 2: Resulting intensity
    fig_resulting_intensity = plt.figure(figsize=(10, 6))
    ax = fig_resulting_intensity.add_subplot(111)
    
    ax.plot(XwgCenter, E2full, 'k--', label='Original Intensity')
    ax.plot(XwgCenter[1:], E2lim, 'b-', label='Limited Intensity')
    
    for waist in XwgWaists:
        ax.axvline(x=waist, color='k', linestyle=':')
    
    ax.axvline(x=xmean, color='r', linestyle='--')
    
    ax.set_xlabel('X Position (μm)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Beam Intensity')
    ax.legend()
    ax.grid(True)
    
    figures.append(fig_resulting_intensity)
    
    # Figure 3: Theta angle plot
    fig_theta = plt.figure(figsize=(10, 6))
    ax = fig_theta.add_subplot(111)
    
    # Convert theta from radians to degrees and plot (skip first point like MATLAB)
    ax.plot(XwgCenter[1:], np.degrees(theta0[1:]), 'b-', label='Theta Angle')
    
    for waist in XwgWaists:
        ax.axvline(x=waist, color='r', linestyle=':')
    
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('θ(x) (degrees)')
    ax.set_title('Theta Angle')
    ax.grid(True)
    ax.legend()
    
    figures.append(fig_theta)
    
    # Figure 4: Real E-field plot
    fig_efield = plt.figure(figsize=(10, 6))
    ax = fig_efield.add_subplot(111)
    
    # Plot real part of E-field (skip first point like MATLAB)
    ax.plot(XwgCenter[1:], np.real(EfieldFullNormed), 'b-', label='Normalized E-field')
    ax.plot(XwgCenter[1:], np.real(EfieldFullLims), 'r-', label='Limited E-field')
    
    for waist in XwgWaists:
        ax.axvline(x=waist, color='r', linestyle=':')
    
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Desired field')
    ax.set_title('Re(E)')
    ax.grid(True)
    ax.legend()
    
    figures.append(fig_efield)
    
    # Figure 5: Combined plot with all 5 subplots (like MATLAB tiledlayout)
    # This matches the MATLAB: figAlphas = figure("Name", "Alphas, thetas, E-field on centerline", 'Position', [300 500 1500 500])
    fig_alphas = plt.figure("Alphas, thetas, E-field on centerline", figsize=(20, 8))  # Wider to accommodate 5 subplots
    axs = [fig_alphas.add_subplot(1, 5, i+1) for i in range(5)]
    
    # Subplot 1: Alpha values (matching MATLAB nexttile 1)
    axs[0].plot(XwgCenter[1:], alpha_ideal[1:], 'k--', label='αs')
    axs[0].plot(XwgCenter[1:], alpha_limited[1:], color=[1, 0.35, 0], linewidth=1, label='αsLim')
    axs[0].plot(XwgCenter, maxAlphas, 'k:', linewidth=0.75, label='Max α')
    axs[0].plot(XwgCenter, minAlphas, 'k:', linewidth=0.75, label='Min α')
    
    for waist in XwgWaists:
        axs[0].axvline(x=waist, color='r', linestyle=':')
    
    axs[0].set_xlabel('Position (μm)')
    axs[0].set_ylabel('α (μm⁻¹)')
    axs[0].set_title(f'MinSize = {minSize*1e3:.0f}nm\nη = {nu:.3f}')
    axs[0].grid(True)
    
    # Subplot 2: Theta angle (matching MATLAB nexttile 2)
    axs[1].plot(XwgCenter[1:], np.degrees(theta0[1:]), 'b-')
    
    for waist in XwgWaists:
        axs[1].axvline(x=waist, color='r', linestyle=':')
    
    axs[1].set_xlabel('Position (μm)')
    axs[1].set_ylabel('θ(x)')
    axs[1].grid(True)
    
    # Subplot 3: Real E-field (matching MATLAB nexttile 3)
    axs[2].plot(XwgCenter[1:], np.real(EfieldFullNormed), 'b-', label='Normalized')
    axs[2].plot(XwgCenter[1:], np.real(EfieldFullLims), 'r-', label='Limited')
    
    for waist in XwgWaists:
        axs[2].axvline(x=waist, color='r', linestyle=':')
    
    axs[2].set_xlabel('Position (μm)')
    axs[2].set_ylabel('Desired field')
    axs[2].set_title('Re(E)')
    axs[2].grid(True)
    
    # Subplot 4: |E|^2 intensity (matching MATLAB nexttile 4)
    axs[3].plot(XwgCenter, E2full, 'b-', label='E2full')
    axs[3].plot(XwgCenter[1:], E2lim, 'r-', label='E2lim')
    axs[3].axvline(x=xmean, color='k', linestyle=':', linewidth=1)
    
    for waist in XwgWaists:
        axs[3].axvline(x=waist, color='r', linestyle=':')
    
    axs[3].set_xlabel('Position (μm)')
    axs[3].set_ylabel('Desired field')
    axs[3].set_title(f'|E|²\nη_limits = {nuLim:.3f}')
    axs[3].grid(True)
    
    # Subplot 5: Grating parameters with dual y-axis (matching MATLAB nexttile 5)
    ax5_left = axs[4]
    ax5_right = ax5_left.twinx()
    
    # Left y-axis: Duty cycle
    line1 = ax5_left.plot(XwgCenter[1:], desDCxx[1:], 'b-', label='Duty Cycle')
    ax5_left.set_ylabel('Design pert DC', color='b')
    ax5_left.tick_params(axis='y', labelcolor='b')
    
    # Right y-axis: Period (convert to nm like MATLAB)
    line2 = ax5_right.plot(XwgCenter[1:], desperxx[1:] * 1000, 'r-', label='Period (nm)')  # Convert μm to nm
    ax5_right.set_ylabel('Design period (nm)', color='r')
    ax5_right.tick_params(axis='y', labelcolor='r')
    
    for waist in XwgWaists:
        ax5_left.axvline(x=waist, color='r', linestyle=':')
    
    ax5_left.set_xlabel('Position (μm)')
    ax5_left.grid(True)
    
    plt.tight_layout()
    figures.append(fig_alphas)
    
    return figures


def plot_grating_design_analysis(p: Dict[str, Any], design: Dict[str, Any], 
                                gp: Array, dc: Array, LUTalph: Array, 
                                LUTthet: Array, LUTnEffs: Array,
                                nOx: float = 1.45) -> List[plt.Figure]:
    """
    Create comprehensive grating design analysis plots matching MATLAB implementation.
    
    Args:
        p: Parameters dictionary
        design: Design results from create_longitudinal_design
        gp: Grating periods array
        dc: Duty cycles array  
        LUTalph: Alpha lookup table
        LUTthet: Theta lookup table
        LUTnEffs: Effective index lookup table
        nOx: Oxide refractive index
        
    Returns:
        List of figure handles
    """
    # Extract design parameters
    XwgCenter = design["XwgCenter"]
    alphas = design["alpha_ideal"]
    alphasLim = design["alpha_limited"]
    maxAlphas = design["maxAlphas"]
    minAlphas = design["minAlphas"]
    theta0 = design["theta0"]
    XwgWaists = design["XwgWaists"]
    desDCxx = design["desDCxx"]
    desperxx = design["desperxx"]
    DCzeros = design.get("DCzeros", np.zeros(len(XwgCenter)))
    
    fitorder = 7
    
    # x range along grating (skip first point like MATLAB)
    xgrat = XwgCenter[1:]
    
    # Design parameters along grating
    desthetas = theta0[1:]
    desalpha = alphasLim[1:] if len(alphasLim) > len(xgrat) else alphasLim
    desDCx = desDCxx[1:]
    desperx = desperxx[1:]
    DCzerosclip = DCzeros[1:] if len(DCzeros) > 1 else np.zeros(len(xgrat))
    
    # Remove clipped regions
    clipIndices = DCzerosclip.astype(bool)
    if np.any(clipIndices):
        xgrat = xgrat[~clipIndices]
        desDCx = desDCx[~clipIndices]
        desperx = desperx[~clipIndices]
    
    # Calculate indices for LUT lookup
    gratpersIdx = np.round((desperx - gp.min()) / (gp[1] - gp[0])).astype(int)
    pertDCsIdx = np.round((desDCx - dc.min()) / (dc[1] - dc[0])).astype(int)
    
    # Handle out-of-bounds indices
    gratpersIdx = np.clip(gratpersIdx, 0, len(gp) - 1)
    pertDCsIdx = np.clip(pertDCsIdx, 0, len(dc) - 1)
    
    # Extract values from LUT
    nEffsFromFit = np.zeros(len(xgrat))
    alphasFromFit = np.zeros(len(xgrat))
    thetasFromFit = np.zeros(len(xgrat))
    
    for idx in range(len(xgrat)):
        dcIdx = pertDCsIdx[idx]
        perIdx = gratpersIdx[idx]
        alphasFromFit[idx] = LUTalph[dcIdx, perIdx]
        thetasFromFit[idx] = LUTthet[dcIdx, perIdx]
        nEffsFromFit[idx] = max(LUTnEffs[dcIdx, perIdx], nOx)
    
    nEffAvg = np.mean(nEffsFromFit)
    
    # Polynomial fits
    pDC = np.polyfit(xgrat, desDCx, fitorder)
    pper = np.polyfit(xgrat, desperx, fitorder)
    pNeff = np.polyfit(xgrat, nEffsFromFit, fitorder)
    
    # Print coefficients (matching MATLAB output)
    print(f"pper: {pper}")
    print(f"pDC: {pDC}")
    
    figures = []
    plot_xlim = [XwgCenter[0], XwgCenter[-1]]
    
    # Figure 1: Design DC, period (6 subplots)
    fig_design_params = plt.figure("Design DC, period", figsize=(18, 10))
    
    # Subplot 1: Duty Cycle
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(xgrat, desDCx, 'k-', linewidth=3, label='Design DC')
    plt.plot(xgrat, dc[pertDCsIdx], 'b-', label='LUT DC')
    plt.plot(xgrat, np.polyval(pDC, xgrat), 'r-', label=f'Fit (order {fitorder})')
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('Position (μm)')
    plt.ylabel('Design perturbation DC')
    plt.xlim(plot_xlim)
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Period
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(xgrat, desperx * 1e3, 'k-', linewidth=3, label='Design Period')
    plt.plot(xgrat, gp[gratpersIdx] * 1e3, 'b-', label='LUT Period')
    plt.plot(xgrat, 1e3 * np.polyval(pper, xgrat), 'r-', label=f'Fit (order {fitorder})')
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('Position (μm)')
    plt.ylabel('Design period (nm)')
    plt.xlim(plot_xlim)
    plt.grid(True)
    plt.legend()
    
    # Subplot 3: Effective Index
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(xgrat, nEffsFromFit, 'k-', linewidth=3, label='n_eff from LUT')
    plt.plot(xgrat, np.ones(len(xgrat)) * nEffAvg, 'b-', label=f'Average: {nEffAvg:.3f}')
    plt.plot(xgrat, np.polyval(pNeff, xgrat), 'r-', label=f'Fit (order {fitorder})')
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('Position (μm)')
    plt.ylabel('n_eff')
    plt.xlim(plot_xlim)
    plt.grid(True)
    plt.legend()
    
    # Subplot 4: Alpha comparison
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(XwgCenter[1:], alphasLim[:len(XwgCenter)-1] if len(alphasLim) >= len(XwgCenter)-1 else alphasLim, 
             'k-', linewidth=2, label='Design α')
    plt.plot(xgrat, alphasFromFit, 'b-', label='LUT α')
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('Position (μm)')
    plt.ylabel('α (μm⁻¹)')
    plt.grid(True)
    plt.legend()
    
    # Subplot 5: Theta comparison
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(XwgCenter[1:], theta0[1:] * 180/np.pi, 'k-', linewidth=2, label='Design θ')
    plt.plot(xgrat, thetasFromFit, 'b-', label='LUT θ')
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('Position (μm)')
    plt.ylabel('θ(x) (degrees)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    figures.append(fig_design_params)
    
    # Figure 2: Surface plots with design overlay
    fig_surf = plt.figure("Alphas, thetas on contour", figsize=(17, 5))
    
    # Alpha contour
    ax1 = plt.subplot(1, 3, 1)
    cs1 = plt.contourf(gp * 1e3, dc, LUTalph, 300, cmap='jet')
    plt.scatter(gp[gratpersIdx] * 1e3, dc[pertDCsIdx], c='red', s=20, marker='o')
    plt.xlabel('Grating period (nm)')
    plt.ylabel('Perturbation DC')
    plt.title('α (μm⁻¹)')
    
    if p.get('material', '') == 'AlOx':
        plt.clim([0, 0.06])
    
    plt.colorbar(cs1)
    
    # Theta contour  
    ax2 = plt.subplot(1, 3, 2)
    cs2 = plt.contourf(gp * 1e3, dc, LUTthet, 300, cmap='jet')
    plt.scatter(gp[gratpersIdx] * 1e3, dc[pertDCsIdx], c='red', s=20, marker='o')
    plt.xlabel('Grating period (nm)')
    plt.ylabel('Perturbation DC')
    plt.title('θ (deg)')
    plt.colorbar(cs2)
    
    # n_eff contour
    ax3 = plt.subplot(1, 3, 3)
    cs3 = plt.contourf(gp * 1e3, dc, LUTnEffs, 30, cmap='jet')
    plt.scatter(gp[gratpersIdx] * 1e3, dc[pertDCsIdx], c='red', s=20, marker='o')
    plt.xlabel('Grating period (nm)')
    plt.ylabel('Perturbation DC')
    plt.title('n_eff')
    plt.colorbar(cs3)
    
    plt.tight_layout()
    figures.append(fig_surf)
    
    # Figure 3: Design params for paper (4 subplots, 2x2)
    fig_paper = plt.figure("DesignParams for paper", figsize=(12, 10))
    
    # Subplot 1: Theta (top left)
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(XwgCenter[1:], theta0[1:] * 180/np.pi, linewidth=1.3)
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('x (μm)')
    plt.ylabel('θ (deg)')
    plt.xlim(plot_xlim)
    plt.grid(True)
    ax1.set_aspect('equal')
    
    # Subplot 2: Alpha (top right) 
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(XwgCenter[1:], alphas[1:] if len(alphas) > len(XwgCenter)-1 else alphas, 'k--', label='Ideal')
    plt.plot(XwgCenter[1:], alphasLim[:len(XwgCenter)-1] if len(alphasLim) >= len(XwgCenter)-1 else alphasLim, 
             color=[1, 0.35, 0], linewidth=1.3, label='Limited')
    plt.plot(XwgCenter, maxAlphas, 'k:', linewidth=0.75, label='Max')
    plt.plot(XwgCenter, minAlphas, 'k:', linewidth=0.75, label='Min')
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('x (μm)')
    plt.ylabel('α (μm⁻¹)')
    plt.xlim(plot_xlim)
    plt.ylim([0, 0.2])
    plt.grid(True)
    ax2.set_aspect('equal')
    
    # Subplot 3: DC and Period (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    ax3_right = ax3.twinx()
    
    line1 = ax3.plot(xgrat, desDCx, linewidth=1.3, color='blue', label='Design DC')
    line2 = ax3_right.plot(xgrat, desperx * 1e3, linewidth=1.3, color='red', label='Design Period (nm)')
    
    for waist in XwgWaists:
        ax3.axvline(x=waist, color='r', linestyle=':')
    
    ax3.set_xlabel('x (μm)')
    ax3.set_ylabel('Design DC', color='blue')
    ax3_right.set_ylabel('Design period (nm)', color='red')
    ax3.set_xlim(plot_xlim)
    ax3.grid(True)
    ax3.set_aspect('equal')
    
    # Subplot 4: n_eff (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    # Apply smoothing like MATLAB smoothdata
    from scipy.ndimage import uniform_filter1d
    nEffsSmooth = uniform_filter1d(nEffsFromFit, size=5)
    plt.plot(xgrat, nEffsSmooth, 'k-', linewidth=1.3)
    
    for waist in XwgWaists:
        plt.axvline(x=waist, color='r', linestyle=':')
    
    plt.xlabel('x (μm)')
    plt.ylabel('n_eff')
    plt.xlim(plot_xlim)
    plt.grid(True)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    figures.append(fig_paper)
    
    return figures 