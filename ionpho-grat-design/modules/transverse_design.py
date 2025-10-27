"""
Transverse grating design module

This module handles the transverse design of gratings, including
phase extraction, contour fitting, and curved grating teeth creation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Any, Optional, Union

# Type aliases
Array = np.ndarray

def parab(ydata, y0, R_inv):
    """
    Parabola function for curve fitting.
    
    Args:
        ydata: Y coordinates
        y0: X offset
        R_inv: Inverse radius of curvature
        
    Returns:
        Parabola values
    """
    ydata = np.asarray(ydata)
    R = -R_inv  # Match the sign inversion in original code
    return y0 + ydata**2 / (2 * R)

def quartic(ydata, y0, R_inv, A_inv):
    """
    Quartic function for curve fitting.
    
    Args:
        ydata: Y coordinates
        y0: X offset
        R_inv: Inverse radius of curvature
        A_inv: Quartic coefficient
        
    Returns:
        Quartic function values
    """
    ydata = np.asarray(ydata)
    R = -R_inv  # Match original sign convention
    A = -A_inv
    return y0 + ydata**2 / (2 * R) + A * ydata**4

def calculate_waveguide_phase(Xwg: Array, Ywg: Array, tapStart: float, 
                             nWG: float, k0: float, 
                             planar_wavefronts: bool = False) -> Array:
    """
    Calculate the phase of the waveguide.
    
    Args:
        Xwg: X coordinates of the waveguide
        Ywg: Y coordinates of the waveguide
        tapStart: Starting position of the taper
        nWG: Waveguide effective index
        k0: Wavenumber
        planar_wavefronts: Whether to use planar wavefronts
        
    Returns:
        Waveguide phase array
    """
    # Calculate distance to taper start
    if planar_wavefronts:
        distToTaperStart = Xwg - tapStart
    else:
        distToTaperStart = np.sqrt((Xwg - tapStart) ** 2 + Ywg ** 2)
    
    # Calculate phase
    phi_wg_basic = k0 * nWG * distToTaperStart
    
    return phi_wg_basic

def calculate_advanced_waveguide_phase(Xwg: Array, Ywg: Array, 
                                    xgrat: Array, nEffsFromFit: Array,
                                    tapStart: float, nWG: float, k0: float,
                                    planar_wavefronts: bool = False) -> Array:
    """
    Calculate advanced waveguide phase using local effective indices.
    
    Args:
        Xwg: X coordinates of the waveguide
        Ywg: Y coordinates of the waveguide
        xgrat: X coordinates of the grating
        nEffsFromFit: Effective indices from LUT
        tapStart: Starting position of the taper
        nWG: Waveguide effective index
        k0: Wavenumber
        planar_wavefronts: Whether to use planar wavefronts
        
    Returns:
        Advanced waveguide phase array
    """
    # Calculate distance to taper start
    if planar_wavefronts:
        distToTaperStart = Xwg - tapStart
    else:
        distToTaperStart = np.sqrt((Xwg - tapStart) ** 2 + Ywg ** 2)
    
    # Grating and taper parameters
    taperLength = xgrat[0] - tapStart  # Distance between tapStart and x1
    gratingLength = xgrat[-1] - xgrat[0]  # Distance between x1 and x2
    totGCLength = taperLength + gratingLength  # Total length of grating coupler
    diff_xgrat_tapStart = xgrat - tapStart
    
    # Create interpolation function for effective indices
    nEffInterpFunc = interp1d(xgrat, nEffsFromFit, kind='linear', fill_value='extrapolate')
    nEffsInterp = nEffInterpFunc(xgrat)
    
    # Compute optical path length
    opticalPathLength = np.zeros_like(distToTaperStart)
    dxgrat = np.diff(xgrat, prepend=xgrat[0])
    
    for ii in range(distToTaperStart.size):
        if distToTaperStart.flat[ii] < taperLength:
            # If within taper region, simply use nWG
            opticalPathLength.flat[ii] = nWG * distToTaperStart.flat[ii]
        else:
            # Outside of taper region, may be in grating region or further out
            dIdx = np.where(diff_xgrat_tapStart <= distToTaperStart.flat[ii])[0]
            if dIdx.size > 0:
                last_idx = dIdx[-1]
                # Integrate n_eff along the path
                opticalPathInGrating = np.trapz(nEffsInterp[:last_idx + 1], dx=np.mean(dxgrat[:last_idx + 1]))
                opticalPathLength.flat[ii] = nWG * taperLength + opticalPathInGrating
                
                # For points beyond the grating region
                if distToTaperStart.flat[ii] > totGCLength:
                    opticalPathLength.flat[ii] += nEffsInterp[-1] * (distToTaperStart.flat[ii] - totGCLength)
    
    # Calculate phase
    phi_wg_full = k0 * opticalPathLength
    
    return phi_wg_full

def extract_phase_contours(Xwg: Array, Ywg: Array, phi_tot: Array, ytap: Array) -> Tuple[List, List, int]:
    """
    Extract phase contours from the total phase.
    
    Args:
        Xwg: X coordinates of the waveguide
        Ywg: Y coordinates of the waveguide
        phi_tot: Total phase (waveguide + free space)
        ytap: Taper height at each position
        
    Returns:
        Tuple containing contour cells, contour points, and number of contours
    """
    # Preprocessing
    phi_tot2 = phi_tot - np.nanmin(phi_tot)
    phi_totMask = np.copy(phi_tot2)
    phi_totMask[np.abs(Ywg) > ytap] = np.nan  # Mask the region outside of taper
    
    # Number of contours and contour levels
    nconts = int(np.floor(np.nanmax(phi_totMask) / (2 * np.pi)) - 3)
    contvs = np.arange(nconts) * 2 * np.pi
    
    # Extract contours
    fig_temp, ax_temp = plt.subplots()
    C = ax_temp.contour(Xwg, Ywg, phi_totMask, levels=contvs)
    plt.close(fig_temp)  # Close the temporary figure
    
    # Extract contour data
    ContCell = []
    pCont = []
    
    for i, collection in enumerate(C.collections):
        for path in collection.get_paths():
            vertices = path.vertices
            x = vertices[:, 0]
            y = vertices[:, 1]
            ContCell.append(np.vstack((x, y)))
            pCont.append(i)
    
    return ContCell, pCont, nconts

def fit_contours(ContCell: List, nconts: int, xgrat: Array) -> Dict[str, Any]:
    """
    Fit parabolas and quartic curves to the phase contours.
    
    Args:
        ContCell: List of contour cells
        nconts: Number of contours
        xgrat: X coordinates of the grating
        
    Returns:
        Dictionary with fitting parameters
    """
    # Initialize arrays for fitting results
    rads = np.zeros(nconts-1)
    x0s = np.zeros(nconts-1)
    goodIndices = np.zeros(nconts-1)
    residual_sums = np.zeros(nconts-1)
    residual_sumsQuart = np.zeros(nconts-1)
    rads_quart = np.zeros(nconts-1)
    x0s_quart = np.zeros(nconts-1)
    coeffsA_quart = np.zeros(nconts-1)
    quart_minus_parab_top = np.zeros(nconts-1)
    
    # Fit loop
    for k in range(nconts - 1):
        curr_contour = ContCell[k]
        curr_contour_x = curr_contour[0, :]
        curr_contour_y = curr_contour[1, :]
        
        try:
            # Fit parabola
            xp, _ = curve_fit(parab, curr_contour_y, curr_contour_x, p0=[xgrat[0], 1], maxfev=5000)
            # Fit quartic
            xp_quart, _ = curve_fit(quartic, curr_contour_y, curr_contour_x, p0=[xgrat[0], 1, 0], maxfev=5000)
            
            # Calculate fits
            parab_xfit = parab(curr_contour_y, *xp)
            quartic_xfit = quartic(curr_contour_y, *xp_quart)
            
            # Store parameters
            x0s[k] = xp[0]
            rads[k] = -xp[1]
            x0s_quart[k] = xp_quart[0]
            rads_quart[k] = -xp_quart[1]
            coeffsA_quart[k] = -xp_quart[2]
            
            # Calculate residuals
            residual_sums[k] = np.sum((curr_contour_x - parab_xfit) ** 2)
            residual_sumsQuart[k] = np.sum((curr_contour_x - quartic_xfit) ** 2)
            
            # Calculate difference at the top
            quart_minus_parab_top[k] = quartic(np.array([curr_contour_y[-1]]), *xp_quart)[0] - parab(np.array([curr_contour_y[-1]]), *xp)[0]
            
            # Mark as a good fit if it meets criteria
            if (np.min(np.abs(curr_contour_y)) < 0.02 and 
                np.max(np.abs(curr_contour_y)) > 0.1 and  # Approximate check for covering taper
                np.max(curr_contour_x) < xgrat[-1] and 
                np.max(curr_contour_x) >= xgrat[0]):
                goodIndices[k] = 1
            
        except Exception as e:
            print(f"Fit failed at k={k}: {e}")
            continue
    
    # Find the first good index and exclude it from consideration
    idx_first = np.argmax(goodIndices)
    goodIndices[idx_first] = False
    
    # Convert to boolean mask
    goodIndices = goodIndices.astype(bool)
    
    # Extract good values
    x0sg = x0s[goodIndices]
    radsg = rads[goodIndices]
    x0sg_q = x0s_quart[goodIndices]
    radsg_q = rads_quart[goodIndices]
    coeffsA_q = coeffsA_quart[goodIndices]
    
    # Perform linear fit to get r0 and dr
    X2g = np.column_stack((np.ones_like(x0sg), x0sg))
    bg = np.linalg.lstsq(X2g, radsg, rcond=None)[0]
    r0 = -bg[0]
    dr = -bg[1]
    
    # Return fitting results
    return {
        "r0": r0,
        "dr": dr,
        "x0sg": x0sg,
        "radsg": radsg,
        "x0sg_q": x0sg_q,
        "radsg_q": radsg_q,
        "coeffsA_q": coeffsA_q,
        "goodIndices": goodIndices,
        "bg": bg
    }

def plot_phase_contours(Xwg: Array, Ywg: Array, phi_tot_wrapped: Array, 
                       ContCell: List, nconts: int, fit_results: Dict[str, Any],
                       ytap: Array, waistCxdata: Array, waistCydata: Array,
                       tapStart: float = 0) -> plt.Figure:
    """
    Plot phase contours with fits.
    
    Args:
        Xwg: X coordinates of the waveguide
        Ywg: Y coordinates of the waveguide
        phi_tot_wrapped: Wrapped total phase
        ContCell: List of contour cells
        nconts: Number of contours
        fit_results: Dictionary with fitting results
        ytap: Taper height at each position
        waistCxdata: X coordinates of waist contour
        waistCydata: Y coordinates of waist contour
        
    Returns:
        Figure with phase contours and fits
    """
    # Unpack fitting results
    x0sg = fit_results["x0sg"]
    radsg = fit_results["radsg"]
    bg = fit_results["bg"]
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot phase contours - shift coordinates so taper start is at x=0
    c = axs[0].contourf(Xwg - tapStart, Ywg, phi_tot_wrapped, levels=100, cmap='viridis')
    axs[0].contour(Xwg - tapStart, Ywg, phi_tot_wrapped, levels=np.arange(nconts) * 2 * np.pi, colors='k', linewidths=0.5)
    axs[0].plot(waistCxdata - tapStart, waistCydata, 'r-', linewidth=1.5)
    fig.colorbar(c, ax=axs[0])
    axs[0].set_title('Total Phase (wrapped)')
    axs[0].set_xlabel('x coord relative to taper start (μm)')
    axs[0].set_ylabel('y coord (μm)')
    
    # Plot contour fits
    draw_each = 5  # Plot every 5th contour
    for k in range(0, nconts-1, draw_each):
        if k < len(ContCell):
            curr_contour = ContCell[k]
            curr_contour_x = curr_contour[0, :]
            curr_contour_y = curr_contour[1, :]
            
            try:
                xp, _ = curve_fit(parab, curr_contour_y, curr_contour_x, p0=[x0sg[0], 1], maxfev=5000)
                xp_quart, _ = curve_fit(quartic, curr_contour_y, curr_contour_x, p0=[x0sg[0], 1, 0], maxfev=5000)
                
                parab_xfit = parab(curr_contour_y, *xp)
                quartic_xfit = quartic(curr_contour_y, *xp_quart)
                
                axs[1].plot(curr_contour_x - tapStart, curr_contour_y, 'k.', alpha=0.3)
                axs[1].plot(parab_xfit - tapStart, curr_contour_y, color='tab:blue', linestyle='--', linewidth=1.2)
                axs[1].plot(quartic_xfit - tapStart, curr_contour_y, color='tab:orange', linestyle='--', linewidth=1.2)
            except Exception:
                continue
    
    axs[1].set_title('Contour Fits')
    axs[1].set_xlabel('x coord relative to taper start (μm)')
    axs[1].set_ylabel('y coord (μm)')
    
    # Plot radius of curvature fit
    axs[2].scatter(x0sg - tapStart, radsg, color='tab:blue', label='Fitted R values')
    x_range = np.linspace(min(x0sg - tapStart), max(x0sg - tapStart), 100)
    axs[2].plot(x_range, -bg[0] - bg[1] * (x_range + tapStart), 'r-', 
               label=f'r0={-bg[0]:.2f}, dr={-bg[1]:.4f}')
    axs[2].set_title('Radius of Curvature')
    axs[2].set_xlabel('x coord relative to taper start (μm)')
    axs[2].set_ylabel('Radius (μm)')
    axs[2].legend()
    
    plt.tight_layout()
    return fig

def plot_contour_coefficients(x0sg_q: Array, radsg_q: Array, bg: Array, 
                             XwgWaists: List, coeffsA_q: Array, 
                             xPtsBothFull: Array, yPtsPerFull: Array,
                             yPtsDCFull: Optional[Array] = None,
                             tapStart: float = 0) -> plt.Figure:
    """
    Plot contour coefficients.
    
    Args:
        x0sg_q: X offsets from quartic fit
        radsg_q: Radii from quartic fit
        bg: Linear fit coefficients
        XwgWaists: Beam waist positions
        coeffsA_q: Quartic coefficients
        xPtsBothFull: X coordinates for period interpolation (relative to taper start)
        yPtsPerFull: Period values for interpolation
        yPtsDCFull: Optional duty cycle values for interpolation
        tapStart: Taper start position for coordinate shifting
        
    Returns:
        Figure with contour coefficients
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot radius of curvature
    ax1 = fig.add_subplot(131)
    ax1.scatter(x0sg_q - tapStart, radsg_q, 20, 'b')  # Shift coordinates
    x_range = np.linspace(min(x0sg_q - tapStart), max(x0sg_q - tapStart), 100)
    ax1.plot(x_range, -bg[0] - bg[1] * (x_range + tapStart), 'r-')  # Adjust fit line
    
    for waist in XwgWaists:
        ax1.axvline(x=waist - tapStart, color='k', linestyle=':')  # Shift waist lines
    
    ax1.set_xlabel('x coord relative to taper start (μm)')
    ax1.set_ylabel('Radius (μm)')
    ax1.set_title('Radius of Curvature')
    ax1.grid(True)
    
    # Plot quartic coefficient
    ax2 = fig.add_subplot(132)
    ax2.scatter(x0sg_q - tapStart, coeffsA_q, 20, 'b')  # Shift coordinates
    ax2.set_xlabel('x coord relative to taper start (μm)')
    ax2.set_ylabel('A coefficient (quartic term)')
    ax2.set_title('Quartic Coefficient')
    ax2.grid(True)
    
    # Plot period and duty cycle
    ax3 = fig.add_subplot(133)
    
    # Plot period on left y-axis
    line1 = ax3.plot(xPtsBothFull, yPtsPerFull, 'b-', linewidth=2, label='Period')
    ax3.set_xlabel('x coord relative to taper start (μm)')
    ax3.set_ylabel('Period (nm)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.set_title('Grating Period & Duty Cycle')
    ax3.grid(True, alpha=0.3)
    
    # Add right y-axis for duty cycle if data is provided
    if yPtsDCFull is not None:
        ax3_right = ax3.twinx()
        line2 = ax3_right.plot(xPtsBothFull, yPtsDCFull, 'r-', linewidth=2, label='Duty Cycle')
        ax3_right.set_ylabel('Duty Cycle', color='red')
        ax3_right.tick_params(axis='y', labelcolor='red')
        ax3_right.set_ylim(0, 1)  # Set DC range from 0 to 1
        
        # Add combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
    else:
        ax3.legend()
    
    plt.tight_layout()
    return fig

def create_transverse_design(Xwg: Array, Ywg: Array, phi_fs: Array, 
                           XwgCenter: Array, XwgWaists: List, YwgWaists: List,
                           nEffsFromFit: Array, tapStart: float, tapang: float,
                           k0: float, p: Dict[str, Any], 
                           planar_wavefronts: bool = False,
                           desperxx: Optional[Array] = None,
                           desDCxx: Optional[Array] = None) -> Dict[str, Any]:
    """
    Create the transverse grating design based on phase analysis.
    
    Args:
        Xwg: X coordinates of the waveguide
        Ywg: Y coordinates of the waveguide
        phi_fs: Free space phase
        XwgCenter: Centerline X coordinates
        XwgWaists: Beam waist X positions
        YwgWaists: Beam waist Y positions
        nEffsFromFit: Effective indices from LUT
        tapStart: Starting position of the taper
        tapang: Taper angle
        k0: Wavenumber
        p: Parameters dictionary
        planar_wavefronts: Whether to use planar wavefronts
        desperxx: Optional array of designed periods from longitudinal design (μm)
        desDCxx: Optional array of designed duty cycles from longitudinal design
        
    Returns:
        Dictionary with transverse design results
    """
    # Determine waveguide index based on wavelength
    wavelength_nm = round(p["lambda"] * 1e3)
    material = p.get("material", "AO_AIM")
    
    if wavelength_nm == 732:
        nWG = 1.7444
    elif wavelength_nm == 866:
        nWG = 1.6041
    elif wavelength_nm == 854:
        nWG = 1.6052
    elif wavelength_nm == 532:
        if material == 'SiN':
            nWG = 1.8383
        elif material == 'AlOx':
            nWG = 1.515
    elif wavelength_nm == 422:
        nWG = 1.5305
    elif wavelength_nm == 375:
        nWG = 1.5488
    elif wavelength_nm == 729:
        nWG = 1.6295
    else:
        nWG = 1.5395  # Default for 397nm
    
    # Calculate the average effective index
    nEffAvg = np.mean(nEffsFromFit)
    
    # Taper height calculation
    if planar_wavefronts:
        ytap = np.ones_like(XwgCenter) * (2.844 * YwgWaists[1])
    else:
        ytap = np.tan(tapang) * (XwgCenter - tapStart)
    
    # Find 1/e waist contour
    EfieldAmpNorm = np.ones_like(Xwg)  # Placeholder for normalized amplitude
    fig_temp, ax_temp = plt.subplots()
    CS = ax_temp.contour(Xwg, Ywg, EfieldAmpNorm, levels=[1/np.exp(1)])
    plt.close(fig_temp)
    
    # Extract waist contour data
    waistCxdata = []
    waistCydata = []
    for collection in CS.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            waistCxdata.extend(vertices[:, 0])
            waistCydata.extend(vertices[:, 1])
    
    waistCxdata = np.array(waistCxdata)
    waistCydata = np.array(waistCydata)
    
    # Calculate waveguide phase
    useBasicNeff = p.get("useBasicNeff", False)
    useAvgNeff = p.get("useAvgNeff", False)
    
    # Extract the grating x-coordinates
    xgrat = XwgCenter
    
    if useBasicNeff:
        # Simple phase calculation
        phi_wg = calculate_waveguide_phase(Xwg, Ywg, tapStart, nWG, k0, planar_wavefronts)
    elif useAvgNeff:
        # Phase calculation with average effective index
        phi_wg = calculate_waveguide_phase(Xwg, Ywg, tapStart, nEffAvg, k0, planar_wavefronts)
    else:
        # Advanced phase calculation with varying effective index
        phi_wg = calculate_advanced_waveguide_phase(Xwg, Ywg, xgrat, nEffsFromFit, tapStart, nWG, k0, planar_wavefronts)
    
    # Calculate total phase
    phi_tot = phi_wg + phi_fs
    phi_tot_wrapped = np.mod(phi_tot, 2 * np.pi) - 2 * np.pi
    
    # Extract phase contours
    ContCell, pCont, nconts = extract_phase_contours(Xwg, Ywg, phi_tot, ytap)
    
    # Fit contours
    fit_results = fit_contours(ContCell, nconts, xgrat)
    
    # Create plots
    fig_phase_contours = plot_phase_contours(Xwg, Ywg, phi_tot_wrapped, ContCell, nconts, 
                                           fit_results, ytap, waistCxdata, waistCydata, tapStart)
    
    # Prepare additional plotting data - shift coordinates so taper start is at x=0
    xPtsBothFull = np.linspace(np.min(xgrat), np.max(xgrat), 200) - tapStart
    
    # Use actual period data if provided, otherwise use placeholder
    if desperxx is not None:
        # Interpolate the designed periods to match the plotting grid
        from scipy.interpolate import interp1d
        # Filter out invalid periods (NaN or zero)
        valid_mask = ~(np.isnan(desperxx) | (desperxx == 0))
        if np.any(valid_mask):
            valid_x = XwgCenter[valid_mask] - tapStart  # Shift to taper start = 0
            valid_periods = desperxx[valid_mask] * 1000  # Convert μm to nm for plotting
            if len(valid_x) > 1:
                interp_func = interp1d(valid_x, valid_periods, kind='linear', 
                                     bounds_error=False, fill_value=valid_periods[0])
                yPtsPerFull = interp_func(xPtsBothFull)
                # Fill any remaining NaN values with the first valid period
                yPtsPerFull = np.where(np.isnan(yPtsPerFull), valid_periods[0], yPtsPerFull)
            else:
                yPtsPerFull = np.ones_like(xPtsBothFull) * valid_periods[0]
        else:
            yPtsPerFull = np.ones_like(xPtsBothFull) * 500  # Fallback to placeholder
    else:
        yPtsPerFull = np.ones_like(xPtsBothFull) * 500  # Placeholder when no data provided
    
    # Use actual duty cycle data if provided, otherwise use placeholder
    if desDCxx is not None:
        # Interpolate the designed duty cycles to match the plotting grid
        from scipy.interpolate import interp1d
        # Don't filter out zeros - they are valid and important!
        # Only filter out NaN values
        valid_dc_mask = ~np.isnan(desDCxx)
        if np.any(valid_dc_mask):
            valid_x_dc = XwgCenter[valid_dc_mask] - tapStart  # Shift to taper start = 0
            valid_dcs = desDCxx[valid_dc_mask]  # Duty cycles are already in correct range (0-1)
            if len(valid_x_dc) > 1:
                # Use linear interpolation but preserve zeros
                interp_func_dc = interp1d(valid_x_dc, valid_dcs, kind='linear', 
                                        bounds_error=False, fill_value=0)
                yPtsDCFull = interp_func_dc(xPtsBothFull)
                # Fill any remaining NaN values with 0 (not with a valid value)
                yPtsDCFull = np.where(np.isnan(yPtsDCFull), 0, yPtsDCFull)
            else:
                yPtsDCFull = np.ones_like(xPtsBothFull) * valid_dcs[0]
        else:
            yPtsDCFull = np.zeros_like(xPtsBothFull)  # All zeros if no valid data
    else:
        yPtsDCFull = np.ones_like(xPtsBothFull) * 0.5  # Placeholder when no data provided
    
    fig_coeff = plot_contour_coefficients(fit_results["x0sg_q"], fit_results["radsg_q"], 
                                        fit_results["bg"], XwgWaists, fit_results["coeffsA_q"],
                                        xPtsBothFull, yPtsPerFull, yPtsDCFull, tapStart)
    
    # Return design results
    return {
        "r0": fit_results["r0"],
        "dr": fit_results["dr"],
        "xPtsRadsQ": fit_results["x0sg_q"],
        "yPtsRadsQ_R": fit_results["radsg_q"],
        "yPtsRadsQ_A": fit_results["coeffsA_q"],
        "tapang": tapang,
        "nWG": nWG,
        "nEffAvg": nEffAvg,
        "phi_wg": phi_wg,
        "phi_tot": phi_tot,
        "phi_tot_wrapped": phi_tot_wrapped,
        "figures": [fig_phase_contours, fig_coeff]
    }

def plot_transverse_design(transverse_results: Dict[str, Any]) -> List[plt.Figure]:
    """
    Create plots for the transverse design.
    
    Args:
        transverse_results: Results from create_transverse_design
        
    Returns:
        List of figure handles
    """
    # The figures are already generated in create_transverse_design
    return transverse_results["figures"] 