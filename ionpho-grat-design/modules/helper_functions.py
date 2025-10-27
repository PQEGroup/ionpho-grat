"""
Helper functions for AIM Grating Design

This module contains various helper functions for optical simulations,
including contour processing, LUT handling, Gaussian beam propagation,
and material property calculations.
"""
import os
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import warnings

# Import the new gaussian beam module
from . import gaussian_beam

# Type aliases for clarity
Array = np.ndarray
PathLike = Union[str, os.PathLike]

#------------------------------------------------------------------------------
# PART 1: CONTOUR FUNCTIONS
#------------------------------------------------------------------------------

def getMaxPerContour(C_list: List[Array], theta: float) -> Optional[Array]:
    """
    Find the contour section that gives the maximum spread in DC for a given theta.
    
    Args:
        C_list: List of (N, 2) numpy arrays, each representing a contour section.
        theta: Theta value to compare against.
        
    Returns:
        The contour section with maximum DC spread (and sufficiently long),
        or None if no suitable contour is found.
    """
    # Short-circuit if only one contour section exists
    if len(C_list) == 1:
        return C_list[0]
    
    # Find longest contours
    lengths = np.array([arr.shape[0] for arr in C_list])
    if len(lengths) >= 2:
        min_length = np.sort(lengths)[-2]  # At least as long as second-longest contour
    else:
        min_length = lengths[0]
    
    # Find contour with max DC spread
    max_diff_dc = -np.inf
    best_contour = None
    
    for contour in C_list:
        if contour.shape[0] <= 1:
            continue
            
        diff_dc = np.max(contour[:, 1]) - np.min(contour[:, 1])
        if (diff_dc > max_diff_dc) and (contour.shape[0] >= min_length):
            max_diff_dc = diff_dc
            best_contour = contour
    
    return best_contour


def contourc(x: Array, y: Array, Z: Array, levels: List[float]) -> List[Array]:
    """
    Extract contour lines for specified levels, similar to MATLAB's contourc.
    
    Args:
        x: x-coordinates grid
        y: y-coordinates grid
        Z: 2D array of values to contour
        levels: List of contour levels
        
    Returns:
        List of arrays, each containing vertices of a contour line
    """
    # Generate contour plot and extract paths
    contours = plt.contour(x, y, Z, levels)
    contour_paths = []
    
    # Process each contour collection
    for collection in contours.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            codes = path.codes
            
            # Find start indices of each segment
            if codes is None:
                # Single contour case
                contour_paths.append(vertices)
            else:
                # Multiple segments case
                start_indices = np.where(codes == 1)[0]
                start_indices = np.append(start_indices, len(codes))
                
                # Split contour into segments
                for i in range(len(start_indices) - 1):
                    start, end = start_indices[i], start_indices[i + 1]
                    contour_paths.append(vertices[start:end])
    
    plt.close()  # Close the temporary figure
    return contour_paths


def calcAlphasAvail(theta: float, gratpers: Array, pertDCs: Array, 
                   LUTthetas: Array, LUTalphas: Array, plotStuff: bool = False) -> Tuple[Array, Array, Array]:
    """
    Calculate available alpha values for a given angle theta.
    
    Args:
        theta: Angle in degrees.
        gratpers: Grating periods mesh.
        pertDCs: Perturbation duty cycles mesh.
        LUTthetas: Lookup table of theta values.
        LUTalphas: Lookup table of alpha values.
        plotStuff: Whether to plot debug information.
        
    Returns:
        Tuple of (alpha values, desired periods, desired duty cycles)
    """
    # At each angle, generate period/DC contour
    contour_paths = contourc(gratpers, pertDCs, LUTthetas, [theta])
    C = getMaxPerContour(contour_paths, theta)
    
    if C is None or len(C) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Extract periods and duty cycles from contour
    desper = C[:, 0]
    desDC = C[:, 1]
    
    # Interpolate alpha values at desired points
    interp_func = RegularGridInterpolator(
        (pertDCs, gratpers), 
        LUTalphas, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Note: order is swapped because RegularGridInterpolator uses different convention
    points = C[:, [1, 0]]
    alphac = interp_func(points)
    
    # Visualization for debugging
    if plotStuff:
        _plot_contours(gratpers, pertDCs, LUTalphas, LUTthetas, theta)
    
    return alphac, desper, desDC


def _plot_contours(gratpers, pertDCs, LUTalphas, LUTthetas, theta):
    """Helper function to plot contours for debugging"""
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    
    # Alpha values
    c1 = axs[0].contourf(gratpers * 1e3, pertDCs, LUTalphas)
    axs[0].set_title(r"$\alpha$ ($\mu$m$^{-1}$)")
    axs[0].set_xlabel('Grating period (nm)')
    axs[0].set_ylabel('Perturbation DC')
    fig.colorbar(c1, ax=axs[0])
    
    # Theta values
    c2 = axs[1].contourf(gratpers * 1e3, pertDCs, LUTthetas)
    axs[1].set_title(r"$\theta$ (deg)")
    axs[1].set_xlabel('Grating period (nm)')
    axs[1].set_ylabel('Perturbation DC')
    fig.colorbar(c2, ax=axs[1])
    
    # Smooth theta contour with specified level
    c3 = axs[2].contourf(gratpers * 1e3, pertDCs, LUTthetas)
    rounded_theta = round(theta, 2)
    axs[2].set_title(r"$\theta$ (deg) = " + str(rounded_theta) + " (smooth)")
    axs[2].set_xlabel('Grating period (nm)')
    axs[2].set_ylabel('Perturbation DC')
    fig.colorbar(c3, ax=axs[2])
    axs[2].contour(c3, levels=[theta - 1e-5, theta + 1e-5], colors='r')
    
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# PART 2: LUT HANDLING FUNCTIONS
#------------------------------------------------------------------------------

def maskLUT_AIM(gp: Array, dc: Array, LUTalphas: Array, LUTthetas: Array, 
               minSize: float, wavelength: float, TIRcorner: bool, 
               LUTparams: Dict[str, Any]) -> Tuple[Array, Array, Array]:
    """
    Apply masks to LUT data based on minimum feature size and other constraints.
    
    Args:
        gp: Grating periods array
        dc: Duty cycles array
        LUTalphas: Alpha values from LUT
        LUTthetas: Theta values from LUT
        minSize: Minimum feature size constraint
        wavelength: Operating wavelength in μm
        TIRcorner: Whether to apply total internal reflection corner mask
        LUTparams: Additional parameters
        
    Returns:
        Tuple of (masked alphas, masked thetas, mask array)
    """
    # Create meshgrid for calculations
    gp2, dc2 = np.meshgrid(gp, dc, indexing='xy')
    lambda_nm = round(wavelength * 1e3)
    
    # Initialize mask
    topCorner = np.zeros_like(gp2, dtype=bool)
    
    # Apply TIR corner mask if requested
    if TIRcorner and not (lambda_nm == 532 and LUTparams.get('forward_em', False)):
        # Define corner parameters based on wavelength
        cornerBott, cornerTopR = _get_tir_corner_params(lambda_nm, LUTparams)
        
        if cornerBott > 0:
            # Define corner line
            x1, y1 = gp[0], cornerBott
            x2, y2 = cornerTopR, dc[-1]
            
            # Calculate slope and apply mask
            m = (y2 - y1) / (x2 - x1)
            topCorner = dc2 > m * (gp2 - x1) + y1
    
    # Apply lower right corner mask if needed
    if TIRcorner:
        cornerTopRDC, cornerBottLper = _get_lower_right_params(lambda_nm)
        
        if cornerTopRDC is not None:
            # Define corner line
            x1, y1 = cornerBottLper, dc[0]
            x2, y2 = gp[-1], cornerTopRDC
            
            # Calculate slope and apply mask
            m = (y2 - y1) / (x2 - x1)
            bottCorner = dc2 < m * (gp2 - x1) + y1
            topCorner = topCorner | bottCorner
    
    # Apply minimum feature size constraints
    tooSmall = (gp2 * dc2) < minSize  # Feature too small
    tooSmall_inverse = (gp2 * (1 - dc2)) < minSize  # Gap too small
    
    # Combine all masks
    limMesh = tooSmall | topCorner | tooSmall_inverse
    
    # Apply mask to data
    LUTalphas2 = np.copy(LUTalphas)
    LUTthetas2 = np.copy(LUTthetas)
    
    LUTalphas2[limMesh] = np.nan
    LUTthetas2[limMesh] = np.nan
    
    return LUTalphas2, LUTthetas2, limMesh


def _get_tir_corner_params(lambda_nm: int, LUTparams: Dict[str, Any]) -> Tuple[float, float]:
    """Helper function to get TIR corner parameters based on wavelength"""
    if lambda_nm == 866:
        return 0.1, 0.3525
    elif lambda_nm == 532 and not LUTparams.get('forward_em', False):
        return 0.15, 0.220
    elif lambda_nm == 732:
        if LUTparams.get('toxt', 0) == 6500:
            return 0.22, 0.300
        else:
            return 0.3, 0.290
    elif lambda_nm == 467:
        return 0.6, 0.450
    elif lambda_nm in [397, 405]:
        return 0.6, 0.45
    else:
        return 0, 0


def _get_lower_right_params(lambda_nm: int) -> Tuple[Optional[float], Optional[float]]:
    """Helper function to get lower right corner parameters based on wavelength"""
    if lambda_nm == 732:
        return 0.3, 0.410
    elif lambda_nm == 397:
        return 0.45, 0.65
    elif lambda_nm == 405:
        return 0.25, 0.85
    else:
        return None, None


def loadLUT_AIM(lambda0: float, maxDC: float, p: Dict[str, Any]) -> Dict[str, Array]:
    """
    Load a Look-Up Table (LUT) for a given wavelength and material.
    
    Args:
        lambda0: Wavelength in μm
        maxDC: Maximum duty cycle
        p: Parameters dictionary with material information and lut_file_path
        
    Returns:
        Dictionary containing LUT data
    """
    # Check if a specific LUT file path is provided
    if 'lut_file_path' in p:
        fnamefull = p['lut_file_path']
        print(f"Loading LUT from specified path: {fnamefull}")
    else:
        # Legacy mode: construct filename from parameters (for backward compatibility)
        print("Legacy mode: constructing filename from parameters. For future use, please provide a lut_file_path in the parameters dictionary.")
        material = p.get('material', 'AO_AIM')
        wghBot, wghTop, inth, boxt, toxt = _get_material_properties(material)
        
        # Construct filename
        lambda_str = str(int(round(lambda0 * 1e3))).replace('.', 'p').replace('-', 'm')
        saveName = (f"LUTsweepData_lam{lambda_str}_{material}_boxt{boxt}nm_"
                    f"wghBot{wghBot}nm_wghTop{wghTop}nm_inth{inth}nm_toxt{toxt}nm_AIM_SL1")
        
        if p.get('forward_em', False):
            saveName += '_forward'
        
        # Find and load the file
        fnamefull = os.path.join('', saveName + '.mat')
        print(f"Loading LUT: {saveName}")
    
    # Define variables to load
    loadVars = [
        'gratpersSave', 'pertDCsSave', 'alphasWGSave', 'alphasAirSave', 
        'thetasPoyntingSave', 'thetasHSave', 'nEffsSave', 'thetasFFSave', 
        'thetasFForder2Save', 'peakHeightsFFSave', 'peakHeightsFForder2Save',
        'nEffs_pureSave', 'alphasWG_pureSave'
    ]
    
    # Load data from .mat file
    try:
        mat_contents = scipy.io.loadmat(fnamefull, squeeze_me=True)
        LUT = {var: mat_contents[var] for var in loadVars if var in mat_contents}
        return LUT
    except Exception as e:
        print(f"Error loading LUT file: {e}")
        raise


def _get_material_properties(material: str) -> Tuple[int, int, int, int, int]:
    """Helper function to get material properties for LUT loading"""
    if material == 'SiN_LL':
        return 100, 100, 90, 3000, 6000
    elif material == 'SiN2':
        return 200, 0, 0, 3000, 1500
    elif material == 'HfO2c':
        return 100, 0, 0, 3000, 1000
    elif material == 'HfAl2O3':
        return 80, 80, 0, 3000, 1500
    elif material == 'Al2O3_LL':
        return 100, 100, 0, 3000, 5900
    elif material == 'FN_AIM':
        return 150, 0, 0, 3060, 3000
    else:  # Default to AO_AIM
        return 130, 0, 0, 3340, 3000


def getInterpLUT_gen(nSamps: int, LUT: Dict[str, Array]) -> Dict[str, Array]:
    """
    Generate interpolated LUT data.
    
    Args:
        nSamps: Number of samples to use for interpolation (0 for no interpolation)
        LUT: Original LUT data
        
    Returns:
        Dictionary with interpolated LUT data
    """
    # Extract grating periods and duty cycles
    gratpers = 1e-3 * LUT['gratpersSave']  # Convert to μm
    pertDCs = LUT['pertDCsSave']
    
    # Initialize output dictionary
    LUT2 = {
        'gratpers': gratpers,
        'pertDCs': pertDCs
    }
    
    # Generate new grid if interpolation is requested
    if nSamps > 0:
        gratpers2 = np.linspace(gratpers[0], gratpers[-1], len(gratpers) * nSamps - (nSamps - 1))
        pertDCs2 = np.linspace(pertDCs[0], pertDCs[-1], len(pertDCs) * nSamps - (nSamps - 1))
        gp2, dc2 = np.meshgrid(gratpers2, pertDCs2)
        
        LUT2['gratpers'] = gratpers2
        LUT2['pertDCs'] = pertDCs2
    else:
        gp2, dc2 = np.meshgrid(gratpers, pertDCs)
    
    # Get expected shape
    expected_shape = (len(pertDCs), len(gratpers))
    
    # Process each field
    for field in LUT:
        # Skip the original axis data
        if field in ['gratpersSave', 'pertDCsSave']:
            continue
        
        # Get the data to interpolate
        data = LUT[field]
        
        # Convert 1D to 2D if needed
        if data.ndim == 1 and data.size == expected_shape[0] * expected_shape[1]:
            data = data.reshape(expected_shape)
        
        # Skip if shape doesn't match
        if data.shape != expected_shape:
            print(f"Skipping '{field}' - unexpected shape {data.shape}, expected {expected_shape}")
            continue
        
        # Apply interpolation if requested
        if nSamps > 0:
            interp_func = RegularGridInterpolator(
                (pertDCs, gratpers), 
                data, 
                bounds_error=False, 
                fill_value=np.nan
            )
            
            pts = np.stack([dc2.ravel(), gp2.ravel()], axis=-1)
            new_data = interp_func(pts).reshape(dc2.shape)
        else:
            new_data = data
        
        # Store result with cleaned field name
        LUT2[field.replace('Save', '')] = new_data
    
    return LUT2

#------------------------------------------------------------------------------
# PART 3: MATERIAL PROPERTIES AND OPTICAL FUNCTIONS
#------------------------------------------------------------------------------

def CreateSiCoeffsPalikData(lambdaAir: Union[float, Array]) -> Tuple[Union[float, Array], Union[float, Array]]:
    """
    Calculate Silicon refractive index using Palik's data.
    
    Args:
        lambdaAir: Wavelength in nm
        
    Returns:
        Tuple of (nSi, kSi) - refractive index and extinction coefficient
    """
    try:
        # Load the data files
        with h5py.File('MaterialData/SiPalik_k.mat', 'r') as f:
            wavelengths_k = np.array(f['lum_figure_1/x1']).flatten()
            k = np.array(f['lum_figure_1/y1']).flatten()
            
        with h5py.File('MaterialData/SiPalik_n.mat', 'r') as f:
            wavelengths_n = np.array(f['lum_figure_2/x1']).flatten()
            n = np.array(f['lum_figure_2/y1']).flatten()
            
        # Verify wavelength consistency
        assert np.allclose(wavelengths_k, wavelengths_n), "Mismatch in wavelength data"
        wavelengths = wavelengths_n
        
        # Filter data for valid range
        valid_idx = wavelengths > 0.300
        wavelengths = wavelengths[valid_idx]
        n = n[valid_idx]
        k = k[valid_idx]
        
        # Create interpolation functions
        n_interp = interp1d(wavelengths, n, kind='linear', bounds_error=False, 
                          fill_value='extrapolate')
        k_interp = interp1d(wavelengths, k, kind='linear', bounds_error=False, 
                          fill_value='extrapolate')
        
        # Convert input wavelength to μm
        lambda_um = np.asarray(lambdaAir) * 1e-3
        
        # Interpolate values
        nSi = n_interp(lambda_um)
        kSi = k_interp(lambda_um)
        
        return nSi, kSi
    
    except Exception as e:
        print(f"Error calculating Si coefficients: {e}")
        # Return default values if data loading fails
        return 3.5, 0.0


def getIndices(lambdaMicrons: float) -> Tuple[float, float, float, float]:
    """
    Get refractive indices for various materials at a given wavelength.
    
    Args:
        lambdaMicrons: Wavelength in microns
        
    Returns:
        Tuple of (nOx, nSiN, nSi, kSi) - refractive indices and extinction coefficient
    """
    # Oxide (SiO2) index calculation - Sellmeier equation
    Aox, Box, Cox = 1, 1.113, 8.69e-15
    epsOx = Aox + Box * (lambdaMicrons * 1e-6) ** 2 / ((lambdaMicrons * 1e-6) ** 2 - Cox)
    nOx = np.sqrt(epsOx)
    
    # Silicon Nitride (Si3N4) index calculation - Sellmeier equation
    Asin, Bsin, Csin = 1, 2.503, 17.29e-15
    epsSiN = Asin + Bsin * (lambdaMicrons * 1e-6) ** 2 / ((lambdaMicrons * 1e-6) ** 2 - Csin)
    nSiN = np.sqrt(epsSiN)
    
    # Silicon index lookup from Palik data
    nSi, kSi = CreateSiCoeffsPalikData(lambdaMicrons * 1e3)  # Convert to nm
    
    return nOx, nSiN, nSi, kSi


def transform_coord_frame(rotX: float, rotY: float, rotZ: float, 
                         x0: float, y0: float, z0: float,
                         x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    """
    Transform coordinates between reference frames with rotations.
    
    Args:
        rotX, rotY, rotZ: Rotation angles (radians) around respective axes
        x0, y0, z0: Origin of new coordinate system
        x, y, z: Coordinates to transform
        
    Returns:
        Tuple of (xp, yp, zp) - transformed coordinates
    """
    # Transform formulas derived from rotation matrices
    xp = ((z0 - z) * np.sin(rotY) + 
          np.cos(rotY) * ((x - x0) * np.cos(rotZ) + (y - y0) * np.sin(rotZ)))
    
    yp = (np.cos(rotX) * ((y - y0) * np.cos(rotZ) + (x0 - x) * np.sin(rotZ)) + 
          np.sin(rotX) * ((z - z0) * np.cos(rotY) + 
                         np.sin(rotY) * ((x - x0) * np.cos(rotZ) + (y - y0) * np.sin(rotZ))))
    
    zp = (np.sin(rotX) * ((y0 - y) * np.cos(rotZ) + (x - x0) * np.sin(rotZ)) + 
          np.cos(rotX) * ((z - z0) * np.cos(rotY) + 
                         np.sin(rotY) * ((x - x0) * np.cos(rotZ) + (y - y0) * np.sin(rotZ))))
    
    return xp, yp, zp 

# Use the imported functions from gaussian_beam module
gaussian_free_space_beam_focusing_qparam = gaussian_beam.gaussian_free_space_beam_focusing_qparam
qparam = gaussian_beam.qparam
qtransform_propagate = gaussian_beam.qtransform_propagate
qtransform_refraction_inc_plane_parallel = gaussian_beam.qtransform_refraction_inc_plane_parallel
qtransform_refraction_inc_plane_normal = gaussian_beam.qtransform_refraction_inc_plane_normal
get_rad_waists_invq = gaussian_beam.get_rad_waists_invq
rayprop_chip_to_wg_radcurv = gaussian_beam.rayprop_chip_to_wg_radcurv 