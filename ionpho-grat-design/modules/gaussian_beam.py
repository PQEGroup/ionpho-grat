"""
Gaussian beam propagation module

This module contains functions for Gaussian beam propagation, q-parameter 
transformations, and beam focusing simulation.
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional, Union, List

# Type aliases
Array = np.ndarray

def qparam(z: Array, w0: float, wavelength: float, n: float) -> Array:
    """
    Calculate the inverse complex q parameter for a Gaussian beam.
    
    Args:
        z: Propagation distance from beam waist (can be array)
        w0: Beam waist radius at focus
        wavelength: Wavelength in same units as z and w0
        n: Refractive index of medium
        
    Returns:
        The inverse complex q parameter (1/q)
    """
    # Rayleigh range
    zR = n * np.pi * w0**2 / wavelength
    
    # Compute inverse q parameter
    return z / (z**2 + zR**2) - 1.0j * zR / (z**2 + zR**2)


def qtransform_propagate(qi: Array, d: float, n: float) -> Array:
    """
    Propagate the inverse complex q parameter through homogeneous medium.
    
    Args:
        qi: Inverse complex q parameter
        d: Propagation distance
        n: Refractive index of medium
        
    Returns:
        Transformed inverse q parameter
    """
    return qi / (1 + (d / n) * qi)


def qtransform_refraction_inc_plane_parallel(qi: Array, nIn: float, nOut: float, thet: float) -> Array:
    """
    Transform q parameter for refraction at interface (parallel component).
    
    Args:
        qi: Inverse complex q parameter
        nIn: Incident medium refractive index
        nOut: Transmitted medium refractive index
        thet: Angle of incidence (radians)
        
    Returns:
        Transformed inverse q parameter
    """
    # Use Snell's law to calculate prefactor
    pref = np.cos(thet) / np.sqrt((nOut / nIn)**2 - np.sin(thet)**2)
    return (pref * qi) / ((nIn / nOut) / pref)


def qtransform_refraction_inc_plane_normal(qi: Array, nIn: float, nOut: float) -> Array:
    """
    Transform q parameter for normal refraction at interface.
    
    Args:
        qi: Inverse complex q parameter
        nIn: Incident medium refractive index
        nOut: Transmitted medium refractive index
        
    Returns:
        Transformed inverse q parameter
    """
    return (nIn / nOut) * qi


def get_rad_waists_invq(qi: Array, lambda_um: float, n: float) -> Tuple[Array, Array]:
    """
    Calculate beam radius of curvature and beam waist from inverse q parameter.
    
    Args:
        qi: Inverse complex q parameter
        lambda_um: Wavelength in micrometers
        n: Refractive index
        
    Returns:
        Tuple of (radius of curvature, beam waist)
    """
    # Use error handling to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Radius of curvature from real part
        R = np.where(np.real(qi) != 0, 1/np.real(qi), np.inf)
        
        # Beam waist from imaginary part
        w_squared = np.where(np.imag(qi) != 0, 
                            (-1/np.imag(qi)) * (lambda_um/(n*np.pi)), 
                            np.nan)
        w = np.sqrt(w_squared)
    
    return R, w


def rayprop_chip_to_wg_radcurv(Xc: Array, Yc: Array, vals: Dict[str, Any]) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Propagate rays from chip to waveguide coordinates.
    
    Args:
        Xc, Yc: Coordinates on the chip
        vals: Parameter dictionary containing beam and material properties
        
    Returns:
        Tuple of (Xwg, Ywg, theta0, thetaOx, phis) - waveguide coordinates and angles
    """
    # Extract parameters
    x0, y0, z0 = vals['x0'], vals['y0'], vals['z0']
    nAir, nOx = vals['nAir'], vals['nOx']
    tOx = vals['tOx']
    w0x, w0y = vals['w0x'], vals['w0y']
    lambda_um = vals['lambda']
    nPts = vals['nPts']
    thetaIncDegrees = vals['thetaIncDegrees']
    
    # Get beam offsets
    beamOffx = vals.get('zOffx', 0) / np.cos(np.radians(thetaIncDegrees))
    beamOffy = vals.get('zOffy', 0) / np.cos(np.radians(thetaIncDegrees))
    
    # Transform coordinates to beam frame
    XbeamC, YbeamC, ZbeamC = transform_coord_frame(
        0, np.pi/2 + (np.pi/2 - np.radians(thetaIncDegrees)), 0,
        x0, y0, z0 + tOx, Xc, Yc, tOx
    )
    
    # Beam parameters in chip coordinates
    qixMeshC = qparam(ZbeamC + beamOffx, w0x, lambda_um, nAir)
    qiyMeshC = qparam(ZbeamC + beamOffy, w0y, lambda_um, nAir)
    
    # Rayleigh ranges
    zRx = np.pi * w0x**2 / lambda_um
    zRy = np.pi * w0y**2 / lambda_um
    
    # Shifted Z coordinates
    Zbx = ZbeamC + beamOffx
    Zby = ZbeamC + beamOffy
    
    # Distance correction from radius of curvature
    # This complex computation is needed for accurate phase front tracking
    dzFromRadsX = compute_dz_correction(Zbx, zRx, XbeamC)
    dzFromRadsY = compute_dz_correction(Zby, zRy, YbeamC)
    
    # Skip correction if requested
    if vals.get('no_dz_corr', False):
        dzFromRadsX = np.zeros_like(dzFromRadsX)
        dzFromRadsY = np.zeros_like(dzFromRadsY)
    
    # Compute q parameters with corrections for angle calculations
    qixMeshCForAngles = qparam(ZbeamC + beamOffx + dzFromRadsX, w0x, lambda_um, nAir)
    qiyMeshCForAngles = qparam(ZbeamC + beamOffy + dzFromRadsY, w0y, lambda_um, nAir)
    
    # Get radius of curvature
    Rxs, _ = get_rad_waists_invq(qixMeshCForAngles, lambda_um, nAir)
    Rys, _ = get_rad_waists_invq(qiyMeshCForAngles, lambda_um, nAir)
    
    # Center line radius
    RxsCenter, _ = get_rad_waists_invq(qixMeshCForAngles[nPts//2, :], lambda_um, nAir)
    
    # Angle calculations
    theta_beamC = np.arcsin(XbeamC[nPts//2, :] / RxsCenter)
    theta0 = np.ones_like(theta_beamC) * np.radians(thetaIncDegrees) - theta_beamC
    
    phis = np.arcsin(YbeamC / Rys)
    theta_beamC_full = np.arcsin(XbeamC / Rxs)
    theta0_full = np.ones_like(theta_beamC_full) * np.radians(thetaIncDegrees) - theta_beamC_full
    
    # Refraction into oxide
    thetaOx = np.arcsin((nAir / nOx) * np.sin(theta0_full))
    
    # Distance traveled through oxide
    deltaR = tOx * np.tan(thetaOx)
    
    # Compute waveguide coordinates
    Xwg = Xc + deltaR * np.cos(np.sign(thetaIncDegrees) * phis)
    Ywg = Yc + deltaR * np.sin(np.sign(thetaIncDegrees) * phis)
    
    return Xwg, Ywg, theta0, thetaOx, phis


def compute_dz_correction(Z: Array, zR: float, coord: Array) -> Array:
    """
    Compute the dz correction for accurate phase front tracking.
    
    Args:
        Z: Z-coordinate
        zR: Rayleigh range
        coord: Transverse coordinate (X or Y)
        
    Returns:
        dz correction
    """
    # Handle zeros and very small values to avoid numerical issues
    with np.errstate(all='ignore'):
        # Calculate the discriminant directly instead of creating a complex number from arrays
        discriminant = 729 * Z**2 * zR**4 - 27 * (coord**2 + Z**2 - 2*zR**2)**3
        # Apply complex sqrt element-wise
        sqrt_term = np.zeros_like(discriminant, dtype=complex)
        sqrt_term = np.sqrt(discriminant + 0j)  # Adding 0j ensures complex result
        
        term1 = (9 * Z * zR**2 + (1/3) * sqrt_term)
        term1 = term1**(1/3) / (3**(2/3))
        
        term2 = (coord**2 + Z**2 - 2*zR**2) / term1
        
        dzFromRads = -Z + term1 + term2
    
    # Return only real component, ignore any small imaginary parts
    return np.real(dzFromRads)


def transform_coord_frame(rotX: float, rotY: float, rotZ: float, 
                         x0: float, y0: float, z0: float,
                         x: Array, y: Array, z: float) -> Tuple[Array, Array, Array]:
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


def gaussian_free_space_beam_focusing_qparam(vals: Dict[str, Any]) -> Tuple:
    """
    Calculate parameters for a focused Gaussian beam.
    
    Args:
        vals: Dictionary of beam and material parameters
        
    Returns:
        Tuple containing phase, amplitude, and spatial data for the beam
    """
    # Extract parameters
    x0, y0, z0 = vals['x0'], vals['y0'], vals['z0']
    nAir, nOx = vals['nAir'], vals['nOx']
    w0x, w0y = vals['w0x'], vals['w0y']
    tOx = vals['tOx']
    wavelength = vals['lambda']
    thetaIncDegrees = vals['thetaIncDegrees']
    nPts = vals['nPts']
    k0 = vals['k0']
    
    # Scanning factors
    apPowFacXFront = vals.get('scanFacXFront', 2)
    apPowFacXBack = vals.get('scanFacXBack', 2)
    apPowFacY = vals.get('scanFacY', 3)
    
    # Center distance 
    dAirCent = z0 / np.cos(np.radians(thetaIncDegrees))
    
    # Beam offsets
    beamOffx = vals.get('zOffx', 0) / np.cos(np.radians(thetaIncDegrees))
    beamOffy = vals.get('zOffy', 0) / np.cos(np.radians(thetaIncDegrees))
    
    # Initial q parameters at center
    qix1 = qparam(dAirCent + beamOffx, w0x, wavelength, nAir)
    qiy1 = qparam(dAirCent + beamOffy, w0y, wavelength, nAir)
    
    # Get waist and radius info for chip plane
    ycRadius, ycWaist = get_rad_waists_invq(qiy1, wavelength, nAir)
    xcRadius, xcWaist = get_rad_waists_invq(qix1, wavelength, nAir)
    xcWaistplane = xcWaist / np.cos(np.radians(thetaIncDegrees))
    
    # Beam center location on chip
    xcent = x0 + z0 * np.tan(np.radians(thetaIncDegrees))
    
    # Waist locations
    xcWaists = [xcent - xcWaistplane, xcent + xcWaistplane, xcent]
    ycWaists = [-ycWaist, ycWaist]
    
    # Create coordinate grid on chip
    xcMax = xcent + np.abs(apPowFacXBack * xcWaistplane)
    xcMin = xcent - np.abs(apPowFacXFront * xcWaistplane)
    
    xcRange = np.linspace(xcMin, xcMax, nPts)
    ycRange = np.linspace(-ycWaist * apPowFacY, ycWaist * apPowFacY, nPts)
    
    Xc, Yc = np.meshgrid(xcRange, ycRange)
    
    # Propagate to waveguide plane
    Xwg, Ywg, theta0, thetaOx, _ = rayprop_chip_to_wg_radcurv(Xc, Yc, vals)
    
    # Refraction angle at center
    thetaOxCent = np.arcsin((nAir/nOx) * np.sin(np.radians(thetaIncDegrees)))
    dOxCent = tOx / np.cos(thetaOxCent)
    
    # Get more q parameters
    qix0 = qparam(beamOffx, w0x, wavelength, nAir)
    qiy0 = qparam(beamOffy, w0y, wavelength, nAir)
    
    # Transform q parameters at interface
    qix2 = qtransform_refraction_inc_plane_parallel(
        qix1, nAir, nOx, np.radians(thetaIncDegrees))
    qiy2 = qtransform_refraction_inc_plane_normal(qiy1, nAir, nOx)
    
    # Optional debug output
    if vals.get('showBeamParams', False):
        _print_beam_params(qix0, qiy0, qix1, qiy1, qix2, qiy2, 
                          dAirCent, ycWaist, wavelength, nAir, nOx)
    
    # Beam waist in waveguide
    _, ywgWaist = get_rad_waists_invq(
        qtransform_propagate(qiy2, dOxCent, nOx), wavelength, nOx)
    _, xwgWaist = get_rad_waists_invq(
        qtransform_propagate(qix2, dOxCent, nOx), wavelength, nOx)
    
    # More waveguide calculations
    xwgWaistplane = xwgWaist / np.cos(thetaOxCent)
    xwgcent = xcent + tOx * np.tan(thetaOxCent)
    
    XwgWaists = [xwgcent - xwgWaistplane, xwgcent + xwgWaistplane, xwgcent]
    YwgWaists = [-ywgWaist, ywgWaist]
    
    # Transform to beam coordinates in WG plane
    Xwg_beam, Ywg_beam, Zwg_beam = transform_coord_frame(
        0, (np.pi/2 + (np.pi/2 - thetaOxCent)), 0,
        xcent, y0, tOx, Xwg, Ywg, 0
    )
    
    # Calculate q parameters in WG plane
    qixMesh = qtransform_propagate(qix2, Zwg_beam, nOx)
    qiyMesh = qtransform_propagate(qiy2, Zwg_beam, nOx)
    
    # Phase calculations
    # Gouy phase
    phi_guoy = 1j * (vals.get('xOrdIdx', 0) + vals.get('yOrdIdx', 0) + 1/2) * (
        np.arctan(np.real(1/qixMesh) / np.imag(1/qixMesh)) +
        np.arctan(np.real(1/qiyMesh) / np.imag(1/qiyMesh))
    )
    
    # Accumulation phase
    phi_accum = -1j * (k0 * nOx * Zwg_beam + k0 * nAir * dAirCent)
    
    # Radius of curvature phase
    phi_radCurv = -1j * k0 * nOx * (
        (Xwg_beam**2) * qixMesh + (Ywg_beam**2) * qiyMesh
    ) / 2
    
    # Total phase term
    phase_term = phi_accum + phi_radCurv + phi_guoy
    
    # Field calculations
    E0z = np.sqrt(qixMesh) * np.sqrt(qiyMesh)
    Efull = E0z * np.exp(phase_term)
    EfieldAmp = np.abs(Efull)
    phi_fs = -1 * np.imag(phase_term)
    
    # Similar calculations for chip plane
    Xc_beam, Yc_beam, Zc_beam = transform_coord_frame(
        0, (np.pi/2 + (np.pi/2 - np.radians(thetaIncDegrees))), 0,
        x0, y0, z0 + tOx, Xc, Yc, tOx
    )
    
    qixMeshC = qparam(Zc_beam + beamOffx, w0x, wavelength, nAir)
    qiyMeshC = qparam(Zc_beam + beamOffy, w0y, wavelength, nAir)
    
    phi_guoyC = 1j * (vals.get('xOrdIdx', 0) + vals.get('yOrdIdx', 0) + 1/2) * (
        np.arctan(np.real(1/qixMeshC) / np.imag(1/qixMeshC)) +
        np.arctan(np.real(1/qiyMeshC) / np.imag(1/qiyMeshC))
    )
    
    phi_accumC = -1j * (k0 * nAir * Zc_beam)
    phi_radCurvC = -1j * k0 * nAir * (
        (Xc_beam**2) * qixMeshC + (Yc_beam**2) * qiyMeshC
    ) / 2
    
    phase_termC = phi_accumC + phi_radCurvC + phi_guoyC
    
    E0zC = np.sqrt(qixMeshC) * np.sqrt(qiyMeshC)
    EfullC = E0zC * np.exp(phase_termC)
    EfieldAmpChip = np.abs(EfullC)
    phi_fsChip = -1 * np.imag(phase_termC)
    
    return (phi_fs, EfieldAmp, phi_fsChip, EfieldAmpChip, theta0,
            Xc, Yc, Xwg, Ywg, XwgWaists, YwgWaists, Efull)


def _print_beam_params(qix0, qiy0, qix1, qiy1, qix2, qiy2, 
                      dAirCent, ycWaist, wavelength, nAir, nOx):
    """Helper function to print beam parameters for debugging"""
    Rx0, wx0 = get_rad_waists_invq(qix0, wavelength, nAir)
    Ry0, wy0 = get_rad_waists_invq(qiy0, wavelength, nAir)
    
    Rx1, wx1 = get_rad_waists_invq(qix1, wavelength, nAir)
    Ry1, wy1 = get_rad_waists_invq(qiy1, wavelength, nAir)
    
    Rx2, wx2 = get_rad_waists_invq(qix2, wavelength, nOx)
    Ry2, wy2 = get_rad_waists_invq(qiy2, wavelength, nOx)
    
    print(f"X-beam parameters:")
    print(f"  q0: R={Rx0:.2f}μm, w={wx0:.2f}μm")
    print(f"  q1: R={Rx1:.2f}μm, w={wx1:.2f}μm")
    print(f"  q2: R={Rx2:.2f}μm, w={wx2:.2f}μm")
    print("------------")
    print(f"Y-beam parameters:")
    print(f"  q0: R={Ry0:.2f}μm, w={wy0:.2f}μm")
    print(f"  q1: R={Ry1:.2f}μm, w={wy1:.2f}μm")
    print(f"  q2: R={Ry2:.2f}μm, w={wy2:.2f}μm")
    print("------------")
    print(f"dAirCent (dist to chip): {dAirCent:.2f}μm")
    print(f"1.422*ycWaist: {1.422*ycWaist:.2f}μm") 