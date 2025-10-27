"""
Beam propagation module

This module contains functions for calculating Gaussian beam propagation
and related properties.
"""
import numpy as np
from typing import Dict, Tuple, Union, List, Any, Optional
import matplotlib.pyplot as plt

# Import functions from our modules
from . import helper_functions
from . import gaussian_beam

# Type aliases
Array = np.ndarray

def calculate_beam_propagation(p: Dict[str, Any], p2: Optional[Dict[str, Any]] = None):
    """
    Calculate beam propagation for one or two beams.
    
    Args:
        p: Parameters dictionary for the first beam
        p2: Parameters dictionary for the second beam (optional)
        
    Returns:
        Dictionary containing beam propagation results
    """
    # First beam parameters
    print("Calculating main beam parameters...")
    phi_fs, EfieldAmp, phi_fsChip, EfieldAmpChip, theta0, Xc, Yc, Xwg, Ywg, XwgWaists, YwgWaists, _ = gaussian_beam.gaussian_free_space_beam_focusing_qparam(p)
    
    # Process second beam if provided
    if p2 is not None:
        print("Calculating second beam parameters...")
        phi_fs2, EfieldAmp2, phi_fsChip2, EfieldAmpChip2, theta02, Xc2, Yc2, Xwg2, Ywg2, _, _, Efull2 = gaussian_beam.gaussian_free_space_beam_focusing_qparam(p2)
        
        # Return combined results for both beams
        return [phi_fs, EfieldAmp, phi_fsChip, EfieldAmpChip, theta0, Xc, Yc, Xwg, Ywg, 
                phi_fs2, EfieldAmp2, phi_fsChip2, EfieldAmpChip2, theta02, Xc2, Yc2, Xwg2, Ywg2]
    
    # Process beam parameters
    nPts = p["nPts"]
    
    # Compare higher-order Gouy phases if requested
    if p.get("compareHigherOrderPhases", False):
        if p.get("useHigherOrderPhases", False):
            print("ALREADY USING HIGHER ORDER PHASE, NOTHING TO COMPARE TO")
        else:
            p0 = p.copy()
            p0["xOrdIdx"] = p.get("xOrdIdx", 0)
            p0["yOrdIdx"] = p.get("yOrdIdx", 0)
            phi_fs_HG, *_ = gaussian_beam.gaussian_free_space_beam_focusing_qparam(p0)
            
            # Create comparison plot (omitted - was using plotting_all)
            # If needed, implement here with matplotlib directly
    
    # Normalize E-field amplitudes
    EfieldAmpNorm = EfieldAmp / np.max(EfieldAmp)
    EinsideWaist = np.where(EfieldAmpNorm < 1/np.exp(1), 0, EfieldAmpNorm)
    
    EfieldAmpNormChip = EfieldAmpChip / np.max(EfieldAmpChip)
    EinsideWaistChip = np.where(EfieldAmpNormChip < 1/np.exp(1), 0, EfieldAmpNormChip)
    
    # Calculate waist parameters
    EiwXaxis = EinsideWaist[nPts // 2, :]
    peakXidx = np.argmax(EiwXaxis)
    
    XwgWaists = [Xwg[nPts // 2, np.where(EiwXaxis)[0][0]],
                 Xwg[nPts // 2, np.where(EiwXaxis)[0][-1]],
                 Xwg[nPts // 2, peakXidx]]
    
    EiwYaxis = EinsideWaist[:, peakXidx]
    YwgWaists = [Ywg[np.where(EiwYaxis)[0][0], peakXidx],
                 Ywg[np.where(EiwYaxis)[0][-1], peakXidx]]
    
    EiwXaxisChip = EinsideWaistChip[nPts // 2, :]
    peakXidxChip = np.argmax(EiwXaxisChip)
    
    XcWaists = [Xc[nPts // 2, np.where(EiwXaxisChip)[0][0]],
                Xc[nPts // 2, np.where(EiwXaxisChip)[0][-1]],
                Xc[nPts // 2, peakXidxChip]]
    
    EiwYaxisChip = EinsideWaistChip[:, peakXidxChip]
    YcWaists = [Yc[np.where(EiwYaxisChip)[0][0], peakXidxChip],
                Yc[np.where(EiwYaxisChip)[0][-1], peakXidxChip]]
    
    # Display beam parameters if requested
    if p.get("showBeamParams", False):
        tapangAdd_showBeamParams = p.get("tapangAdd_showBeamParams", 3)
        tapStart = p["tapStart"]
        
        tapang = np.arctan((2.844 * YwgWaists[1] / 2) / (XwgWaists[2] - tapStart))
        print(f"tapang: {np.degrees(tapang):.2f}°")
        
        dyCenterWG_tapangAdd = (XwgWaists[2] - tapStart) * (np.tan(tapang + np.radians(tapangAdd_showBeamParams)) - np.tan(tapang))
        
        YwgCenterOrig = 2.844 * YwgWaists[1] / 2
        YwgCenterNew_tapangAdd = YwgCenterOrig + dyCenterWG_tapangAdd
        
        Xwgt, Ywgt, *_ = gaussian_beam.rayprop_chip_to_wg_radcurv(Xc, Yc, p)
        
        yidxOrig = np.argmin(np.abs(Ywgt[:, 0] - YwgCenterOrig))
        yChipCenterOrig = Yc[yidxOrig, 0]
        
        print(f"-----No tapang:-----\nNo correction (at chip): {yChipCenterOrig:.3f}")
        
        yidx = np.argmin(np.abs(Ywgt[:, 0] - YwgCenterNew_tapangAdd))
        YChipCenterNew_tapangAdd = Yc[yidx, 0]
        
        print(f"-------- tapangAdd {tapangAdd_showBeamParams} -----\nWG: {YwgCenterNew_tapangAdd:.3f}\nChip: {YChipCenterNew_tapangAdd:.3f}")
    
    # Prepare and return results
    waist_info = {
        "XwgWaists": XwgWaists,
        "YwgWaists": YwgWaists,
        "XcWaists": XcWaists,
        "YcWaists": YcWaists
    }
    
    return phi_fs, EfieldAmp, phi_fsChip, EfieldAmpChip, theta0, Xc, Yc, Xwg, Ywg, waist_info 