"""
Configuration loader module

Loads and processes configuration files for the grating design process.
"""
import json
import copy
import numpy as np
import os
from typing import Dict, Tuple, Any, Optional

# Import from our modules
from . import helper_functions

def load_config(config_file: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load configuration from a JSON file and prepare parameter dictionaries.
    
    Args:
        config_file: Path to the configuration JSON file
        
    Returns:
        Tuple of (main parameters dict, secondary beam parameters dict)
    """
    # Load JSON configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create parameters dictionary
    p = copy.deepcopy(config)
    
    # Calculate derived parameters
    wavelength = p["wavelength"]
    
    # Get refractive indices
    nOx, _, _, _ = helper_functions.getIndices(p["wavelength"])
    nAir = 1.0
    
    # Calculate beam center and taper start
    if p.get("forward_em", False):
        thetaIncDegrees = -1 * p["thetaIncDegrees"]
        beamCenter = (p["z0"] * np.tan(np.radians(thetaIncDegrees)) +
                     p["tOx"] * np.tan(np.arcsin((nAir / nOx) * 
                                                np.sin(np.radians(thetaIncDegrees)))))
        tapStart = beamCenter - p["tapStartOffsetBeamCentForward"]
        
        p["beamCenter"] = beamCenter
        p["beamCenterChip"] = p["z0"] * np.tan(np.radians(thetaIncDegrees))
        p["beamCenterNoRefract"] = (p["z0"] + p["tOx"]) * np.tan(np.radians(thetaIncDegrees))
    else:
        thetaIncDegrees = p["thetaIncDegrees"]
        tapStart = p.get("tapStart_backEm", -5)
    
    # Calculate wavenumber
    k0 = 2 * np.pi / wavelength
    
    # Update parameters dictionary with calculated values
    p.update({
        "lambda": wavelength,
        "nAir": nAir,
        "nOx": nOx,
        "thetaIncDegrees": thetaIncDegrees,
        "k0": k0,
        "tapStart": tapStart
    })
    
    # Create second beam parameters if needed
    p2 = None
    if p.get("two_beams", False):
        p2 = p.copy()
        p2["w0x"] = p["w0x2"]
        p2["w0y"] = p["w0y2"]
    
    return p, p2 