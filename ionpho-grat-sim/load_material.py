import json
import os
import numpy as np

# Path for materials folder
folder_name = os.path.dirname(os.path.abspath(__file__))
materials_folder = "materials/"  # Folder containing material JSON files

def load_material_json(material_file):
    """Load material specifications from a JSON file.
    
    Parameters:
    -----------
    material_file : str
        Path to the JSON file containing material specifications
        
    Returns:
    --------
    dict
        Dictionary containing material properties
    """
    try:
        with open(material_file, 'r') as f:
            material_data = json.load(f)
        return material_data
    except Exception as e:
        print(f"Error loading material file {material_file}: {e}")
        return None

def get_material_params(material_file):
    """Get material stack parameters from a JSON file.
    
    Parameters:
    -----------
    material_file : str
        Path to material JSON file
        
    Returns:
    --------
    tuple or None
        (wgh_low, wgh_high, inth, material_data) in meters, or None if material not found
    """
    # Material file must be provided
    if not material_file or not os.path.exists(material_file):
        return None
    
    # Load material data from file
    material_data = load_material_json(material_file)
    if not material_data:
        return None
    
    # Extract layer parameters
    wgh_low, wgh_high, inth, is_single = get_layer_parameters(material_data)
    return wgh_low, wgh_high, inth, material_data

def get_layer_parameters(material_data):
    """Extract layer parameters from material data.
    
    Parameters:
    -----------
    material_data : dict
        Dictionary containing material properties
        
    Returns:
    --------
    tuple
        (wgh_low, wgh_high, inth, is_single_layer) in meters
    """
    # Extract layer stack parameters
    layer_stack = material_data.get('layer_stack', {})
    
    # Convert from microns to meters
    wgh_low = layer_stack.get('wghb')   # Bottom waveguide height
    wgh_high = layer_stack.get('wght')  # Top waveguide height
    inth = layer_stack.get('inth')      # Intermediate layer height
    
    # Check if it's a single layer
    is_single_layer = layer_stack.get('is_single_waveguide', True)
    
    # For single layer, ensure no top layer or intermediate layer
    if is_single_layer:
        wgh_high = 0.0
        inth = 0.0
    
    return wgh_low, wgh_high, inth, is_single_layer

def get_optical_properties(material_data, wavelength=None):
    """Extract optical properties from material data, with wavelength-dependent index selection.
    
    Parameters:
    -----------
    material_data : dict
        Dictionary containing material properties
    wavelength : float, optional
        Wavelength in microns to select appropriate index value
        
    Returns:
    --------
    tuple
        (si_index, ox_index, wg_index)
        
    Raises:
    -------
    ValueError
        If wavelength is not provided, wavelength properties are not available,
        or the specific wavelength is not found in the properties
    """
    # Check if wavelength is provided
    if wavelength is None:
        raise ValueError("Wavelength must be provided")
    
    # Get wavelength-specific properties
    wavelength_props = material_data.get('wavelength_properties')
    if not wavelength_props:
        raise ValueError("No wavelength properties found in material data")
    
    # Convert wavelength to nm if it's in microns
    wl_nm = wavelength * 1000 if wavelength < 10 else wavelength
    wl_str = str(int(round(wl_nm)))
    
    # If exact wavelength is available, use it
    if wl_str in wavelength_props:
        props = wavelength_props[wl_str]
        print(f"Using properties for wavelength {wl_str} nm")
    else:
        raise ValueError(f"No properties found for wavelength {wl_str} nm")
    
    # Extract indices from properties
    if 'si_index' not in props or 'ox_index' not in props or 'wg_index' not in props:
        raise ValueError(f"Missing index values for wavelength {wl_str} nm")
    
    si_index = props['si_index']
    ox_index = props['ox_index']
    wg_index = props['wg_index']
    
    return si_index, ox_index, wg_index

def get_simulation_parameters(material_data):
    """Extract simulation parameters from material data.
    
    Parameters:
    -----------
    material_data : dict
        Dictionary containing material properties
        
    Returns:
    --------
    dict
        Dictionary with simulation parameters
    """
    layer_stack = material_data.get('layer_stack', {})
    
    params = {
        'toxt': layer_stack.get('toxt', 2.0),        # Top oxide thickness (um)
        'boxt': layer_stack.get('boxt', 3.0),        # Bottom oxide thickness (um)
        'tSi': layer_stack.get('tSi', 1.0),          # Silicon substrate thickness (um)
        'tair': layer_stack.get('tair', 2.0),        # Air layer thickness (um)
        'process_name': material_data.get('process_name', 'Unknown'),
        'material_name': material_data.get('material_name', 'Unknown')
    }
    
    return params

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        material_file = sys.argv[1]
        material_data = load_material_json(material_file)
        
        if material_data:
            wgh_low, wgh_high, inth, is_single = get_layer_parameters(material_data)
            
            # Test with different wavelengths
            for test_wavelength in [729, 854, 866]:
                try:
                    si_index, ox_index, wg_index = get_optical_properties(material_data, test_wavelength)
                    print(f"\nWavelength: {test_wavelength} nm")
                    print(f"Refractive indices - Si: {si_index}, Oxide: {ox_index}, WG: {wg_index}")
                except ValueError as e:
                    print(f"\nWavelength: {test_wavelength} nm")
                    print(f"Error: {e}")
            
            params = get_simulation_parameters(material_data)
            
            print(f"\nMaterial: {params['material_name']} ({params['process_name']})")
            print(f"Bottom WG height: {wgh_low*1e6:.3f} µm")
            print(f"Top WG height: {wgh_high*1e6:.3f} µm")
            print(f"Intermediate layer height: {inth*1e6:.3f} µm")
            print(f"Single layer: {is_single}")
    else:
        print("Usage: python load_material.py <material_json_file>") 