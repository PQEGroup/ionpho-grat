import json
import os

def load_material_data(material_file):
    """
    Load material data from a material JSON file
    
    Args:
        material_file: Path to the material JSON file
        
    Returns:
        Dictionary containing material parameters
        
    Raises:
        FileNotFoundError: If the material file doesn't exist
        json.JSONDecodeError: If the material file is not valid JSON
    """
    try:
        with open(material_file, 'r') as f:
            material_data = json.load(f)
        return material_data
    except FileNotFoundError:
        print(f"Error: Material file not found: {material_file}")
        print_format_info()
        raise FileNotFoundError(f"Material file not found: {material_file}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in material file: {material_file}")
        print(f"JSON error: {str(e)}")
        print_format_info()
        raise

def get_material_params(material_file):
    """
    Get material parameters from a material JSON file
    
    Args:
        material_file: Path to the material JSON file
        
    Returns:
        For dual waveguide configuration:
            toxt, boxt, wghb, wght, inth, tair, tSi: Layer thicknesses for dual waveguide
        For single waveguide configuration:
            toxt, boxt, wghb, tair, tSi: Layer thicknesses for single waveguide
        
    Raises:
        FileNotFoundError: If the material file doesn't exist
        KeyError: If required parameters are missing from the material file
        json.JSONDecodeError: If the material file is not valid JSON
    """
    try:
        material_data = load_material_data(material_file)
        
        if "layer_stack" not in material_data:
            print(f"Error: 'layer_stack' section missing from material file: {material_file}")
            print_format_info()
            raise KeyError(f"'layer_stack' section missing from material file: {material_file}")
            
        layer_stack = material_data["layer_stack"]
        
        # Always required parameters for any stack configuration
        required_params = ["toxt", "boxt", "tair", "tSi", "wghb"]
        missing_params = [param for param in required_params if param not in layer_stack]
        
        if missing_params:
            missing_str = ", ".join(missing_params)
            print(f"Error: Required parameters missing from layer_stack: {missing_str}")
            print_format_info()
            raise KeyError(f"Required parameters missing from layer_stack: {missing_str}")
        
        # Extract basic parameters
        toxt = layer_stack["toxt"]
        boxt = layer_stack["boxt"]
        wghb = layer_stack["wghb"]  # Bottom waveguide height
        tair = layer_stack["tair"]
        tSi = layer_stack["tSi"]
        
        # Check if this is a dual waveguide stack
        is_dual_waveguide = False
        
        # First check for explicit flag
        if "is_single_waveguide" in layer_stack:
            is_dual_waveguide = not layer_stack["is_single_waveguide"]
        # Otherwise check for non-zero wght and inth
        elif "wght" in layer_stack and "inth" in layer_stack:
            wght = layer_stack.get("wght", 0)
            inth = layer_stack.get("inth", 0)
            is_dual_waveguide = wght > 0 and inth > 0
        
        if is_dual_waveguide:
            # Additional parameters for dual waveguide
            if "wght" not in layer_stack or "inth" not in layer_stack:
                missing_str = ", ".join(param for param in ["wght", "inth"] if param not in layer_stack)
                print(f"Error: Dual waveguide requires 'wght' and 'inth' parameters, missing: {missing_str}")
                print_format_info()
                raise KeyError(f"Dual waveguide requires 'wght' and 'inth' parameters, missing: {missing_str}")
            
            wght = layer_stack["wght"]  # Top waveguide height
            inth = layer_stack["inth"]  # Intermediate layer height
            
            return toxt, boxt, wghb, wght, inth, tair, tSi
        else:
            # Single waveguide configuration
            return toxt, boxt, wghb, tair, tSi
        
    except (KeyError, TypeError) as e:
        print(f"Error processing material file '{material_file}': {str(e)}")
        print_format_info()
        raise

def get_medium_from_index(n):
    """
    Create a Tidy3D medium from a refractive index
    
    Args:
        n: Refractive index
        
    Returns:
        td.Medium object with the specified index
    """
    try:
        import tidy3d as td
        return td.Medium(permittivity=n**2)
    except ImportError:
        print("Warning: tidy3d not found. Returning n instead.")
        return n

def get_process_name_from_file(material_file):
    """
    Extract the process name from a material JSON file
    
    Args:
        material_file: Path to the material JSON file
        
    Returns:
        Process name from the material file
        
    Raises:
        FileNotFoundError: If the material file doesn't exist
        KeyError: If process_name is not found in the material file
        json.JSONDecodeError: If the material file is not valid JSON
    """
    material_data = load_material_data(material_file)
    if "process_name" not in material_data or not material_data["process_name"]:
        print(f"Error: process_name not defined in material file: {material_file}")
        print_format_info()
        raise KeyError(f"process_name not defined or empty in material file: {material_file}")
    return material_data["process_name"]

def is_dual_waveguide_stack(material_file):
    """
    Check if the material file contains a dual waveguide stack configuration
    
    Args:
        material_file: Path to the material JSON file
        
    Returns:
        True if the material file contains a dual waveguide stack configuration, False otherwise
    """
    material_data = load_material_data(material_file)
    if "layer_stack" not in material_data:
        return False
        
    layer_stack = material_data["layer_stack"]
    # Check for the explicit flag first
    if "is_single_waveguide" in layer_stack:
        return not layer_stack["is_single_waveguide"]
    
    # Fall back to checking if wght and inth are non-zero
    if "wght" in layer_stack and "inth" in layer_stack:
        return layer_stack["wght"] > 0 and layer_stack["inth"] > 0
    
    return False

def print_expected_material_format():
    """Print the expected format of a material JSON file."""
    expected_format = {
        "process_name": "Process name (e.g., AIM_SL1, Lionix)",
        "material_name": "Material name (e.g., AO_AIM, FN_AIM)",
        "csv_file": "Path to CSV file with wavelength and refractive index data (can be absolute or relative to the material file)",
        "layer_stack": {
            "toxt": "Top oxide thickness in microns (float)",
            "boxt": "Bottom oxide thickness in microns (float)",
            "wghb": "Bottom waveguide height in microns (float)",
            "wght": "Top waveguide height in microns (float, 0 for single waveguide)",
            "inth": "Intermediate layer height between waveguides in microns (float, 0 for single waveguide)",
            "tair": "Air thickness in microns (float)",
            "tSi": "Silicon thickness in microns (float)",
            "is_single_waveguide": "Boolean flag to indicate single waveguide mode (true = single, false = dual)"
        }
    }
    
    print("\nExpected material JSON format:")
    print(json.dumps(expected_format, indent=2))
    print("\nSingle Waveguide Configuration (is_single_waveguide = true):")
    example_single = {
        "process_name": "AIM_SL1",
        "material_name": "AO_AIM",
        "csv_file": "AO_AIM.csv",
        "layer_stack": {
            "toxt": 3.0,
            "boxt": 3.34,
            "wghb": 0.13,
            "wght": 0.0,
            "inth": 0.0,
            "tair": 1.0,
            "tSi": 1.0,
            "is_single_waveguide": True
        }
    }
    print(json.dumps(example_single, indent=2))
    
    print("\nDual Waveguide Configuration (is_single_waveguide = false):")
    example_dual = {
        "process_name": "AIM_SL1",
        "material_name": "FN_AIM",
        "csv_file": "FN_AIM.csv",
        "layer_stack": {
            "toxt": 3.0,
            "boxt": 3.34,
            "wghb": 0.22,
            "wght": 0.13,
            "inth": 0.12,
            "tair": 1.0,
            "tSi": 1.0,
            "is_single_waveguide": False
        }
    }
    print(json.dumps(example_dual, indent=2))
    
    print("\nExpected CSV file format (comma-separated):")
    print("wavelength_um,n,k")
    print("0.375,1.679,0")
    print("0.397,1.674,0")
    print("0.422,1.674,0")
    print("\nFirst row is skipped as header. The k column (imaginary part) is optional.")
    print("\nNOTE: The CSV file should be named to match the material_name in the JSON file (e.g., AO_AIM.csv)")
    print("\nIMPORTANT: The 'csv_file' field in the material JSON is mandatory and should point to a CSV file with wavelength and refractive index data.")

def print_expected_config_format():
    """Print the expected format of a simulation configuration JSON file."""
    expected_format = {
        "simulation": {
            "material": "Material name (string)",
            "material_file": "Path to material JSON file (string)",
            "process_name": "Process name (string)",
            "output_dir": "Output directory path or null",
            "include_2d_monitor": "Whether to include a 2D field monitor (boolean)",
            "period": {
                "min": "Minimum grating period in microns (float)",
                "max": "Maximum grating period in microns (float)",
                "steps": "Number of period steps (integer)"
            },
            "gratDC": {
                "min": "Minimum grating duty cycle (float between 0-1)",
                "max": "Maximum grating duty cycle (float between 0-1)",
                "steps": "Number of duty cycle steps (integer)"
            },
            "material_config": {
                "oxide_source": "Source of oxide material: 'default', 'fixed_index', or 'csv_file' (string)",
                "oxide_index": "Fixed refractive index for oxide if oxide_source='fixed_index' (float or null)",
                "oxide_csv_file": "Path to CSV file with oxide refractive index data if oxide_source='csv_file' (string or null)",
                "max_num_poles": "Maximum number of poles to use in dispersion fitting (integer)"
            },
            "boundary_conditions": {
                "use_pml": "Whether to use PML boundary conditions (boolean)",
                "absorber_layers": "Number of absorber layers if use_pml=false (integer)"
            },
            "simulation_params": {
                "l_pre": "Pre-grating length in microns (float)",
                "l_post": "Post-grating length in microns (float)",
                "run_time": "Simulation run time in seconds (float)",
                "N": "Number of grating periods (integer)",
                "grid_dl": "Grid spacing for simulation mesh (float)",
                "addl_str": "Additional string tag for file naming (string)",
                "wavelengths": "List of wavelengths in nm to simulate (array of floats)"
            },
            "execution": {
                "sequential": "Whether to run simulations sequentially (boolean)",
                "max_workers": "Maximum number of concurrent simulations (integer)",
                "plot_first": "Whether to plot the first simulation (boolean)"
            }
        },
        "post_processing": {
            "show_plots": "Whether to display plots for each simulation (boolean)",
            "plot_results": "Whether to generate summary plots (boolean)",
            "metric": "Primary metric to plot (string)",
            "wavelength": "Specific wavelength to plot or null for all wavelengths (float or null)"
        }
    }
    
    print("\nExpected simulation configuration JSON format:")
    print(json.dumps(expected_format, indent=2))
    print("\nExample configuration JSON file:")
    example = {
        "simulation": {
            "material": "AO_AIM",
            "material_file": "materials/AIM_SL1/AO_AIM.json",
            "process_name": "AIM_SL1",
            "output_dir": None,
            "include_2d_monitor": False,
            "period": {
                "min": 0.5,
                "max": 0.6,
                "steps": 3
            },
            "gratDC": {
                "min": 0.4,
                "max": 0.6,
                "steps": 3
            },
            "material_config": {
                "oxide_source": "default",
                "oxide_index": None,
                "oxide_csv_file": None,
                "max_num_poles": 1
            },
            "boundary_conditions": {
                "use_pml": True,
                "absorber_layers": 60
            },
            "simulation_params": {
                "l_pre": 5.0,
                "l_post": 5.0,
                "run_time": 0.55e-12,
                "N": 60,
                "grid_dl": 0.01,
                "addl_str": "test_config",
                "wavelengths": [1550, 1310]
            },
            "execution": {
                "sequential": False,
                "max_workers": 4,
                "plot_first": True
            }
        },
        "post_processing": {
            "show_plots": False,
            "plot_results": True,
            "metric": "up_vs_input",
            "wavelength": None
        }
    }
    print(json.dumps(example, indent=2))

def print_format_info():
    """Print information about the expected format of JSON files."""
    print("\n=== JSON FORMAT INFORMATION ===\n")
    print("The simulation requires two types of JSON files:")
    print("1. Material JSON file: Contains material properties, layer thicknesses, and refractive indices")
    print("2. Simulation configuration JSON file: Contains simulation parameters and settings")
    
    print_expected_material_format()
    print_expected_config_format()
    
    print("\nPlease ensure your JSON files follow these formats.")
    print("All parameters shown are required unless otherwise specified.")
    print("=== END OF JSON FORMAT INFORMATION ===\n")

def get_medium_from_csv(csv_file, max_num_poles=1):
    """
    Create a Tidy3D medium from wavelength and refractive index data in a CSV file.
    The CSV file should have 2 or 3 columns: wavelength (nm), n, and optionally k.
    
    Args:
        csv_file: Path to the CSV file containing wavelength and refractive index data
        max_num_poles: Maximum number of poles to use in the dispersion fitting (default: 1)
        
    Returns:
        td.Medium object with the dispersion data
    """
    try:
        import tidy3d as td
        from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter
        
        # Create a fitter from the CSV file, skipping the first row as header
        print(f"Reading dispersion data from {csv_file}")
        fitter = FastDispersionFitter.from_file(csv_file, skiprows=1, delimiter=",")
        
        # Perform the fit with specified max number of poles
        print(f"Fitting dispersion model to data with max {max_num_poles} poles...")
        advanced_param = AdvancedFastFitterParam(weights=(1, 1))
        medium, rms_error = fitter.fit(
            max_num_poles=max_num_poles, 
            advanced_param=advanced_param
        )
        
        print(f"Fit complete with RMS error: {rms_error:.6f}")
        print(f"Medium: {medium}")
        
        return medium
        
    except ImportError as e:
        print(f"Warning: Required libraries not found: {str(e)}")
        print("Please install the latest version of tidy3d with dispersion support")
        raise
    except Exception as e:
        print(f"Error creating medium from CSV: {str(e)}")
        raise

def load_json(json_file):
    """
    Load data from a JSON file
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Dictionary containing the data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {json_file}")
        print(f"JSON error: {str(e)}")
        raise 

def validate_material_name(material_file, csv_file):
    """
    Validate that the material name in the JSON file matches the CSV filename
    
    Args:
        material_file: Path to the material JSON file
        csv_file: Path to the CSV file with wavelength and refractive index data
        
    Returns:
        True if the material name matches the CSV filename, False otherwise
        
    Raises:
        ValueError: If the material_name field is missing from the JSON file
    """
    material_data = load_material_data(material_file)
    
    # Check if material_name is in material file
    if "material_name" not in material_data:
        print(f"Error: material_name not defined in material file: {material_file}")
        print("Please ensure the material file includes a material_name field.")
        raise ValueError(f"material_name field missing from material file: {material_file}")
    
    # Get material name from JSON
    material_name = material_data["material_name"]
    
    # Get CSV basename without extension
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Check if they match
    if material_name != csv_basename:
        print(f"Warning: Material name mismatch between JSON ({material_name}) and CSV filename ({csv_basename}).")
        print("The material_name in the JSON file should match the CSV filename (without extension).")
        return False
    
    return True 