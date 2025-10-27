import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def fit_exponential_decay(x, y, from_max=True, exclude_after=None, start_from=None):
    """
    Fit an exponential decay curve to field intensity data
    
    Args:
        x: x-coordinates (position)
        y: y-coordinates (intensity)
        from_max: If True, fit starts from the maximum intensity point
                  If False, fit starts from the beginning of the array
        exclude_after: If provided, exclude data beyond this x-value
        start_from: If provided, fit starts from this x-value (overrides from_max)
    
    Returns:
        params: Dictionary with fit parameters (A, alpha, offset)
        x_fit: x-coordinates for the fitted region
        y_fit: fitted curve values
    """
    # Ensure data is numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    if start_from is not None:
        # Find index where x >= start_from
        start_idx = np.argmax(x >= start_from)
        
        # Use data from the specified starting point
        x_decay = x[start_idx:]
        y_decay = y[start_idx:]
    elif from_max:
        # Find index of maximum intensity
        max_idx = np.argmax(y)
        
        # Only use data after the maximum
        x_decay = x[max_idx:]
        y_decay = y[max_idx:]
    else:
        # Use all data
        x_decay = x
        y_decay = y
    
    # Exclude data beyond a given x-value if specified
    if exclude_after is not None:
        mask = x_decay <= exclude_after
        x_decay = x_decay[mask]
        y_decay = y_decay[mask]
    
    # Define exponential decay function without offset: A*exp(-alpha*x)
    def exp_decay(x, A, alpha):
        return A * np.exp(-alpha * x)
    
    # Initial parameter guesses: amplitude = max(y_decay), alpha = 0.1
    p0 = [np.max(y_decay), 0.1]
    
    try:
        # Perform curve fitting (no offset term)
        popt, pcov = curve_fit(exp_decay, x_decay - x_decay[0], y_decay, p0=p0, maxfev=5000)
        
        # Extract fit parameters and fix offset to zero
        A, alpha = popt
        offset = 0.0
        
        # Generate fitted curve
        x_fit = x_decay
        y_fit = exp_decay(x_fit - x_fit[0], A, alpha)
        
        # Return parameters (including zero offset) and fitted curve
        params = {
            'A': A,
            'alpha': alpha,
            'offset': offset
        }
        return params, x_fit, y_fit
    except (RuntimeError, ValueError) as e:
        print(f"Fitting error: {str(e)}")
        return None, x_decay, y_decay

def extract_farfield_peaks(theta, ff_data, height_threshold=0.1, distance_min=10, sort_by_angle=False):
    """
    Extract peak angles and heights from far-field intensity data without plotting
    
    Args:
        theta: Array of angle values in radians
        ff_data: Dictionary with field components {Etheta, Ephi, Er}
        height_threshold: Threshold for peak detection (fraction of max intensity)
        distance_min: Minimum distance between peaks (in array indices)
        sort_by_angle: If True, sort peaks by angle from most positive to most negative.
                       If False (default), sort peaks by height (descending).
    
    Returns:
        peaks_list: List of peak angles for each frequency (in degrees)
        heights_list: List of peak heights for each frequency (normalized)
    """
    # Convert theta to degrees for output
    angles_deg = np.rad2deg(theta)
    
    # Store peak data
    peaks_list = []
    heights_list = []
    
    # Process each frequency
    num_freq = len(ff_data['Etheta'])
    for i in range(num_freq):
        # Extract field components for this frequency
        Etheta = np.abs(ff_data['Etheta'][i])
        Ephi = np.abs(ff_data['Ephi'][i])
        Er = np.abs(ff_data['Er'][i])
        
        # Calculate total intensity
        I_far = Etheta**2 + Ephi**2 + Er**2
        I_far_norm = np.squeeze(I_far / np.max(I_far))
        
        # Find peaks
        peaks, peak_props = find_peaks(I_far_norm, height=height_threshold, distance=distance_min)
        peak_angles_deg = angles_deg[peaks]
        peak_heights = peak_props['peak_heights']
        
        # Sort peaks based on specified criteria
        if sort_by_angle:
            # Sort by angle (most positive to most negative)
            sorted_indices = np.argsort(peak_angles_deg)[::-1]
        else:
            # Sort by height (descending)
            sorted_indices = np.argsort(peak_heights)[::-1]
            
        peak_angles_deg = peak_angles_deg[sorted_indices]
        peak_heights = peak_heights[sorted_indices]
        
        # Store for return value
        peaks_list.append(peak_angles_deg)
        heights_list.append(peak_heights)
    
    return peaks_list, heights_list

def calculate_power_window_fractions(theta, ff_data, peak_angles, window_width_deg=10):
    """
    Calculate fraction of power within windows around peak angles
    
    Args:
        theta: Array of angle values in radians
        ff_data: Dictionary with field components {Etheta, Ephi, Er}
        peak_angles: List of lists of peak angles (in degrees) for each frequency
        window_width_deg: Width of the window around each peak (in degrees)
        
    Returns:
        power_fractions: List of dictionaries containing power fractions for each frequency
    """
    # Convert theta to degrees
    angles_deg = np.rad2deg(theta)
    
    # Store power fractions
    power_fractions = []
    
    # Process each frequency
    num_freq = len(ff_data['Etheta'])
    for i in range(num_freq):
        # Extract field components for this frequency
        Etheta = np.abs(ff_data['Etheta'][i])
        Ephi = np.abs(ff_data['Ephi'][i])
        Er = np.abs(ff_data['Er'][i])
        
        # Calculate intensity
        I_far = Etheta**2 + Ephi**2 + Er**2
        I_far_norm = np.squeeze(I_far / np.max(I_far))
        
        # Calculate total power
        total_power = np.trapz(I_far_norm, angles_deg)
        
        # Initialize dictionary for this frequency
        freq_power_fractions = {}
        
        # For each peak, calculate power within window
        for j, peak_angle in enumerate(peak_angles[i]):
            window_min = peak_angle - window_width_deg/2
            window_max = peak_angle + window_width_deg/2
            
            # Find indices within window
            window_idx = np.where((angles_deg >= window_min) & (angles_deg <= window_max))[0]
            
            if window_idx.size:
                # Calculate power within window
                window_power = np.trapz(I_far_norm[window_idx], angles_deg[window_idx])
                power_frac = window_power / total_power if total_power > 0 else 0
                freq_power_fractions[f'peak_{j+1}_power_fraction'] = power_frac
        
        power_fractions.append(freq_power_fractions)
    
    return power_fractions

def combine_monitor_data(field_data, flux_data):
    """
    Combine field and flux monitor data for comprehensive analysis
    
    Args:
        field_data: Dictionary of field monitor data
        flux_data: Dictionary of flux monitor data
        
    Returns:
        combined_data: Dictionary with combined field and flux information
    """
    # Create combined data structure
    combined_data = {
        'field': field_data,
        'flux': flux_data
    }
    
    return combined_data 