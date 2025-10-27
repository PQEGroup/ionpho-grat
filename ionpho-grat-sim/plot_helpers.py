import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td

def plot_material_index_profile(sim, freqs, figsize=(8, 6), title=None):
    """Plot refractive index vs Z (μm) for each frequency in the simulation.
    
    Args:
        sim: Tidy3D simulation object
        freqs: Frequencies to compute refractive indices at (in Hz)
        figsize: Figure size (default: (8, 6))
        title: Custom title for the plot (default: None)
    
    Returns:
        fig, ax: Figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Convert freqs to numpy array if not already
    if np.isscalar(freqs):
        freqs = np.array([freqs])
    elif not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)

    # Get simulation bounds
    bounds = sim.geometry.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]

    # Create a uniform grid of z points
    z_points = np.linspace(z_min, z_max, 1000)

    # Dictionary to track unique mediums and their descriptive names
    medium_names = {}
    medium_indices = {}

    # For each frequency, compute the index profile
    for i, f in enumerate(freqs):
        # Convert to wavelength in μm for labeling
        wl_um = td.C_0 / f 
        label = f"{wl_um:.3f} μm"

        # Compute refractive index at each z point
        n_profile = []
        for z in z_points:
            # Find the medium at this z position
            med = None
            for struct in sim.structures:
                geom = getattr(struct, 'geometry', None)
                if geom and hasattr(geom, 'center') and hasattr(geom, 'size'):
                    z_min_struct = geom.center[2] - geom.size[2] / 2
                    z_max_struct = geom.center[2] + geom.size[2] / 2
                    if z_min_struct <= z <= z_max_struct:
                        med = struct.medium
                        # No break to allow later structures to override earlier ones

            # Compute refractive index
            if med:
                if hasattr(med, 'eps_model'):
                    eps = med.eps_model(f)
                    n_val = np.sqrt(np.real(eps))
                    source = "Dispersive"
                else:
                    n_val = np.sqrt(float(med.permittivity.real))
                    source = "Constant"
                    
                # Track medium name and index
                if med not in medium_names:
                    if hasattr(med, 'name'):
                        medium_names[med] = f"{med.name} ({source})"
                    else:
                        medium_names[med] = f"Medium_{len(medium_names)+1} ({source})"
                    medium_indices[med] = n_val
            else:
                n_val = 1.0  # Default to air if no medium found
            n_profile.append(n_val)

        # Plot the index profile for this frequency
        ax.plot(z_points, n_profile, label=label, linewidth=2)

    # Add medium name information to the plot
    for med, name in medium_names.items():
        n_val = medium_indices.get(med, 0)
        if n_val > 0:
            # Find a suitable z position for the label
            for struct in sim.structures:
                if struct.medium == med:
                    z_center = struct.geometry.center[2]
                    ax.text(z_center, n_val+0.05, name, ha='center', va='bottom', 
                            fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                    break

    # Set labels and title
    ax.set_xlabel('Z (μm)', fontsize=12)
    ax.set_ylabel('Refractive Index', fontsize=12)
    plot_title = title if title else 'Refractive Index Profile vs Z'
    ax.set_title(plot_title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend(title='Wavelength', loc='best')
    
    plt.tight_layout()
    return fig, ax

def plot_mode_source_spectrum(source_time, freqs=None, figsize=(8, 6)):
    """Plot the spectrum of the mode source
    
    Args:
        source_time: Tidy3D source_time object (typically GaussianPulse)
        freqs: Optional list of frequencies to highlight (in Hz).
               Can be a single value, an array, or None.
        figsize: Figure size (default: (8, 6))
    
    Returns:
        fig, ax: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure source_time is a GaussianPulse
    if not isinstance(source_time, td.GaussianPulse):
        print(f"Warning: Expected GaussianPulse, got {type(source_time)}. Plot may not be accurate.")
    
    # Get center frequency and width
    f0 = source_time.freq0
    df = source_time.fwidth
    
    # Generate frequency array for plotting (3 standard deviations each side)
    f_min = f0 - 3*df
    f_max = f0 + 3*df
    f_plot = np.linspace(f_min, f_max, 1000)
    
    # Calculate Gaussian spectrum
    spectrum = np.exp(-((f_plot - f0) / df)**2)
    
    # Convert frequencies to wavelengths for display
    lambda0 = td.C_0 / f0 
    dlambda = (td.C_0 / (f0 - df) - td.C_0 / (f0 + df)) 
    
    # Plot source spectrum
    ax.plot(f_plot, spectrum, 'b-', lw=2, label='Source Spectrum')
    
    # Mark the center frequency
    ax.axvline(f0, color='r', linestyle='--', label=f'Center: {lambda0:.3f} μm')
    
    # Mark the frequencies of interest if provided
    if freqs is not None:
        # Convert a single frequency value to a list
        if np.isscalar(freqs):
            freqs = [freqs]
        
        for f in freqs:
            lambda_um = td.C_0 / f
            ax.axvline(f, color='g', linestyle='-', alpha=0.3)
            ax.annotate(f'{lambda_um:.3f} μm', 
                       xy=(f, 0.1), 
                       xytext=(0, -10),
                       textcoords='offset points',
                       ha='center', va='top', 
                       fontsize=8, color='green')
    
    # Add annotations to show bandwidth info
    bandwidth_text = (
        f"Center: {lambda0:.3f} μm ({f0:.3e} Hz)\n"
        f"Bandwidth: {dlambda:.3f} μm ({df:.3e} Hz)"
    )
    ax.text(0.02, 0.95, bandwidth_text,
           transform=ax.transAxes,
           fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
           verticalalignment='top')
    
    # Add axis labels and grid
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.set_title('Mode Source Spectrum', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(0, 1.05)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    return fig, ax

def plot_combined_view(sim, source_time, freqs=None, plot_geometry=True, figsize=(14, 7)):
    """Create a combined plot with index profile, source spectrum, and optional geometry
    
    Args:
        sim: Tidy3D simulation object
        source_time: Tidy3D source_time object (typically GaussianPulse)
        freqs: Optional list of frequencies to highlight (in Hz)
        plot_geometry: Whether to include a geometry plot (default: True)
        figsize: Figure size (default: (14, 7))
    
    Returns:
        fig: Figure object
    """
    if plot_geometry:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2)
        
        # Index profile (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        # Source spectrum (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        # Geometry (bottom, spans both columns)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Plot index profile
        _plot_index_profile_on_ax(sim, freqs, ax1)
        
        # Plot source spectrum
        _plot_source_spectrum_on_ax(source_time, freqs, ax2)
        
        # Plot geometry
        sim.plot(y=0, ax=ax3)
        ax3.set_title('Simulation Geometry (y=0 cross-section)', fontsize=14)
        ax3.set_xlabel('X (μm)', fontsize=12)
        ax3.set_ylabel('Z (μm)', fontsize=12)
    else:
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 2)
        
        # Index profile (left)
        ax1 = fig.add_subplot(gs[0, 0])
        # Source spectrum (right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot index profile
        _plot_index_profile_on_ax(sim, freqs, ax1)
        
        # Plot source spectrum
        _plot_source_spectrum_on_ax(source_time, freqs, ax2)
    
    plt.tight_layout()
    return fig

def _plot_index_profile_on_ax(sim, freqs, ax):
    """Helper function to plot index profile on a given axis"""
    # Convert freqs to numpy array if not already
    if np.isscalar(freqs):
        freqs = np.array([freqs])
    elif not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)
    
    # Get simulation bounds
    bounds = sim.geometry.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]

    # Create a uniform grid of z points
    z_points = np.linspace(z_min, z_max, 1000)

    # Dictionary to track unique mediums and their descriptive names
    medium_names = {}
    medium_indices = {}

    # For each frequency, compute the index profile
    for i, f in enumerate(freqs):
        # Convert to wavelength in μm for labeling
        wl_um = td.C_0 / f 
        label = f"{wl_um:.3f} μm"

        # Compute refractive index at each z point
        n_profile = []
        for z in z_points:
            # Find the medium at this z position
            med = None
            for struct in sim.structures:
                geom = getattr(struct, 'geometry', None)
                if geom and hasattr(geom, 'center') and hasattr(geom, 'size'):
                    z_min_struct = geom.center[2] - geom.size[2] / 2
                    z_max_struct = geom.center[2] + geom.size[2] / 2
                    if z_min_struct <= z <= z_max_struct:
                        med = struct.medium

            # Compute refractive index
            if med:
                if hasattr(med, 'eps_model'):
                    eps = med.eps_model(f)
                    n_val = np.sqrt(np.real(eps))
                else:
                    n_val = np.sqrt(float(med.permittivity.real))
                    
                # Track medium name and index
                if med not in medium_names:
                    if hasattr(med, 'name'):
                        medium_names[med] = med.name
                    else:
                        medium_names[med] = f"Medium_{len(medium_names)+1}"
                    medium_indices[med] = n_val
            else:
                n_val = 1.0  # Default to air if no medium found
            n_profile.append(n_val)

        # Plot the index profile for this frequency
        ax.plot(z_points, n_profile, label=label, linewidth=2)

    # Set labels and title
    ax.set_xlabel('Z (μm)', fontsize=12)
    ax.set_ylabel('Refractive Index', fontsize=12)
    ax.set_title('Refractive Index Profile', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend(title='Wavelength', loc='best')

def _plot_source_spectrum_on_ax(source_time, freqs, ax):
    """Helper function to plot source spectrum on a given axis"""
    # Get center frequency and width
    f0 = source_time.freq0
    df = source_time.fwidth
    
    # Generate frequency array for plotting
    f_min = f0 - 3*df
    f_max = f0 + 3*df
    f_plot = np.linspace(f_min, f_max, 1000)
    
    # Calculate Gaussian spectrum
    spectrum = np.exp(-((f_plot - f0) / df)**2)
    
    # Convert frequencies to wavelengths for display
    lambda0 = td.C_0 / f0 
    
    # Plot source spectrum
    ax.plot(f_plot, spectrum, 'b-', lw=2, label='Source Spectrum')
    
    # Mark the center frequency
    ax.axvline(f0, color='r', linestyle='--', label=f'Center: {lambda0:.3f} μm')
    
    # Mark the frequencies of interest if provided
    if freqs is not None:
        # Convert a single frequency value to a list
        if np.isscalar(freqs):
            freqs = [freqs]
            
        for f in freqs:
            lambda_um = td.C_0 / f 
            ax.axvline(f, color='g', linestyle='-', alpha=0.3)
            ax.annotate(f'{lambda_um:.3f} μm', 
                       xy=(f, 0.1), 
                       xytext=(0, -10),
                       textcoords='offset points',
                       ha='center', va='top', 
                       fontsize=8, color='green')
    
    # Add axis labels and grid
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.set_title('Source Spectrum', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(0, 1.05)
    
    ax.legend(loc='upper right') 