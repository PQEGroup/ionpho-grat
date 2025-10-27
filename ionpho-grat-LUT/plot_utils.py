import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import tidy3d as td 
import os
import sys
import json
from matplotlib.patches import Rectangle

# Plotting utilities for post-processing

def quick_plot(x, y, labels=None, title=None, figsize=(6,4), style='-k', markers=None, annotations=None, **kwargs):
    """Create a plot with minimal code

    Args:
        x: x-data (or list of arrays)
        y: y-data (or list of arrays)
        labels: Optional list of labels
        title: Plot title
        figsize: Tuple for figure size
        style: Line style or list of styles
        markers: Dict or tuple for highlighting points
        annotations: Dict with 'points', 'texts', 'props' for annotating
        **kwargs: Additional axis settings passed to ax.set()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize input to lists: if x or y is not a list/tuple, wrap in list
    if isinstance(x, (list, tuple)):
        x_list = x
    else:
        x_list = [x]
    if isinstance(y, (list, tuple)):
        y_list = y
    else:
        y_list = [y]
    style_list = style if isinstance(style, (list, tuple)) else [style]
    labels_list = labels if isinstance(labels, (list, tuple)) else ([labels] if labels else None)

    # Plot data
    for idx, (xi, yi) in enumerate(zip(x_list, y_list)):
        s = style_list[idx] if idx < len(style_list) else style_list[0]
        lbl = labels_list[idx] if labels_list and idx < len(labels_list) else None
        ax.plot(xi, yi, s, label=lbl)

    # Plot markers
    if markers:
        if isinstance(markers, dict):
            ax.plot(markers['x'], markers['y'], markers.get('style', 'ro'))
        else:
            ax.plot(markers[0], markers[1], 'ro')

    # Plot annotations
    if annotations:
        pts = annotations.get('points', [])
        texts = annotations.get('texts', [])
        props = annotations.get('props', {})
        for (xp, yp), txt in zip(pts, texts):
            ax.annotate(txt, (xp, yp), **props)

    # Axis settings
    if kwargs:
        ax.set(**kwargs)
    if title:
        ax.set_title(title)
    if labels_list:
        ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig, ax 

def plot_power_distribution(power_up, power_down, power_end, power_start, freqs=None):
    """Plot per-frequency power distribution as a grouped bar chart
    
    Args:
        power_up: Power in upward direction per frequency
        power_down: Power in downward direction per frequency
        power_end: Power at grating end per frequency
        power_start: Input power per frequency
        freqs: Optional frequency values in Hz
    
    Returns:
        fig, ax: The figure and axes objects
    """
    # Set up x-coordinates and bar widths
    nf = len(power_up)
    x = np.arange(nf)
    width = 0.2
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each power component as grouped bars
    ax.bar(x - 1.5*width, power_up, width, label='Upward')
    ax.bar(x - 0.5*width, power_down, width, label='Downward')
    ax.bar(x + 0.5*width, power_end, width, label='End')
    ax.bar(x + 1.5*width, power_start, width, label='Input')
    
    # Set x-ticks to wavelengths if frequencies provided
    if freqs is not None:
        try:
            import tidy3d as td
            # Compute wavelengths (m) and convert to nm
            lam_nm = td.C_0 / np.array(freqs) * 1e3
        except ImportError:
            # Fallback speed of light
            c = 299792458
            lam_nm = c / np.array(freqs) * 1e3
        labels = [f"{l:.1f} nm" for l in lam_nm]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    
    # Add labels and grid
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Power (arb. units)')
    ax.set_title('Per-frequency power distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_farfield_intensity(theta, ff_data, freqs=None, height_threshold=0.1, distance_min=5):
    """Plot far-field intensity vs. angle for multiple frequencies
    
    Args:
        theta: Array of angle values in radians
        ff_data: Dictionary with field components {Etheta, Ephi, Er}
        freqs: Optional array of frequency values in Hz
        height_threshold: Threshold for peak detection
        distance_min: Minimum distance between peaks
    
    Returns:
        fig, axes: The figure and axes objects
        peaks_list: List of peak angles for each frequency
        heights_list: List of peak heights for each frequency
    """
    from scipy.signal import find_peaks
    
    # Get the number of frequencies
    num_freq = len(ff_data['Etheta'])
    freqs = np.arange(num_freq) if freqs is None else freqs
    
    # Convert theta to degrees for plotting
    angles_deg = np.rad2deg(theta)
    
    # Store peak data
    peaks_list = []
    heights_list = []
    
    # Create subplots for each frequency
    fig, axes = plt.subplots(1, num_freq, figsize=(num_freq*6, 4), squeeze=False)
    fig.suptitle('Far-field Intensity vs. Angle', fontsize=16)
    
    # Process each frequency
    for i in range(num_freq):
        # Extract field components for this frequency
        Etheta = np.abs(ff_data['Etheta'][i])
        Ephi = np.abs(ff_data['Ephi'][i])
        Er = np.abs(ff_data['Er'][i])
        
        # Calculate intensity
        I_far = Etheta**2 + Ephi**2 + Er**2
        I_far_norm = np.squeeze(I_far / np.max(I_far))
        
        # Find peaks
        peaks, peak_props = find_peaks(I_far_norm, height=height_threshold, distance=distance_min)
        peak_angles_deg = angles_deg[peaks]
        peak_heights = peak_props['peak_heights']
        
        # Store for return value
        peaks_list.append(peak_angles_deg)
        heights_list.append(peak_heights)
        
        # Plot on the appropriate subplot
        ax = axes[0, i]
        ax.plot(angles_deg, I_far_norm)
        ax.plot(peak_angles_deg, I_far_norm[peaks], 'ro')
        
        # Add peak labels
        for j, (ang, height) in enumerate(zip(peak_angles_deg, peak_heights)):
            ax.annotate(f"{ang:.1f}°", (ang, height), 
                        textcoords='offset points', xytext=(0,5), ha='center')
        
        # Set labels and title
        # Convert frequency to wavelength label
        try:
            import tidy3d as td
            wl_nm = td.C_0 / freqs[i] * 1e3
        except ImportError:
            c = 299792458
            wl_nm = c / freqs[i] * 1e3
        freq_label = f"{wl_nm:.1f} nm"
        ax.set_title(freq_label)
        ax.set_xlabel('Angle (°)')
        ax.set_ylabel('Normalized Intensity')
        ax.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the suptitle
    plt.show()
    
    return fig, axes, peaks_list, heights_list 

def plot_farfield_comparison(theta, ff_data, freqs=None, upward_frac=None, height_threshold=0.5, distance_min=15):
    """Plot all far-field intensity curves on a single axis with legend for each wavelength."""
    from scipy.signal import find_peaks
    fig, ax = plt.subplots(figsize=(8, 6))
    angles_deg = np.rad2deg(theta)
    # Determine wavelength labels if frequencies provided
    if freqs is not None:
        try:
            import tidy3d as td
            lam_nm = td.C_0 / np.array(freqs) * 1e3 # some weird convention between um and m so a factor of 1e6 difference to get the right number in nm
        except Exception:
            c = 299792458
            lam_nm = c / np.array(freqs) * 1e3
        labels = [f"{l:.1f} nm" for l in lam_nm]
    else:
        labels = [f"Freq {i}" for i in range(len(ff_data['Etheta']))]
    # Compute intensities and global maximum for normalization
    I_fars = []
    for Etheta, Ephi, Er in zip(ff_data['Etheta'], ff_data['Ephi'], ff_data['Er']):
        I_far = Etheta**2 + Ephi**2 + Er**2
        I_fars.append(np.squeeze(I_far))
    global_max = max(np.max(I_far) for I_far in I_fars)
    
    # Plot each frequency's curve normalized by global maximum and annotate peaks
    shading_list = []
    boundary_list = []
    for i, I_far in enumerate(I_fars):
        # For plotting: normalize by global max for consistent visualization
        I_far_norm = I_far / global_max
        
        # For peak finding: normalize by this wavelength's max to find appropriate peaks
        I_far_peak_norm = I_far / np.max(I_far)
        
        # Find peaks using wavelength-specific normalization
        peaks, props = find_peaks(I_far_peak_norm, height=height_threshold, distance=distance_min)
        peak_angles = angles_deg[peaks]
        
        # But use the global-normalized values for the plot heights
        peak_heights = I_far_norm[peaks]

        # Update label to include upward fraction if provided
        label = labels[i]
        if upward_frac is not None:
            label = f"{label} ({upward_frac[i]*100:.1f}% up)"

        # Plot curve and get its color
        curve, = ax.plot(angles_deg, I_far_norm, label=label)
        ax.plot(peak_angles, peak_heights, 'ro')
        
        # Calculate and highlight power window
        window_deg = 5  # deg
        for ang, h in zip(peak_angles, peak_heights):
            window_min = ang - window_deg
            window_max = ang + window_deg
            # Shaded region
            span = ax.axvspan(window_min, window_max, color=curve.get_color(), alpha=0.2)
            shading_list.append(span)
            # Boundary lines
            ln1 = ax.axvline(window_min, color=curve.get_color(), linestyle='--', linewidth=1)
            ln2 = ax.axvline(window_max, color=curve.get_color(), linestyle='--', linewidth=1)
            boundary_list.extend([ln1, ln2])
            # Annotate with angle and power percentage
            window_idx = np.where((angles_deg >= window_min) & (angles_deg <= window_max))[0]
            if window_idx.size:
                window_power = np.trapz(I_far_norm[window_idx], angles_deg[window_idx])
                total_power = np.trapz(I_far_norm, angles_deg)
                power_frac = window_power / total_power * 100 if total_power > 0 else 0
                ax.annotate(f"{ang:.1f}°\n{power_frac:.1f}%", (ang, h), textcoords='offset points', xytext=(0,5), ha='center')
            else:
                ax.annotate(f"{ang:.1f}°", (ang, h), textcoords='offset points', xytext=(0,5), ha='center')

    ax.set_xlabel('Angle (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Far-field Intensity Comparison')
    # Force legend to upper right with transparency
    legend = ax.legend(loc='upper right')
    legend.get_frame().set_alpha(0.3)
    ax.grid(True)

    # Position toggle button at bottom-left just below the axis
    pos = ax.get_position()
    btn_w = pos.width * 0.2
    btn_h = pos.height * 0.05  # slightly taller for readability
    btn_x = pos.x0
    btn_y = pos.y0 - btn_h - 0.05  # small gap below axis
    btn_ax = fig.add_axes([btn_x, btn_y, btn_w, btn_h], facecolor='none')
    btn_ax.set_axis_off()
    check = CheckButtons(btn_ax, ['power window fraction'], [True])
    def _toggle(label):
        # Toggle visibility of shaded regions and boundary lines
        for span in shading_list:
            span.set_visible(not span.get_visible())
        for ln in boundary_list:
            ln.set_visible(not ln.get_visible())
        fig.canvas.draw_idle()
    check.on_clicked(_toggle)
    plt.show()
    return fig, ax 

# Add exponential-decay subplot plotting for air and waveguide fits
def plot_decay_fits(wavelengths, x_list, I_list, fit_params, x_fit_list, y_fit_list, title):
    """Plot exponential-decay fits for multiple wavelengths."""
    num_freqs = len(wavelengths)
    
    # Create subplots with proper handling
    if num_freqs == 1:
        # For single frequency, create a single subplot
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        axs = [ax]
    else:
        # For multiple frequencies, create a grid
        num_rows = (num_freqs + 1) // 2
        fig, axs = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
        # Always flatten to handle both 1D and 2D cases
        axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    fig.suptitle(title, fontsize=16)

    for i in range(num_freqs):
        ax = axs[i]
        wl = wavelengths[i]
        ax.set_title(f'λ = {wl:.1f} nm')
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Electric Field Intensity |E|²')
        # Plot data
        data_line = ax.plot(x_list[i], I_list[i], 'k', label='Data')[0]
        # Plot fit if available
        if i in fit_params:
            alpha = fit_params[i]['alpha']
            x_fit = x_fit_list[i]
            y_fit = y_fit_list[i]
            ax.plot(x_fit, y_fit, '--', label=f'Fit (α={alpha:.3f} μm⁻¹)')
            # Shade fit region
            ax.axvspan(x_fit[0], x_fit[-1], color=data_line.get_color(), alpha=0.2)
            ax.text(0.05, 0.95, f'α = {alpha:.4f} μm⁻¹', transform=ax.transAxes,
                    va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.grid(True)
        ax.legend()

    # Hide any extra subplots
    for j in range(num_freqs, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()
    return fig, axs

# Add phase-evolution plotting
def plot_phase_evolution(wavelengths, x_neff, phase_list, phase_unwrap_list, n_eff_values):
    """Plot phase evolution and unwrapped phase, computing effective index labels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Waveguide Mode Phase')
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('Phase (rad)')

    for i, wl in enumerate(wavelengths):
        ax.plot(x_neff, phase_list[i], '-', label=f'λ={wl:.1f}nm (phase)')
        ax.plot(x_neff, phase_unwrap_list[i], '--', label=f'λ={wl:.1f}nm (unwrapped), n_eff={n_eff_values[i]:.4f}')

    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    return fig, ax 

def plot_simulation_geometry(sim, material, material_file=None, l_grat=None, gratper=None, gratDC=None, N=None, 
                           wghb=None, boxt=None, toxt=None, tair=None, tSi=None, wg_w=None, mode_solver=None,
                           is_dual_waveguide=False, wght=None, inth=None):
    """Plot the simulation geometry with material annotations
    
    Args:
        sim: Tidy3D simulation object
        material: Material name
        material_file: Optional path to material JSON file
        l_grat: Grating length
        gratper: Grating period
        gratDC: Grating duty cycle
        N: Number of periods
        wghb: Bottom waveguide height
        boxt: Bottom oxide thickness
        toxt: Top oxide thickness
        tair: Air layer thickness
        tSi: Silicon substrate thickness
        wg_w: Waveguide width
        mode_solver: Mode solver object containing source plane info
        is_dual_waveguide: Whether this is a dual waveguide configuration (default: False)
        wght: Top waveguide height (for dual waveguide)
        inth: Intermediate layer height between waveguides (for dual waveguide)
    
    Returns:
        fig, ax: The figure and axes objects
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the simulation geometry
    sim.plot(y=0, lw=1, edgecolor='k', ax=ax)
    
    # Get the simulation bounds for annotations
    bbox = sim.geometry.bounds
    xmin, xmax = bbox[0][0], bbox[1][0]
    zmin, zmax = bbox[0][2], bbox[1][2]
    
    # Add source if mode_solver is provided
    if mode_solver is not None:
        src_plane = mode_solver.plane
        src_center = src_plane.center
        src_size = src_plane.size
        
        # Create a rectangle for the source with thinner lines
        src_rect = plt.Rectangle(
            (src_center[0] - src_size[0]/2, src_center[2] - src_size[2]/2),
            src_size[0], src_size[2],
            linewidth=1, edgecolor='r', facecolor='none', linestyle='--'
        )
        ax.add_patch(src_rect)
        
        # Label the source below the line
        ax.text(src_center[0], src_center[2] - src_size[2]/2 - 0.2, "Source",
               color='r', ha='center', va='top', fontsize=8)
    
    # Highlight the waveguide region(s) with more prominent color
    if wghb is not None:
        # Add a slight highlight to the waveguide region
        waveguide_color = 'orange'
        alpha_value = 0.2
        
        if is_dual_waveguide and wght is not None and inth is not None:
            # Draw bottom waveguide
            if l_grat is not None:
                # Pre-grating bottom waveguide
                wg_rect_pre_bottom = plt.Rectangle(
                    (xmin, 0),
                    abs(xmin),
                    wghb,
                    linewidth=0, facecolor=waveguide_color, alpha=alpha_value
                )
                ax.add_patch(wg_rect_pre_bottom)
                
                # Post-grating bottom waveguide
                wg_rect_post_bottom = plt.Rectangle(
                    (l_grat, 0),
                    (xmax - l_grat),
                    wghb,
                    linewidth=0, facecolor=waveguide_color, alpha=alpha_value
                )
                ax.add_patch(wg_rect_post_bottom)
                
                # Pre-grating top waveguide
                wg_rect_pre_top = plt.Rectangle(
                    (xmin, wghb + inth),
                    abs(xmin),
                    wght,
                    linewidth=0, facecolor=waveguide_color, alpha=alpha_value
                )
                ax.add_patch(wg_rect_pre_top)
                
                # Post-grating top waveguide
                wg_rect_post_top = plt.Rectangle(
                    (l_grat, wghb + inth),
                    (xmax - l_grat),
                    wght,
                    linewidth=0, facecolor=waveguide_color, alpha=alpha_value
                )
                ax.add_patch(wg_rect_post_top)
            
            # Add label for bottom waveguide
            ax.text(l_grat/4 if l_grat else 0, wghb/2, 
                    f"Bottom WG",
                    color='darkblue', ha='center', va='center', fontsize=8, 
                    fontweight='bold')
            
            # Add label for top waveguide
            ax.text(l_grat/4 if l_grat else 0, wghb + inth + wght/2, 
                    f"Top WG",
                    color='darkblue', ha='center', va='center', fontsize=8, 
                    fontweight='bold')
        else:
            # Single waveguide - pre-grating waveguide
            if l_grat is not None:
                # Pre-grating waveguide
                wg_rect_pre = plt.Rectangle(
                    (xmin, 0),
                    abs(xmin),
                    wghb,
                    linewidth=0, facecolor=waveguide_color, alpha=alpha_value
                )
                ax.add_patch(wg_rect_pre)
                
                # Post-grating waveguide
                wg_rect_post = plt.Rectangle(
                    (l_grat, 0),
                    (xmax - l_grat),
                    wghb,
                    linewidth=0, facecolor=waveguide_color, alpha=alpha_value
                )
                ax.add_patch(wg_rect_post)
        
        # Combine waveguide material and grating info on the same line
        grat_info = ""
        if gratper is not None and gratDC is not None and N is not None:
            grat_info = f" - Period={gratper*1e3:.1f}nm, DC={gratDC:.2f}, N={N}"
            
        # Create prominent label for waveguide/grating material above the structure
        if is_dual_waveguide and wght is not None and inth is not None:
            label_height = wghb + inth + wght + 0.2
        else:
            label_height = wghb + 0.2
            
        ax.text(l_grat/4 if l_grat else 0, label_height, 
                f"{material}{grat_info}",
                color='darkblue', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Determine the x position for labels
    x_label_pos = xmax * 0.8  # Position for labels on the right
    
    # Add vertical arrows and labels for material regions on the right side
    if toxt is not None:
        # Top oxide layer (SiO2)
        arrow_x = x_label_pos
        # Draw a vertical line with arrow caps at both ends
        if is_dual_waveguide and wght is not None and inth is not None:
            label_ht = wghb+wght+inth
        else:
            label_ht = wghb
        ax.plot([arrow_x, arrow_x], [label_ht, toxt], 'b-', lw=1) # since for dual waveguide, wght and inth are 0 anyway
        ax.plot([arrow_x, arrow_x-0.2], [label_ht, label_ht+0.2], 'b-', lw=1)    # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.2], [label_ht, label_ht+0.2], 'b-', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [toxt+label_ht, toxt+label_ht-0.2], 'b-', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.2], [toxt+label_ht, toxt+label_ht-0.2], 'b-', lw=1)
        
        # Text position for measurement
        ax.text(arrow_x + 0.2, toxt/2, f"toxt={toxt}μm", ha='left', va='center', fontsize=8, color='blue')
        
        # Text position for material
        ax.text(arrow_x - 2.0, toxt/2, "SiO2", ha='left', va='center', fontsize=9)
    
    if boxt is not None:
        # Bottom oxide layer (SiO2)
        arrow_x = x_label_pos
        # Draw a vertical line with arrow caps at both ends
        ax.plot([arrow_x, arrow_x], [-boxt, 0], 'b-', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [-boxt, -boxt+0.2], 'b-', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.2], [-boxt, -boxt+0.2], 'b-', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [0, -0.2], 'b-', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.2], [0, -0.2], 'b-', lw=1)
        
        # Text position for measurement
        ax.text(arrow_x + 0.2, -boxt/2, f"boxt={boxt}μm", ha='left', va='center', fontsize=8, color='blue')
        
        # Text position for material
        ax.text(arrow_x - 2.0, -boxt/2, "SiO2", ha='left', va='center', fontsize=9)
    
    if boxt is not None and tSi is not None:
        # Silicon substrate
        arrow_x = x_label_pos
        # Draw a vertical line with arrow caps at both ends
        ax.plot([arrow_x, arrow_x], [-boxt-tSi, -boxt], 'gray', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [-boxt-tSi, -boxt-tSi+0.2], 'gray', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.2], [-boxt-tSi, -boxt-tSi+0.2], 'gray', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [-boxt, -boxt-0.2], 'gray', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.2], [-boxt, -boxt-0.2], 'gray', lw=1)
        
        # Text position for measurement
        ax.text(arrow_x + 0.2, -boxt-tSi/2, f"tSi={tSi}μm", ha='left', va='center', fontsize=8, color='gray')
        
        # Text position for material
        ax.text(arrow_x - 2.0, -boxt-tSi/2, "Si", ha='left', va='center', fontsize=9)
    
    # Label the air region if thickness provided
    if tair is not None and toxt is not None:
        arrow_x = x_label_pos
        # Calculate z position based on waveguide configuration
        if is_dual_waveguide and wght is not None and inth is not None:
            z_start = toxt + wghb + inth + wght
        else:
            z_start = toxt + wghb
            
        # Draw a vertical line with arrow caps at both ends
        ax.plot([arrow_x, arrow_x], [z_start, z_start+tair], 'lightblue', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [z_start, z_start+0.2], 'lightblue', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.2], [z_start, z_start+0.2], 'lightblue', lw=1)
        ax.plot([arrow_x, arrow_x-0.2], [z_start+tair, z_start+tair-0.2], 'lightblue', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.2], [z_start+tair, z_start+tair-0.2], 'lightblue', lw=1)
        
        # Text position for measurement
        ax.text(arrow_x + 0.2, z_start+tair/2, f"tair={tair}μm", ha='left', va='center', fontsize=8, color='lightblue')
        
        # Text position for material
        ax.text(arrow_x - 2.0, z_start+tair/2, "Air", ha='left', va='center', fontsize=9)
    
    # Add dimension indicators for dual waveguide if applicable
    if is_dual_waveguide and wght is not None and inth is not None and wghb is not None:
        arrow_x = x_label_pos - 1.0  # Slightly to the left of the main dimension indicators
        
        # Bottom waveguide height (wghb)
        ax.plot([arrow_x, arrow_x], [0, wghb], 'orange', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [0, 0.1], 'orange', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.1], [0, 0.1], 'orange', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [wghb, wghb-0.1], 'orange', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.1], [wghb, wghb-0.1], 'orange', lw=1)
        
        # Text for bottom waveguide
        ax.text(arrow_x + 0.2, wghb/2, f"wghb={wghb}μm", ha='left', va='center', fontsize=8, color='orange')
        
        # Intermediate layer height (inth)
        ax.plot([arrow_x, arrow_x], [wghb, wghb+inth], 'gray', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [wghb, wghb+0.1], 'gray', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.1], [wghb, wghb+0.1], 'gray', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [wghb+inth, wghb+inth-0.1], 'gray', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.1], [wghb+inth, wghb+inth-0.1], 'gray', lw=1)
        
        # Text for intermediate layer
        ax.text(arrow_x + 0.2, wghb+inth/2, f"inth={inth}μm", ha='left', va='center', fontsize=8, color='gray')
        
        # Top waveguide height (wght)
        ax.plot([arrow_x, arrow_x], [wghb+inth, wghb+inth+wght], 'orange', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [wghb+inth, wghb+inth+0.1], 'orange', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.1], [wghb+inth, wghb+inth+0.1], 'orange', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [wghb+inth+wght, wghb+inth+wght-0.1], 'orange', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.1], [wghb+inth+wght, wghb+inth+wght-0.1], 'orange', lw=1)
        
        # Text for top waveguide
        ax.text(arrow_x + 0.2, wghb+inth+wght/2, f"wght={wght}μm", ha='left', va='center', fontsize=8, color='orange')
    elif wghb is not None:
        # Single waveguide - show the waveguide height
        arrow_x = x_label_pos - 1.0
        ax.plot([arrow_x, arrow_x], [0, wghb], 'orange', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [0, 0.1], 'orange', lw=1)  # Bottom arrow
        ax.plot([arrow_x, arrow_x+0.1], [0, 0.1], 'orange', lw=1)
        ax.plot([arrow_x, arrow_x-0.1], [wghb, wghb-0.1], 'orange', lw=1)  # Top arrow
        ax.plot([arrow_x, arrow_x+0.1], [wghb, wghb-0.1], 'orange', lw=1)
        
        # Text for waveguide height
        ax.text(arrow_x + 0.2, wghb/2, f"wghb={wghb}μm", ha='left', va='center', fontsize=8, color='orange')
    
    # Label monitors with small text - position based on monitor type
    for i, monitor in enumerate(sim.monitors):
        if hasattr(monitor, 'center') and hasattr(monitor, 'size'):
            center = monitor.center
            size = monitor.size
            
            # Create thin lines for the monitor
            mon_rect = plt.Rectangle(
                (center[0] - size[0]/2, center[2] - size[2]/2),
                size[0], size[2],
                linewidth=0.5, edgecolor='green', facecolor='none', linestyle='-.'
            )
            ax.add_patch(mon_rect)
            
            # Get monitor name
            mon_name = monitor.name if hasattr(monitor, 'name') else f"Monitor {i+1}"
            
            # Position text based on monitor type
            if isinstance(monitor, td.FluxMonitor) or 'flux' in mon_name.lower():
                # For flux monitors, place text below
                ax.text(center[0], center[2] - size[2]/2 - 0.1, mon_name,
                      color='green', ha='center', va='top', fontsize=6)
            elif 'far_field' in mon_name.lower() or 'farfield' in mon_name.lower() or 'angle' in mon_name.lower():
                # For far field angle monitors, place text at the center
                ax.text(center[0], center[2]+0.3, mon_name,
                      color='green', ha='center', va='center', fontsize=6,
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.1))
            else:
                # For other field monitors, place text at the right end
                right_end_x = center[0] + size[0]/2
                ax.text(right_end_x + 0.2, center[2], mon_name,
                      color='green', ha='left', va='center', fontsize=6)
    
    # Add axis labels with units
    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('z (μm)', fontsize=12)
    
    # Add a title with configuration type
    if is_dual_waveguide and wght is not None and inth is not None:
        ax.set_title(f"Dual Waveguide Grating Coupler - {material}")
    else:
        ax.set_title(f"Grating Coupler Simulation - {material}")
    
    plt.tight_layout()
    
    return fig, ax

def plot_mode_fields(modes, wavelengths=None, labels=None, sim=None, wghb=None, wght=None, inth=None, is_dual_waveguide=False):
    """Plot the mode field profiles for the solved mode
    
    Args:
        modes: Tidy3D modes object containing field components
        wavelengths: Wavelength of the mode (in nm)
        labels: Optional custom labels for the plot
        sim: Tidy3D simulation object to extract geometry information
        wghb: Bottom waveguide height in microns
        wght: Top waveguide height in microns (for dual waveguide)
        inth: Intermediate layer height in microns (for dual waveguide)
        is_dual_waveguide: Whether this is a dual waveguide configuration
    
    Returns:
        fig, ax: The figure and axes objects
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get the z-coordinates
    z_vals = modes.Ey.z.values  # Convert to numpy array
    
    # Get field components as numpy arrays
    Ex_arr = np.abs(modes.Ex.values)
    Ey_arr = np.abs(modes.Ey.values)
    Ez_arr = np.abs(modes.Ez.values)
    
    # Slice profiles at y=0
    Ex_prof = Ex_arr[0,0,:, 0,0]
    Ey_prof = Ey_arr[0,0,:, 0,0]
    Ez_prof = Ez_arr[0,0,:, 0,0]
    
    # Set default colors for field components
    colors = ['k', 'b', 'r']
    linestyles = ['-', '--', ':']
    
    # Plot each field component
    ax.plot(z_vals, Ey_prof, color=colors[0], linestyle=linestyles[0], lw=2, label='|Ey|')
    ax.plot(z_vals, Ex_prof, color=colors[1], linestyle=linestyles[1], lw=1.5, label='|Ex|')
    ax.plot(z_vals, Ez_prof, color=colors[2], linestyle=linestyles[2], lw=1.5, label='|Ez|')
    
    # Annotate the maximum field value
    max_idx = np.argmax(Ey_prof)
    max_val = Ey_prof[max_idx]
    max_z = z_vals[max_idx]
    ax.annotate(f'Max: {max_val:.3f}', xy=(max_z, max_val), xytext=(max_z+0.1, max_val),
              arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    
    # Build title with wavelength if provided
    if wavelengths is not None:
        if isinstance(wavelengths, (list, tuple)):
            wl = wavelengths[0] if len(wavelengths) > 0 else None
        else:
            wl = wavelengths
        
        if wl is not None:
            title = f'Mode Field at λ={wl:.1f}nm'
        else:
            title = 'Mode Field at Source Plane (y=0)'
    else:
        title = 'Mode Field at Source Plane (y=0)'
    
    # Add labels and grid
    ax.set_xlabel('z (μm)', fontsize=12)
    ax.set_ylabel('Field amplitude', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Shade waveguide regions based on provided parameters
    if wghb is not None:
        # Shade the bottom waveguide region
        ax.axvspan(0, wghb, color='lightblue', alpha=0.3, label='Bottom WG')
        
        # Add a label for the bottom waveguide
        y_max = 0.8 * max_val  # Position label at 80% of max field
        ax.text(wghb/2, y_max, f'WG\n{wghb}μm', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
        
        # For dual waveguide configuration, shade top waveguide too
        if is_dual_waveguide and wght is not None and inth is not None:
            # Top waveguide starts at bottom waveguide + intermediate layer
            top_wg_start = wghb + inth
            top_wg_end = top_wg_start + wght
            
            # Shade the intermediate layer
            ax.axvspan(wghb, top_wg_start, color='lightgray', alpha=0.2, label='Int. Layer')
            
            # Shade the top waveguide
            ax.axvspan(top_wg_start, top_wg_end, color='lightgreen', alpha=0.3, label='Top WG')
            
            # Add labels for intermediate layer and top waveguide
            ax.text(wghb + inth/2, y_max*0.6, f'Int\n{inth}μm', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
                   
            ax.text(top_wg_start + wght/2, y_max*0.8, f'Top WG\n{wght}μm', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
    
    # If no geometry parameters were explicitly provided but sim is available, extract from structures
    elif sim is not None:
        try:
            # Extract waveguide boundaries from simulation structures
            wg_regions = []
            
            # Iterate through structures to find waveguide regions
            for struct in sim.structures:
                # Check if structure might be a waveguide
                if hasattr(struct, 'geometry') and hasattr(struct.geometry, 'bounds'):
                    # Assume structures with medium not SiO2 are waveguides or other features
                    if struct.medium.name != 'SiO2' and struct.medium.name != 'Air':
                        # Extract z-bounds
                        z_min, z_max = struct.geometry.bounds[2]
                        wg_regions.append((z_min, z_max, struct.medium.name))
            
            # Sort regions by z position
            wg_regions.sort(key=lambda x: x[0])
            
            # Shade each found waveguide region
            for i, (z_min, z_max, name) in enumerate(wg_regions):
                color = ['lightblue', 'lightgreen', 'lightyellow'][i % 3]
                ax.axvspan(z_min, z_max, color=color, alpha=0.3, label=f'WG {name}')
                
                # Add label
                y_pos = 0.8 * max_val * (0.9 - i*0.2)  # Stagger labels vertically
                ax.text((z_min+z_max)/2, y_pos, f'{name}\n{z_max-z_min:.2f}μm', 
                       ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
        except Exception as e:
            # If structure extraction fails, fall back to field-based estimation
            print(f"Warning: Could not extract waveguide regions from sim: {e}")
            _estimate_waveguide_from_field(ax, z_vals, Ey_prof, max_val)
    else:
        # If no simulation or geometry parameters, estimate from the field profile
        _estimate_waveguide_from_field(ax, z_vals, Ey_prof, max_val)
    
    # Improve the legend - make sure it's added after all shading
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Improve grid and tick labels
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    return fig, ax

def _estimate_waveguide_from_field(ax, z_vals, field_prof, max_val):
    """Helper function to estimate waveguide region from field profile"""
    try:
        # Use the dominant field component
        max_idx = np.argmax(field_prof)
        
        threshold = max_val * 0.5  # 50% threshold
        
        # Find region where field is above threshold
        above_threshold = field_prof > threshold
        transitions = np.diff(above_threshold.astype(int))
        rising_edges = np.where(transitions == 1)[0]
        falling_edges = np.where(transitions == -1)[0]
        
        if len(rising_edges) > 0 and len(falling_edges) > 0:
            # Get the first rising and falling edges around the maximum
            rising_idx = rising_edges[rising_edges < max_idx][-1] if any(rising_edges < max_idx) else 0
            falling_idx = falling_edges[falling_edges > max_idx][0] if any(falling_edges > max_idx) else len(z_vals)-1
            
            wg_min = z_vals[rising_idx]
            wg_max = z_vals[falling_idx]
            
            # Add a shaded region to highlight the estimated waveguide region
            ax.axvspan(wg_min, wg_max, color='lightsalmon', alpha=0.3, label='Est. WG')
            
            # Add a label
            ax.text((wg_min+wg_max)/2, 0.7*max_val, f'Est. WG\n{wg_max-wg_min:.2f}μm', 
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
        else:
            # Fallback if edges aren't detected properly
            wg_width = z_vals[-1] * 0.2  # Assume 20% of z-range
            wg_center = z_vals[max_idx]
            
            ax.axvspan(wg_center - wg_width/2, wg_center + wg_width/2, 
                      color='lightsalmon', alpha=0.3, label='Est. WG')
            
            ax.text(wg_center, 0.7*max_val, 'Est. WG position\n(field maximum)', 
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
    except Exception as e:
        # If waveguide estimation fails, just skip it
        print(f"Warning: Could not estimate waveguide region: {e}")

def plot_source_spectrum(freqs, source_time=None, mean_freq=None, fwidth=None):
    """Plot the spectrum of the source
    
    Args:
        freqs: List of frequencies in the simulation
        source_time: Tidy3D source_time object if available
        mean_freq: Mean frequency if source_time not available
        fwidth: Frequency width if source_time not available
    
    Returns:
        fig, ax: Figure and axes objects
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tidy3d as td
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine center frequency and width
    if source_time is not None and isinstance(source_time, td.GaussianPulse):
        f0 = source_time.freq0
        df = source_time.fwidth
    elif mean_freq is not None and fwidth is not None:
        f0 = mean_freq
        df = fwidth
    else:
        # Use mean of provided frequencies if no explicit source spectrum info
        f0 = np.mean(freqs)
        df = 0.1 * f0  # Assume 10% bandwidth
    
    # Generate frequency array for plotting
    f_min = f0 - 3*df
    f_max = f0 + 3*df
    f_plot = np.linspace(f_min, f_max, 1000)
    
    # Calculate Gaussian spectrum
    spectrum = np.exp(-((f_plot - f0) / df)**2)
    
    # Convert frequencies to wavelengths for display (nm)
    lambda0 = td.C_0 / f0 * 1e6
    dlambda = (td.C_0 / (f0 - df) - td.C_0 / (f0 + df)) * 1e6
    
    # Plot source spectrum
    ax.plot(f_plot, spectrum, 'b-', lw=2, label='Source Spectrum')
    
    # Mark the center frequency
    ax.axvline(f0, color='r', linestyle='--', label=f'Center: {f0:.3e} Hz')
    
    # Mark the frequencies of interest
    for f in freqs:
        ax.axvline(f, color='g', linestyle='-', alpha=0.3)
        # Add wavelength label
        lambda_nm = td.C_0 / f * 1e6
        ax.annotate(f'{lambda_nm:.1f}nm', 
                   xy=(f, 0.1), 
                   xytext=(0, -10),
                   textcoords='offset points',
                   ha='center', va='top', 
                   fontsize=8, color='green')
    
    # Add wavelength axis on top
    ax_top = ax.twiny()
    lambda_plot = td.C_0 / f_plot * 1e6  # nm
    ax_top.plot(lambda_plot, spectrum, alpha=0)  # Invisible plot to set the correct range
    ax_top.set_xlabel('Wavelength (nm)', fontsize=12)
    
    # Add annotations to show bandwidth info
    bandwidth_text = (
        f"Center: {lambda0:.1f} nm ({f0:.3e} Hz)\n"
        f"Bandwidth: {dlambda:.1f} nm ({df:.3e} Hz)"
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
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    
    return fig, ax

def plot_2d_field(sim_data, freq_idx=0, field_component='Ey', plot_type='intensity'):
    """Plot 2D electric field intensity from the XZ plane monitor
    
    Args:
        sim_data: Tidy3D simulation data object
        freq_idx: Index of frequency to plot (default: 0)
        field_component: Field component to plot ('Ex', 'Ey', or 'Ez')
        plot_type: Type of plot ('intensity', 'real', 'imag', 'abs', 'phase')
    
    Returns:
        fig, ax: Figure and axes objects
    """
    # Extract the 2D field monitor data
    monitor_name = "field_2d_xz"
    if monitor_name not in sim_data.monitor_data:
        # If monitor doesn't exist, return None
        print(f"Monitor '{monitor_name}' not found in simulation data. Use --include-2d-monitor flag when running the simulation.")
        return None, None, None
    
    field_data = sim_data[monitor_name]
    
    # Get the frequency data and convert to wavelength
    freqs = field_data.f.values
    wavelengths = td.C_0 / freqs * 1e6  # nm
    
    # Select the field component
    if field_component not in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
        print(f"Invalid field component: {field_component}, defaulting to 'Ey'")
        field_component = 'Ey'
    
    field = getattr(field_data, field_component)
    
    # Select the frequency
    if freq_idx >= len(freqs):
        freq_idx = 0
        print(f"Warning: freq_idx {freq_idx} out of range, using 0 instead")
    
    # Get the x and z coordinates
    x_coords = field.x.values
    z_coords = field.z.values
    
    # Select the field data at y=0 (it's a 2D simulation)
    field_2d = field.sel(f=freqs[freq_idx]).values
    
    # Reshape to 2D array if needed (depends on Tidy3D output format)
    if field_2d.ndim > 2:
        # For 2D simulations, typically field has shape [x, 1, z]
        field_2d = np.squeeze(field_2d)
    
    # Process the field data based on plot_type
    if plot_type == 'intensity':
        # |E|²
        plot_data = np.abs(field_2d) ** 2
        plot_title = f"|{field_component}|² at λ={wavelengths[freq_idx]:.1f}nm"
        cmap = 'viridis'
    elif plot_type == 'real':
        # Real part
        plot_data = np.real(field_2d)
        plot_title = f"Re({field_component}) at λ={wavelengths[freq_idx]:.1f}nm"
        cmap = 'RdBu'
    elif plot_type == 'imag':
        # Imaginary part
        plot_data = np.imag(field_2d)
        plot_title = f"Im({field_component}) at λ={wavelengths[freq_idx]:.1f}nm"
        cmap = 'RdBu'
    elif plot_type == 'abs':
        # Absolute value
        plot_data = np.abs(field_2d)
        plot_title = f"|{field_component}| at λ={wavelengths[freq_idx]:.1f}nm"
        cmap = 'viridis'
    elif plot_type == 'phase':
        # Phase
        plot_data = np.angle(field_2d)
        plot_title = f"Phase of {field_component} at λ={wavelengths[freq_idx]:.1f}nm"
        cmap = 'twilight'
    else:
        print(f"Invalid plot_type: {plot_type}, defaulting to 'intensity'")
        plot_data = np.abs(field_2d) ** 2
        plot_title = f"|{field_component}|² at λ={wavelengths[freq_idx]:.1f}nm"
        cmap = 'viridis'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the field
    im = ax.pcolormesh(x_coords, z_coords, plot_data.T, cmap=cmap, shading='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(plot_type.capitalize())
    
    # Add axis labels
    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('z (μm)', fontsize=12)
    
    # Add title
    ax.set_title(plot_title, fontsize=14)
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    
    # Add overlay of the simulation geometry (optional)
    # Get the simulation from the data
    sim = sim_data.simulation
    
    # Plot geometry outlines
    for structure in sim.structures:
        # Skip non-box structures for simplicity
        if not isinstance(structure.geometry, td.Box):
            continue
        
        # Get box bounds
        box = structure.geometry
        
        # Extract x and z coordinates for 2D slice
        x_min = box.center[0] - box.size[0]/2
        x_max = box.center[0] + box.size[0]/2
        z_min = box.center[2] - box.size[2]/2
        z_max = box.center[2] + box.size[2]/2
        
        # Plot box outline
        rect = plt.Rectangle((x_min, z_min), box.size[0], box.size[2],
                           fill=False, edgecolor='white', linestyle='--', linewidth=0.5)
        ax.add_patch(rect)
    
    plt.tight_layout()
    
    return fig, ax, plot_data

def plot_mode_profile_and_spectrum(modes, freqs, source_time, figsize=(14, 5), sim=None, wghb=None, wght=None, inth=None, is_dual_waveguide=False):
    """Plot mode profile and source spectrum side by side.
    
    Args:
        modes: Tidy3D modes object containing field components
        freqs: List of frequencies in the simulation (Hz)
        source_time: Tidy3D source_time object (typically GaussianPulse)
        figsize: Figure size as tuple (width, height)
        sim: Tidy3D simulation object to extract geometry information
        wghb: Bottom waveguide height in microns
        wght: Top waveguide height in microns (for dual waveguide)
        inth: Intermediate layer height in microns (for dual waveguide)
        is_dual_waveguide: Whether this is a dual waveguide configuration
        
    Returns:
        fig, (ax1, ax2): Figure and axes objects for the two subplots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tidy3d as td
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Convert frequencies to wavelengths in nm
    mode_wavelengths = [td.C_0 / f * 1e3 for f in freqs]  # nm
    # Calculate mean wavelength (using the harmonic mean - converting frequency to wavelength)
    mean_wavelength = td.C_0 / np.mean(freqs) * 1e3
    
    #--------------------------------------------------------------------------
    # SUBPLOT 1: Mode Profile
    #--------------------------------------------------------------------------
    # Get the z-coordinates
    z_vals = modes.Ey.z.values
    
    # Get field components as numpy arrays
    Ex_arr = np.abs(modes.Ex.values)
    Ey_arr = np.abs(modes.Ey.values)
    Ez_arr = np.abs(modes.Ez.values)
    
    # Slice profiles at y=0
    Ex_prof = Ex_arr[0,0,:, 0,0]
    Ey_prof = Ey_arr[0,0,:, 0,0]
    Ez_prof = Ez_arr[0,0,:, 0,0]
    
    # Set default colors for field components
    colors = ['k', 'b', 'r']
    linestyles = ['-', '--', ':']
    
    # Plot each field component on the first subplot
    ax1.plot(z_vals, Ey_prof, color=colors[0], linestyle=linestyles[0], lw=2, label='|Ey|')
    ax1.plot(z_vals, Ex_prof, color=colors[1], linestyle=linestyles[1], lw=1.5, label='|Ex|')
    ax1.plot(z_vals, Ez_prof, color=colors[2], linestyle=linestyles[2], lw=1.5, label='|Ez|')
    
    # Annotate the maximum field value
    max_idx = np.argmax(Ey_prof)
    max_val = Ey_prof[max_idx]
    max_z = z_vals[max_idx]
    ax1.annotate(f'Max: {max_val:.3f}', xy=(max_z, max_val), xytext=(max_z+0.1, max_val),
              arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    
    # Add labels and grid to mode profile subplot
    ax1.set_xlabel('z (μm)', fontsize=12)
    ax1.set_ylabel('Field amplitude', fontsize=12)
    ax1.set_title(f'Mode Field at Mean λ={mean_wavelength:.1f}nm (harmonic mean of wavelengths)', fontsize=14)
    
    # Shade waveguide regions if parameters are provided
    if wghb is not None:
        # Shade the bottom waveguide region
        ax1.axvspan(0, wghb, color='lightblue', alpha=0.3, label='Bottom WG')
        
        # Add a label for the bottom waveguide
        y_max = 0.8 * max_val  # Position label at 80% of max field
        ax1.text(wghb/2, y_max, f'WG\n{wghb}μm', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
        
        # For dual waveguide configuration, shade top waveguide too
        if is_dual_waveguide and wght is not None and inth is not None:
            # Top waveguide starts at bottom waveguide + intermediate layer
            top_wg_start = wghb + inth
            top_wg_end = top_wg_start + wght
            
            # Shade the intermediate layer
            ax1.axvspan(wghb, top_wg_start, color='lightgray', alpha=0.2, label='Int. Layer')
            
            # Shade the top waveguide
            ax1.axvspan(top_wg_start, top_wg_end, color='lightgreen', alpha=0.3, label='Top WG')
            
            # Add labels for intermediate layer and top waveguide
            ax1.text(wghb + inth/2, y_max*0.6, f'Int\n{inth}μm', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
                   
            ax1.text(top_wg_start + wght/2, y_max*0.8, f'Top WG\n{wght}μm', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
    
    # If no geometry parameters were explicitly provided but sim is available, extract from structures
    elif sim is not None:
        try:
            # Extract waveguide boundaries from simulation structures
            wg_regions = []
            
            # Iterate through structures to find waveguide regions
            for struct in sim.structures:
                # Check if structure might be a waveguide
                if hasattr(struct, 'geometry') and hasattr(struct.geometry, 'bounds'):
                    # Assume structures with medium not SiO2 are waveguides or other features
                    if struct.medium.name != 'SiO2' and struct.medium.name != 'Air':
                        # Extract z-bounds
                        z_min, z_max = struct.geometry.bounds[2]
                        wg_regions.append((z_min, z_max, struct.medium.name))
            
            # Sort regions by z position
            wg_regions.sort(key=lambda x: x[0])
            
            # Shade each found waveguide region
            for i, (z_min, z_max, name) in enumerate(wg_regions):
                color = ['lightblue', 'lightgreen', 'lightyellow'][i % 3]
                ax1.axvspan(z_min, z_max, color=color, alpha=0.3, label=f'WG {name}')
                
                # Add label
                y_pos = 0.8 * max_val * (0.9 - i*0.2)  # Stagger labels vertically
                ax1.text((z_min+z_max)/2, y_pos, f'{name}\n{z_max-z_min:.2f}μm', 
                       ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
        except Exception as e:
            # If structure extraction fails, fall back to field-based estimation
            print(f"Warning: Could not extract waveguide regions from sim: {e}")
            _estimate_waveguide_from_field(ax1, z_vals, Ey_prof, max_val)
    else:
        # If no simulation or geometry parameters, estimate from the field profile
        _estimate_waveguide_from_field(ax1, z_vals, Ey_prof, max_val)
    
    # Add legend (after adding all curves and shading)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    #--------------------------------------------------------------------------
    # SUBPLOT 2: Source Spectrum
    #--------------------------------------------------------------------------
    f0 = source_time.freq0
    df = source_time.fwidth
    
    # Generate frequency array for plotting
    f_min = f0 - 3*df
    f_max = f0 + 3*df
    f_plot = np.linspace(f_min, f_max, 1000)
    
    # Calculate Gaussian spectrum
    spectrum = np.exp(-((f_plot - f0) / df)**2)
    
    # Convert frequencies to wavelengths for display (nm)
    lambda0 = td.C_0 / f0 * 1e3  # Center wavelength in nm
    dlambda = (td.C_0 / (f0 - df) - td.C_0 / (f0 + df)) * 1e3
    
    # Plot source spectrum
    ax2.plot(f_plot, spectrum, 'b-', lw=2, label='Source Spectrum')
    
    # Mark the center frequency
    ax2.axvline(f0, color='r', linestyle='--', label=f'Center: {lambda0:.1f}nm')
    
    # Mark the frequencies of interest
    for f in freqs:
        ax2.axvline(f, color='g', linestyle='-', alpha=0.3)
        # Add wavelength label
        lambda_nm = td.C_0 / f * 1e3
        ax2.annotate(f'{lambda_nm:.1f}nm', 
                   xy=(f, 0.1), 
                   xytext=(0, -10),
                   textcoords='offset points',
                   ha='center', va='top', 
                   fontsize=8, color='green')
    
    # Add annotations to show bandwidth info
    bandwidth_text = (
        f"Center λ: {lambda0:.1f} nm\n"
        f"Bandwidth: {dlambda:.1f} nm"
    )
    ax2.text(0.02, 0.95, bandwidth_text,
           transform=ax2.transAxes,
           fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
           verticalalignment='top')
    
    # Add labels to source spectrum subplot
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Normalized Amplitude', fontsize=12)
    ax2.set_title('Mode Source Spectrum', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(f_min, f_max)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def plot_material_dispersion(material_file, figsize=(10, 6)):
    """
    Plot the wavelength-dependent refractive index of a material and the fitted Lorentz model
    
    Args:
        material_file: Path to material JSON file
        figsize: Figure size (default: (10, 6))
    
    Returns:
        fig, ax: Figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    sys.path.append(os.path.dirname(__file__))
    from material_helper import load_material_data, get_dispersive_medium
    
    # Get material data
    material_data = load_material_data(material_file)
    material_name = os.path.splitext(os.path.basename(material_file))[0]
    process_name = material_data.get("process_name", "unknown")
    
    # Extract wavelengths and indices
    try:
        wavelengths = material_data["core_index"]["wavelengths"]
        indices = material_data["core_index"]["indices"]
    except (KeyError, IndexError):
        print(f"Error: No wavelength-dependent refractive index data in {material_file}")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data points
    ax.plot(wavelengths, indices, 'o', label='Original Data', markersize=6)
    
    # Create a dense wavelength array for plotting the fit
    min_wl, max_wl = min(wavelengths), max(wavelengths)
    plot_wls = np.linspace(min_wl * 0.9, max_wl * 1.1, 500)
    
    # Try to get and plot the fitted Lorentz model
    try:
        import tidy3d as td
        
        # Get the Lorentz medium
        lorentz_medium = get_dispersive_medium(wavelengths, indices)
        
        # If we got a Medium back instead of a LorentzMedium (fallback case), use the permittivity
        if isinstance(lorentz_medium, td.Medium):
            n_fit = np.sqrt(lorentz_medium.permittivity.real) * np.ones_like(plot_wls)
            fit_label = "Average Index (Fit Failed)"
        else:
            # Calculate the refractive index from the Lorentz model
            plot_freqs = td.C_0 / (plot_wls * 1e-3)  # Convert to frequencies in Hz
            epsilon_r = lorentz_medium.eps_model(plot_freqs)
            n_fit = np.sqrt(epsilon_r.real)
            fit_label = "Lorentz Model Fit"
            
        # Plot the fitted curve
        ax.plot(plot_wls, n_fit, '-', label=fit_label, linewidth=2)
        
        # Calculate and display the RMS error for the fit
        interp_fit = np.interp(wavelengths, plot_wls, n_fit)
        rms_error = np.sqrt(np.mean((interp_fit - indices)**2))
        print(f"RMS Error (Lorentz Fit): {rms_error:.6f}")
        
        # Add error info to the plot
        ax.text(0.02, 0.15, 
                f"RMS Error: {rms_error:.6f}", 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
            
    except (ImportError, Exception) as e:
        print(f"Warning: Could not generate Lorentz model fit: {str(e)}")
    
    # Plot original data points again to make them visible over the curve
    ax.plot(wavelengths, indices, 'o', color='blue', markersize=6)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Refractive Index')
    ax.set_title(f'Material Dispersion: {process_name} / {material_name}')
    
    # Add text with min, max, and average values
    min_idx = min(indices)
    max_idx = max(indices)
    avg_idx = sum(indices) / len(indices)
    
    info_text = (
        f"Min index: {min_idx:.4f} @ {wavelengths[indices.index(min_idx)]} nm\n"
        f"Max index: {max_idx:.4f} @ {wavelengths[indices.index(max_idx)]} nm\n"
        f"Avg index: {avg_idx:.4f}"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig, ax

def plot_material_dispersion_from_csv(csv_file, material_name=None, figsize=(10, 6), max_num_poles=1):
    """
    Plot the wavelength-dependent refractive index of a material from CSV data and show the fitted dispersion model
    
    Args:
        csv_file: Path to CSV file with wavelength and refractive index data
        material_name: Material name for the title (optional)
        figsize: Figure size (default: (10, 6))
        max_num_poles: Maximum number of poles to use in the dispersion fitting (default: 1)
    
    Returns:
        fig, ax: Figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from material_helper import get_medium_from_csv
    
    try:
        # Read the CSV file, skipping the first row as header
        data = pd.read_csv(csv_file, delimiter=',', skiprows=1, header=None)
        
        # Check if there are enough columns
        if data.shape[1] < 2:
            print(f"Error: CSV file {csv_file} should have at least 2 columns (wavelength, n)")
            return None, None
        
        # Extract wavelengths and indices
        wavelengths = data.iloc[:, 0]
        indices = data.iloc[:, 1]
        
        # Extract k values if available
        has_k = data.shape[1] >= 3
        if has_k:
            k_values = data.iloc[:, 2]
        else:
            k_values = None
        
        # If material_name is not provided, extract it from the filename
        if material_name is None:
            material_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot original data points
        ax.plot(wavelengths, indices, 'o', label='Original Data', markersize=6, alpha=0.7)
        
        # Create a dense wavelength array for plotting the fit
        min_wl, max_wl = min(wavelengths), max(wavelengths)
        plot_wls = np.linspace(min_wl * 0.9, max_wl * 1.1, 500)
        
        # Try to get and plot the fitted model
        try:
            import tidy3d as td
            from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter
            
            # Create the fitter
            fitter = FastDispersionFitter.from_file(csv_file, skiprows=1, delimiter=",")
            
            # Perform the fit with specified max poles
            advanced_param = AdvancedFastFitterParam(weights=(1, 1))
            medium, rms_error = fitter.fit(
                max_num_poles=max_num_poles, 
                advanced_param=advanced_param
            )
            
            # Calculate the refractive index from the fitted model
            plot_freqs = td.C_0 / (plot_wls )  # Convert to frequencies in Hz
            epsilon_r = medium.eps_model(plot_freqs)
            n_fit = np.sqrt(epsilon_r.real)
            
            # Plot the fitted curve
            ax.plot(plot_wls, n_fit, '-', label=f'Fitted Model ({max_num_poles} poles)', linewidth=2)
            
            # If k values are available, also plot them
            if has_k and np.any(k_values != 0):
                # Plot original k values
                ax_k = ax.twinx()  # Create a second y-axis
                ax_k.plot(wavelengths, k_values, 'x', color='red', label='k values', alpha=0.7)
                
                # Plot fitted k values
                k_fit = np.sqrt(epsilon_r.imag)
                ax_k.plot(plot_wls, k_fit, ':', color='red', label='Fitted k values')
                
                # Set label for k-axis
                ax_k.set_ylabel('Extinction Coefficient (k)', color='red')
                ax_k.tick_params(axis='y', labelcolor='red')
                
                # Add legend for k values
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_k.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='best')
            else:
                ax.legend(loc='best')
            
            # Add error info to the plot
            ax.text(0.02, 0.95, 
                    f"RMS Error: {rms_error:.6f}\nPoles: {max_num_poles}", 
                    transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
            
            # Add medium info to the plot
            medium_info = str(medium)
            # Format the medium info to be more readable
            medium_info = medium_info.replace(", ", ",\n")
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.02, f"Medium Parameters:\n{medium_info}", 
                    transform=ax.transAxes, 
                    fontsize=8,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=props)
                
        except Exception as e:
            print(f"Warning: Could not generate dispersion model fit: {str(e)}")
            print(f"Will display raw data only")
        
        # Plot original data points again to make them visible over the curve
        ax.plot(wavelengths, indices, 'o', color='blue', markersize=6, alpha=0.7)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Refractive Index (n)')
        ax.set_title(f'Material Dispersion: {material_name}')
        
        # Add text with min, max, and average values
        min_idx = min(indices)
        max_idx = max(indices)
        avg_idx = sum(indices) / len(indices)
        
        info_text = (
            f"Min index: {min_idx:.4f} @ {wavelengths.iloc[indices.argmin()]} nm\n"
            f"Max index: {max_idx:.4f} @ {wavelengths.iloc[indices.argmax()]} nm\n"
            f"Avg index: {avg_idx:.4f}"
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')
        
        plt.tight_layout()
        return fig, ax
        
    except Exception as e:
        print(f"Error plotting material dispersion: {str(e)}")
        return None, None

def plot_material_indices_visualization(sim, freqs, figsize=(6, 8), oxide_source=None):
    """
    Plot refractive index vs Z (μm) for each frequency in freqs.
    Uses simulation bounds to create a uniform grid of z points and computes the refractive index at each point.
    Labels each distinct medium used with descriptive names (e.g., 'Oxide', 'Substrate', 'Waveguide', 'Air').
    If the medium is from the material_library, its name is displayed.
    Note: The Silicon medium used by default is the Green2008 one from the tidy3d material library. This should not affect performance significantly.
    
    Args:
        sim: Tidy3D simulation object
        freqs: Frequencies to compute refractive indices at
        figsize: Figure size (default: (6, 8))
        oxide_source: Optional string describing the oxide medium source (default: None)
    
    Returns:
        fig, ax: Figure and axis objects
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tidy3d as td

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Convert freqs to numpy array if not already
    freqs = np.array(freqs)

    # Get simulation bounds from sim.geometry.bounds
    bounds = sim.geometry.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]

    # Create a uniform grid of z points
    z_points = np.linspace(z_min, z_max, 1000)

    # Dictionary to track unique mediums and their descriptive names
    medium_names = {}

    for f in freqs:
        # Convert to wavelength in nm for labeling
        wl_nm = td.C_0 / f
        label = f"{wl_nm:.4f} um"

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
                        # No break since structures defined after will override preceding ones

            # Compute refractive index
            if med:
                if hasattr(med, 'eps_model'):
                    eps = med.eps_model(f)
                    n_val = np.sqrt(np.real(eps))
                    source = "Fit"
                else:
                    n_val = np.sqrt(float(med.permittivity))
                    source = "Fixed Index"
                # Track medium name
                if med not in medium_names:
                    if med in td.material_library:
                        medium_names[med] = f"{td.material_library[med].name} ({source})"
                    else:
                        medium_names[med] = f"{med.__class__.__name__} ({source})"
            else:
                n_val = 1.0  # Default to air if no medium found
            n_profile.append(n_val)

        ax.plot(n_profile, z_points, label=label)

    # Add medium name information to the legend
    # for med, name in medium_names.items():
    #     ax.plot([], [], label=name, linestyle='--')

    ax.set_xlabel('Refractive Index n')
    ax.set_ylabel('Z (μm)')
    ax.set_title('Refractive Index Profile vs Z')
    ax.legend()
    ax.grid(True)
    
    # Annotate default Silicon medium note on plot
    note = "Default Silicon medium: Green2008 (tidy3d material library)\nNo significant performance impact."
    if oxide_source:
        note += f"\nOxide medium source: {oxide_source}"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax