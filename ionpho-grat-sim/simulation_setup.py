import numpy as np
import tidy3d as td
import matplotlib.pyplot as plt
import os
import json
from load_material import get_layer_parameters, get_optical_properties, get_simulation_parameters

def load_simulation_config(config_file=None):
    """Load simulation configuration from JSON file."""
    folder_name = os.path.dirname(os.path.abspath(__file__))
    
    if not config_file:
        config_file = os.path.join(folder_name, "simulation_config.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_file}")
            return config
    except Exception as e:
        raise ValueError(f"Failed to load configuration file: {e}")

def setup_simulation(p, mat, grating_etches, taper_structures, dual_layer=False, wgh_low=None, wgh_high=None, inth=None, material_data=None, config_file=None, gds_structures=[]):
    """Set up the Tidy3D simulation with the grating structures."""
    # Load configuration
    config = load_simulation_config(config_file)
    
    # Get grid size from config
    dl = config["simulation"]["dl"]
    simtime = config["simulation"]["simulation_time"]
    print(f"Using grid size dl={dl} from configuration, simulation time={simtime} from configuration")
    
    # Get bottom boundary type from config
    bottom_boundary = config["simulation"]["bottom_boundary"]
    
    # Determine substrate type based on bottom boundary
    # PEC boundary means metal substrate, PML means silicon substrate
    substrate_type = "metal" if bottom_boundary.upper() == "PEC" else "si"
    print(f"Using substrate type: {substrate_type} based on bottom_boundary={bottom_boundary}")
    
    # Material data must be provided
    if material_data is None:
        raise ValueError("Material data must be provided")
        
    # Use parameters from material_data
    wgh_low, wgh_high, inth, _ = get_layer_parameters(material_data)
        
    # Convert to um for tidy3d
    wgh_low_um = wgh_low  # is already in um
    wgh_high_um = wgh_high 
    inth_um = inth 
    
    # Get thickness parameters from material data
    sim_params = get_simulation_parameters(material_data)
    toxt = sim_params.get('toxt')  # Top oxide thickness (um)
    boxt = sim_params.get('boxt')  # Bottom oxide (BOX) thickness (um)
    tSi = sim_params.get('tSi')    # Silicon substrate thickness (um)
    tAir = sim_params.get('tair')  # Air layer thickness (um)
    
    # Calculate total waveguide height
    wg_max_z = wgh_low_um + (inth_um + wgh_high_um if dual_layer else 0)
    etch_depth = wgh_low_um  # Default etch depth is the bottom waveguide height
    
    # Setup material properties based on wavelength
    wavelength = p.get('lambda')  # Default wavelength 500nm
    
    # Get optical properties from material data, passing wavelength for index selection
    si_index, ox_index, wg_index = get_optical_properties(material_data, wavelength)
    print(f"Using wavelength {wavelength} um: si_index={si_index}, ox_index={ox_index}, wg_index={wg_index}")
    
    # Create Tidy3D medium objects
    ox = td.Medium(permittivity=ox_index**2)
    si = td.Medium(permittivity=si_index**2)
    wg = td.Medium(permittivity=wg_index**2)
    
    # Set simulation boundaries
    tapang_in = mat.get("tapang")
    tapl_SI = mat.get("tapl")
    gratl = mat.get("lgrat")
    wg_width = p.get('wg_width', 0.5)
    wg_start = -2  # Start position of waveguide (um)
    wg_end_margin = config["simulation"]["wg_end_margin"]  # Extra margin along x (um)
    wg_end = gratl + tapl_SI + wg_end_margin  # End position with margin
    
    # Box for oxide layers
    pre_tap_length = 5
    max_y = (tapl_SI + gratl*0.95) * np.tan(tapang_in)
    oxide_box = td.Box.from_bounds(
        rmin=(wg_start-pre_tap_length, -max_y-2-pre_tap_length, -boxt-pre_tap_length), 
        rmax=(wg_end+pre_tap_length, max_y+2+pre_tap_length, toxt)
    )
    
    # Create structures list
    structures = []
    structures.append(td.Structure(geometry=oxide_box, medium=ox))
    
    # Add silicon substrate or set PEC boundary based on substrate_type
    if substrate_type.lower() == "si":
        # Silicon substrate
        si_substrate = td.Box.from_bounds(
            rmin=(wg_start-pre_tap_length, -max_y-2-pre_tap_length, -boxt-tSi - pre_tap_length), 
            rmax=(wg_end+pre_tap_length, max_y+2+pre_tap_length, -boxt)
        )
        structures.append(td.Structure(geometry=si_substrate, medium=si))
    
    # Feed waveguide
    feed_wg = td.Box(
        center=(wg_start/2-pre_tap_length/2, 0, wgh_low_um/2), 
        size=(0-wg_start+pre_tap_length, wg_width, wgh_low_um)
    )
    structures.append(td.Structure(geometry=feed_wg, medium=wg))
    
    # Add taper(s)
    for taper in taper_structures:
        structures.append(td.Structure(geometry=taper, medium=wg))
    
    # Add etches (as holes in the waveguide material)
    for etch in grating_etches:
        structures.append(td.Structure(geometry=etch, medium=ox))

    for s in gds_structures:
        structures.append(s)
    
    # If dual layer, add top waveguide feed
    if dual_layer and wgh_high_um > 0:
        feed_wg_top = td.Box(
            center=(wg_start/2-pre_tap_length/2, 0, wgh_low_um + inth_um + wgh_high_um/2), 
            size=(0-wg_start+pre_tap_length, wg_width, wgh_high_um)
        )
        structures.append(td.Structure(geometry=feed_wg_top, medium=wg))
    
    print(len(structures))
    
    # Setup simulation size and center
    # For metal substrate, we don't need to include the silicon substrate in the simulation
    if substrate_type.lower() == "si":
        sim_size = [wg_end - wg_start, max_y*2+4, boxt + tSi + wg_max_z + toxt + tAir]
        sim_center = [(wg_end + wg_start)/2, 0, (wg_max_z + toxt + tAir - boxt - tSi)/2]
    else:  # metal substrate
        sim_size = [wg_end - wg_start, max_y*2+2, boxt + wg_max_z + toxt + tAir]
        sim_center = [(wg_end + wg_start)/2, 0, (wg_max_z + toxt + tAir - boxt)/2]
    
    # Define frequency and source parameters
    freq0 = td.C_0 / wavelength
    f_width = 0.1 * freq0
    
    # Create monitors
    field_monitor = td.FieldMonitor(
        center=[sim_center[0], sim_center[1], toxt+tAir/2], 
        size=[sim_size[0], sim_size[1], 0], 
        freqs=[freq0], 
        name="field"
    )
    
    field_monitor2 = td.FieldMonitor(
        center=[sim_center[0], 0, sim_center[2]], 
        size=[sim_size[0], 0, sim_size[2]], 
        freqs=[freq0], 
        name="field2"
    )
    
    flux_monitor = td.FluxMonitor(
        center=[sim_center[0], 0, toxt+tAir/2], 
        size=[sim_size[0], sim_size[1], 0], 
        freqs=[freq0], 
        name="flux_air"
    )
    
    flux_monitor2 = td.FluxMonitor(
        center=[sim_center[0], 0, -boxt+0.5], 
        size=[sim_size[0], sim_size[1], 0], 
        freqs=[freq0], 
        name="flux_si"
    )
    
    flux_monitor_end = td.FluxMonitor(
        center=[wg_end-0.5, 0, 0], 
        size=[0, sim_size[1], 5], 
        freqs=[freq0], 
        name="flux_end"
    )
    
    flux_monitor_source = td.FluxMonitor(
        center=[0.5, 0, 0], 
        size=[0, 5, 5], 
        freqs=[freq0], 
        name="flux_source"
    )
    
    # Get the ion trap height from config for specific monitors
    ion_z_height = config["ion_trap"]["z_height"]
    print(f"Using ion trap height: {ion_z_height} um for specific far field monitors")
    
    # Get monitor configuration
    xy_config = config["far_field"]["xy_monitor"]
    xz_config = config["far_field"]["xz_monitor"]
    
    # Get number of points from config
    xy_num_points = xy_config.get("num_points")
    xz_num_points = xz_config.get("num_points")
    
    # Get z range for height-specific monitors
    xy_z_min = xy_config.get("z_min")
    xy_z_max = xy_config.get("z_max")
    num_z_heights = xy_config.get("num_z_heights")
    
    # Calculate x position based on angle
    theta_inc_degrees = p.get('thetaIncDegrees')
    theta_inc_rad = np.radians(theta_inc_degrees)
    
    # Calculate x-offset for ion trap height
    ion_x_offset = -ion_z_height * np.tan(theta_inc_rad) # negative since the grating is backward emitting
    print(f"Using theta: {theta_inc_degrees} degrees, x_offset: {ion_x_offset} um for ion trap height")
    
    # Create xy monitors at the ion trap height with different monitor sizes
    xy_ff_monitors = []
    
    # Define different monitor sizes
    monitor_sizes = [10, 30, 50, 100]  # Sizes in µm
    
    # Create monitors with different sizes at ion trap height
    for i, size in enumerate(monitor_sizes):
        # Calculate new sampling points based on the monitor size
        x_start_size = ion_x_offset - size/2
        x_end_size = ion_x_offset + size/2
        y_start_size = -size/2
        y_end_size = size/2
        
        # Create coordinate arrays
        x_points_size = np.linspace(x_start_size, x_end_size, xy_num_points)
        y_points_size = np.linspace(y_start_size, y_end_size, xy_num_points+1)
        
        # Create the monitor at ion trap height
        ff_monitor_xy = td.FieldProjectionCartesianMonitor(
            center=[sim_center[0], sim_center[1], toxt + tAir/2], 
            size=[sim_size[0], sim_size[1], 0], 
            freqs=[freq0], 
            name=f"far_field_xy_size{size}_ionheight",
            normal_dir="+",
            proj_axis=2,  # z-axis projection
            proj_distance=ion_z_height,
            far_field_approx=False,
            x=x_points_size,
            y=y_points_size
        )

        # Add to the list
        xy_ff_monitors.append(ff_monitor_xy)


        # Create the monitor at ion trap height
        ff_monitor_xy = td.FieldProjectionCartesianMonitor(
            center=[sim_center[0], sim_center[1], toxt + tAir/2], 
            size=[sim_size[0], sim_size[1], 0], 
            freqs=[freq0], 
            name=f"far_field_xy_size{size}_ionheight_1",
            normal_dir="+",
            proj_axis=2,  # z-axis projection
            proj_distance=ion_z_height-1,
            far_field_approx=False,
            x=x_points_size,
            y=y_points_size
        )

        # Add to the list
        #xy_ff_monitors.append(ff_monitor_xy)
        # Create the monitor at ion trap height
        ff_monitor_xy = td.FieldProjectionCartesianMonitor(
            center=[sim_center[0], sim_center[1], toxt + tAir/2], 
            size=[sim_size[0], sim_size[1], 0], 
            freqs=[freq0], 
            name=f"far_field_xy_size{size}_ionheight1",
            normal_dir="+",
            proj_axis=2,  # z-axis projection
            proj_distance=ion_z_height+1,
            far_field_approx=False,
            x=x_points_size,
            y=y_points_size
        )
        # Add to the list
        #xy_ff_monitors.append(ff_monitor_xy)

        # Create the monitor at ion trap height
        ff_monitor_xy = td.FieldProjectionCartesianMonitor(
            center=[sim_center[0], sim_center[1], toxt + tAir/2], 
            size=[sim_size[0], sim_size[1], 0], 
            freqs=[freq0], 
            name=f"far_field_xy_size{size}_ionheight_2",
            normal_dir="+",
            proj_axis=2,  # z-axis projection
            proj_distance=ion_z_height-2,
            far_field_approx=False,
            x=x_points_size,
            y=y_points_size
        )
        # Add to the list
        #xy_ff_monitors.append(ff_monitor_xy)
    
    # Add monitors at different heights from config
    height_ff_monitors = []
    
    # Use the largest monitor size (100 μm) for all height monitors
    size = 100
    
    # Create a linspace of heights between min and max with specified number of points
    z_heights = np.linspace(xy_z_min, xy_z_max, num_z_heights)
    print(f"Creating {num_z_heights} XY monitors at heights: {z_heights}")
    
    for i, height in enumerate(z_heights):
        # Calculate height-specific x-offset
        height_x_offset = -height * np.tan(theta_inc_rad)
        
        # Create coordinate arrays for this height
        x_points_height = np.linspace(height_x_offset - size/2, height_x_offset + size/2, xy_num_points)
        y_points_height = np.linspace(-size/5, size/5, xy_num_points//2)
        
        # Create monitor at this specific height
        ff_monitor_height = td.FieldProjectionCartesianMonitor(
            center=[sim_center[0], sim_center[1], toxt + tAir/2], 
            size=[sim_size[0], sim_size[1], 0], 
            freqs=[freq0], 
            name=f"far_field_xy_height{int(height)}",
            normal_dir="+",
            proj_axis=2,  # z-axis projection
            proj_distance=height,  # Use the specified height
            far_field_approx=False,
            x=x_points_height,
            y=y_points_height
        )
        
        # Add to the list
        height_ff_monitors.append(ff_monitor_height)
    
    # Define x and z ranges for the XZ monitor
    xz_size = 100  # um
    
    # Get z range for XZ monitors
    xz_z_min = xz_config.get("z_min", 25)
    xz_z_max = xz_config.get("z_max", 75)
    
    xs_far = np.linspace(ion_x_offset-xz_size/2, ion_x_offset+xz_size/2, xz_num_points)
    zs_far = np.linspace(xz_z_min, xz_z_max, xz_num_points+1)
    print(f"XZ monitor z-range: {xz_z_min}-{xz_z_max} um with {xz_num_points+1} points")
    
    # Create multiple XZ monitors at different projection distances
    xz_ff_monitors = []
    
    # Create a central XZ monitor (y=0) and additional monitors at different y positions
    projection_distances = [0]  # Different y-distances in µm
    
    for dist in projection_distances:
        # Create an XZ monitor for this projection distance
        ff_monitor_xz = td.FieldProjectionCartesianMonitor(
            center=[sim_center[0], 0, toxt + tAir/2],  # Center at the simulation center
            size=[sim_size[0], sim_size[1], tAir*0.9],       # Size covering the full x-z plane
            freqs=[freq0], 
            name=f"far_field_xz_ydist{dist}",
            normal_dir="+",
            proj_axis=1,                              # Project along y-axis
            proj_distance=dist,                       # Use specified y-projection distance
            far_field_approx=False,
            x=xs_far,                                # X coordinates
            y=zs_far                                 # These are actually Z coordinates
        )
        
        # Add to the list
        xz_ff_monitors.append(ff_monitor_xz)
    
    # Add a far field angle monitor covering the top hemisphere
    # Define angle ranges for the far field monitor
    n_phi = 101    # Number of phi points (every 5 degrees)
    n_theta = 101  # Number of theta points (every 5 degrees from 0 to 90 degrees)
    
    # Create arrays for phi and theta angles
    phi_angles = np.linspace(0, 2*np.pi, n_phi, endpoint=False)  # Full 360 degrees in xy-plane
    theta_angles = np.linspace(0, np.pi/2, n_theta)  # 0 to 90 degrees from z-axis (top hemisphere)
    
    # Create multiple far field angle monitors at different projection distances
    far_field_angle_monitors = []
    
    # Use projection distances from config
    projection_distances = config["far_field"].get("angle_proj_distances")
    
    for distance in projection_distances:
        # Create a far field angle monitor for this projection distance
        ff_angle_monitor = td.FieldProjectionAngleMonitor(
            center=[sim_center[0], sim_center[1], toxt + tAir/2],  # Just above the top oxide
            size=[sim_size[0], sim_size[1], 0],       # Size covering the simulation area
            freqs=[freq0],
            name=f"far_field_angle_dist{distance}",
            normal_dir="+",                           # Project in the positive z direction
            proj_distance=distance,                   # Use the specified projection distance
            phi=phi_angles,                          # Azimuthal angles (xy-plane)
            theta=theta_angles,                      # Polar angles from z-axis
            far_field_approx=(distance > 100)         # Use far field approximation for larger distances
        )
        
        # Add to the list
        far_field_angle_monitors.append(ff_angle_monitor)
    
    # Initialize simulation with the monitors
    all_monitors = [field_monitor, field_monitor2, flux_monitor, flux_monitor2, 
                 flux_monitor_end, flux_monitor_source]
    all_monitors.extend(xz_ff_monitors)  # Add all XZ monitors
    all_monitors.extend(xy_ff_monitors)
    all_monitors.extend(height_ff_monitors)
    all_monitors.extend(far_field_angle_monitors)  # Add all angle monitors
    
    # Set boundary conditions based on substrate type
    if substrate_type.lower() == "metal":
        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary(minus=td.PECBoundary(), plus=td.PML())
        )
    else:
        boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())
    
    # Initialize simulation
    sim = td.Simulation(
        size=sim_size,
        center=sim_center,
        grid_spec=td.GridSpec.uniform(dl=dl),
        structures=structures,
        sources=[],
        monitors=all_monitors,
        run_time=simtime,  # Use run time from config
        boundary_spec=boundary_spec,
        symmetry=(0, -1, 0),
    )
    
    return sim

def add_mode_source(sim, wavelength):
    """Add a mode source to the simulation and display mode profiles."""
    freq0 = td.C_0 / wavelength
    f_width = 0.1 * freq0
    
    # Create a mode solver
    from tidy3d.plugins.mode import ModeSolver
    src_pos = -0.5
    src_plane = td.Box(center=[src_pos, 0, 0], size=[0, 5, 5])
    mode_spec = td.ModeSpec(num_modes=3)
    ms = ModeSolver(simulation=sim, plane=src_plane, mode_spec=mode_spec, freqs=[freq0])
    modes = ms.solve()
    
    # Plot mode profiles
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for mode_idx in range(3):
        ms.plot_field(field_name="Ey", mode_index=mode_idx, ax=axes[mode_idx])
        axes[mode_idx].set_title(f"Mode {mode_idx}")
    plt.tight_layout()
    plt.show()
    
    # Create a Gaussian pulse source
    source_time = td.GaussianPulse(freq0=freq0, fwidth=f_width)
    mode_source = ms.to_source(mode_index=0, direction="+", source_time=source_time)
    
    # Add source to simulation
    sim_with_source = sim.copy(update=dict(sources=[mode_source]))
    
    # Return the source_time directly along with the other objects
    return sim_with_source, ms, modes, source_time 