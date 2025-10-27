import numpy as np
import tidy3d as td

def create_grating_var_interp_quartic(dim, tapl, gratl, tapang, tapStart, r0, dr, x0sRs, Rs, As, x0sper, x0sdc, per0s, dc0s, minSize, wg_max_z, etch_depth, wgh_low_um, wgh_high_um, inth_um, dual_layer=False):
    """Create grating structures with variable interpolation (quartic)."""
    print(f"Creating grating with dimensions (um): wgh_low={wgh_low_um}, wgh_high={wgh_high_um}, inth={inth_um}")
    x0 = tapl
    
    # Adjust coordinate system
    x0sper_SI = x0sper - tapStart
    x0sdc_SI = x0sdc - tapStart
    x0sRs_SI = x0sRs - tapStart
    per0s_SI = per0s

    # Check if we need to convert period values (if they're in nm)
    if np.max(per0s) > 10:  # Likely in nm
        print("Converting period values from nm to um")
        per0s_SI = per0s_SI * 1e-3  # Convert to um

    etches = []
    while x0 < tapl + gratl:
        per = np.interp(x0, x0sper_SI, per0s_SI)
        dc = np.interp(x0, x0sdc_SI, dc0s)
        r = np.interp(x0, x0sRs_SI, Rs)

        # Skip etch creation if duty cycle is 0 (or very close to 0)
        if abs(dc) < 1e-10:
            x0 = x0 + per
            continue

        etch_length = max([minSize, per*dc])

        r_tap = r0 + (tapStart + x0)*dr
        max_y = r_tap * (1/np.tan(tapang)) * (-1 + np.sqrt(1 + (2*x0*(np.tan(tapang)**2))/r_tap)) + 1

        # Bottom layer parameters - adjust to etch through the bottom waveguide layer
        zmin_bottom = 0  # Start from bottom of waveguide layer
        zmax_bottom = wgh_low_um  # Etch through the entire bottom layer
        
        # Top layer parameters (if dual layer)
        if dual_layer:
            zmin_top = wgh_low_um + inth_um
            zmax_top = zmin_top + wgh_high_um

        length = etch_length

        if dim == 3:
            nPts = 100
            y = np.linspace(-max_y, max_y, nPts)

            y2_comp = ((y)**2)/(2*r)
            A = np.interp(x0, x0sRs_SI, As)
            y4_comp = ((y)**4)*A

            x_left = x0 + y2_comp + y4_comp
            x_right = x_left + length

            V = np.zeros((2*nPts, 2))
            V[0:nPts, 0] = x_left
            V[nPts:2*nPts, 0] = np.flip(x_right, 0)
            V[0:nPts, 1] = y
            V[nPts:2*nPts, 1] = np.flip(y, 0)
            
            # Bottom layer etch
            s_bottom = td.PolySlab(
                vertices=V,
                slab_bounds=(zmin_bottom, zmax_bottom),
                axis=2,
                sidewall_angle=0,
                reference_plane="top",
            )
            etches.append(s_bottom)
            
            # Top layer etch (if dual layer)
            if dual_layer:
                s_top = td.PolySlab(
                    vertices=V,
                    slab_bounds=(zmin_top, zmax_top),
                    axis=2,
                    sidewall_angle=0,
                    reference_plane="top",
                )
                etches.append(s_top)
        
        x0 = x0 + per
    
    print(f"Created {len(etches)} grating etches")
    return etches

def create_taper(wg_width, zmin, zmax, tapStart, tapang, r0, dr, x0_right, dual_layer=False, zmin_top=0, zmax_top=0):
    """Create taper structure with optional dual layer support."""
    nPts = 100
    r = r0 + (tapStart + x0_right)*dr
    max_y = r * (1/np.tan(tapang)) * (-1 + np.sqrt(1 + (2*x0_right*(np.tan(tapang)**2))/r))

    y = np.linspace(-max_y, max_y, nPts)
    y2_comp = ((y)**2)/(2*r)

    x_right = x0_right - y2_comp
    
    V = np.zeros((nPts+2, 2))
    V[0:nPts, 0] = x_right
    V[0:nPts, 1] = y

    V[nPts, 0] = 0
    V[nPts+1, 0] = 0
    V[nPts, 1] = wg_width/2
    V[nPts+1, 1] = -1*wg_width/2

    # Bottom layer taper
    t_bottom = td.PolySlab(
        vertices=V,
        slab_bounds=(zmin, zmax),
        axis=2,
        sidewall_angle=0,
        reference_plane="top",
    )
    
    if dual_layer:
        # Top layer taper
        t_top = td.PolySlab(
            vertices=V,
            slab_bounds=(zmin_top, zmax_top),
            axis=2,
            sidewall_angle=0,
            reference_plane="top",
        )
        return [t_bottom, t_top]
    
    return [t_bottom] 