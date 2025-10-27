import numpy as np
import gdstk
import tidy3d as td
import os
from typing import List, Tuple, Union

def extract_polygons_from_polyslab(polyslab: td.PolySlab, z_slice: float = None) -> List[np.ndarray]:
    """
    Extract 2D polygon vertices from a Tidy3D PolySlab geometry.
    
    Parameters:
    -----------
    polyslab : td.PolySlab
        The PolySlab geometry to extract polygons from
    z_slice : float, optional
        Z-coordinate at which to slice the geometry (unused for PolySlab)
        
    Returns:
    --------
    List of numpy arrays, each containing vertices of a polygon
    """
    # For PolySlab, the vertices are already defined in 2D
    vertices = np.array(polyslab.vertices)
    
    # Ensure vertices form a closed polygon
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0:1]])
    
    return [vertices]

def extract_polygons_from_box(box: td.Box, z_slice: float = None) -> List[np.ndarray]:
    """
    Extract 2D polygon vertices from a Tidy3D Box geometry.
    
    Parameters:
    -----------
    box : td.Box
        The Box geometry to extract polygons from
    z_slice : float, optional
        Z-coordinate at which to slice the geometry
        
    Returns:
    --------
    List of numpy arrays, each containing vertices of a polygon
    """
    # Get the bounds of the box
    rmin, rmax = box.bounds
    
    # Create rectangle vertices in the XY plane
    vertices = np.array([
        [rmin[0], rmin[1]],  # bottom-left
        [rmax[0], rmin[1]],  # bottom-right
        [rmax[0], rmax[1]],  # top-right
        [rmin[0], rmax[1]],  # top-left
        [rmin[0], rmin[1]]   # close the polygon
    ])
    
    return [vertices]

def extract_polygons_from_geometry(geometry: td.Geometry, z_slice: float = None) -> List[np.ndarray]:
    """
    Extract 2D polygon vertices from any Tidy3D geometry.
    
    Parameters:
    -----------
    geometry : td.Geometry
        The geometry to extract polygons from
    z_slice : float, optional
        Z-coordinate at which to slice the geometry
        
    Returns:
    --------
    List of numpy arrays, each containing vertices of a polygon
    """
    if isinstance(geometry, td.PolySlab):
        return extract_polygons_from_polyslab(geometry, z_slice)
    elif isinstance(geometry, td.Box):
        return extract_polygons_from_box(geometry, z_slice)
    else:
        # For other geometry types, try to use Tidy3D's cross-section functionality
        # This is a fallback that may need refinement based on specific geometry types
        print(f"Warning: Unsupported geometry type {type(geometry)}. Attempting generic extraction.")
        try:
            # Use a default z-slice if not provided
            if z_slice is None:
                z_slice = 0.0
                
            # This is a placeholder - in practice, you might need to implement
            # specific handlers for other geometry types
            return []
        except Exception as e:
            print(f"Error extracting polygons from {type(geometry)}: {e}")
            return []

def create_gds_from_structures(taper_structures: List[Union[td.Structure, td.Geometry]], 
                             etch_structures: List[Union[td.Structure, td.Geometry]],
                             output_path: str,
                             taper_layer: Tuple[int, int] = (1, 0),
                             etch_layer: Tuple[int, int] = (2, 0),
                             cell_name: str = "grating_device",
                             z_slice: float = None) -> str:
    """
    Create a GDS file from Tidy3D structures with tapers and etches on separate layers.
    
    Parameters:
    -----------
    taper_structures : List[td.Structure]
        List of taper structures
    etch_structures : List[td.Structure]
        List of etch structures (grating teeth)
    output_path : str
        Path where to save the GDS file
    taper_layer : Tuple[int, int]
        GDS layer and datatype for tapers (layer, datatype)
    etch_layer : Tuple[int, int]
        GDS layer and datatype for etches (layer, datatype)
    cell_name : str
        Name of the main cell in the GDS
    z_slice : float, optional
        Z-coordinate at which to slice 3D geometries
        
    Returns:
    --------
    str : Path to the created GDS file
    """
    # Create a new library
    lib = gdstk.Library()
    
    # Create the main cell
    cell = lib.new_cell(cell_name)
    
    # Process taper structures
    print(f"Processing {len(taper_structures)} taper structures...")
    for i, item in enumerate(taper_structures):
        try:
            # Handle both Structure objects and raw geometries
            if hasattr(item, 'geometry'):
                geometry = item.geometry
            elif hasattr(item, '__class__') and 'td.' in str(item.__class__):
                geometry = item
            else:
                # Handle dictionary format
                geometry = item.get('geometry') if isinstance(item, dict) else item
                
            polygons = extract_polygons_from_geometry(geometry, z_slice)
            for j, vertices in enumerate(polygons):
                if len(vertices) >= 3:  # Valid polygon needs at least 3 vertices
                    # Remove duplicate consecutive vertices
                    vertices_clean = []
                    for k in range(len(vertices)):
                        if k == 0 or not np.allclose(vertices[k], vertices[k-1], atol=1e-10):
                            vertices_clean.append(vertices[k])
                    
                    if len(vertices_clean) >= 3:
                        polygon = gdstk.Polygon(vertices_clean, layer=taper_layer[0], datatype=taper_layer[1])
                        cell.add(polygon)
                        print(f"Added taper polygon {i}-{j} with {len(vertices_clean)} vertices")
        except Exception as e:
            print(f"Error processing taper structure {i}: {e}")
    
    # Process etch structures
    print(f"Processing {len(etch_structures)} etch structures...")
    for i, item in enumerate(etch_structures):
        try:
            # Handle both Structure objects and raw geometries
            if hasattr(item, 'geometry'):
                geometry = item.geometry
            elif hasattr(item, '__class__') and 'td.' in str(item.__class__):
                geometry = item
            else:
                # Handle dictionary format
                geometry = item.get('geometry') if isinstance(item, dict) else item
                
            polygons = extract_polygons_from_geometry(geometry, z_slice)
            for j, vertices in enumerate(polygons):
                if len(vertices) >= 3:  # Valid polygon needs at least 3 vertices
                    # Remove duplicate consecutive vertices
                    vertices_clean = []
                    for k in range(len(vertices)):
                        if k == 0 or not np.allclose(vertices[k], vertices[k-1], atol=1e-10):
                            vertices_clean.append(vertices[k])
                    
                    if len(vertices_clean) >= 3:
                        polygon = gdstk.Polygon(vertices_clean, layer=etch_layer[0], datatype=etch_layer[1])
                        cell.add(polygon)
                        print(f"Added etch polygon {i}-{j} with {len(vertices_clean)} vertices")
        except Exception as e:
            print(f"Error processing etch structure {i}: {e}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the GDS file
    lib.write_gds(output_path)
    print(f"GDS file saved to: {output_path}")
    
    # Print summary
    total_polygons = len(cell.polygons)
    print(f"Created GDS with {total_polygons} total polygons:")
    print(f"  - Layer {taper_layer[0]}.{taper_layer[1]}: Tapers")
    print(f"  - Layer {etch_layer[0]}.{etch_layer[1]}: Etches")
    
    return output_path

def create_gds_with_separate_layers(taper_structures: List[Union[td.Structure, td.Geometry]], 
                                  etch_structures: List[Union[td.Structure, td.Geometry]],
                                  mat_file_path: str,
                                  taper_layer: Tuple[int, int] = (1, 0),
                                  etch_layer: Tuple[int, int] = (2, 0)) -> str:
    """
    Convenience function to create GDS file in the same directory as the mat file.
    
    Parameters:
    -----------
    taper_structures : List[td.Structure]
        List of taper structures
    etch_structures : List[td.Structure]
        List of etch structures (grating teeth)
    mat_file_path : str
        Path to the original mat file (used to determine output location and naming)
    taper_layer : Tuple[int, int]
        GDS layer and datatype for tapers (layer, datatype)
    etch_layer : Tuple[int, int]
        GDS layer and datatype for etches (layer, datatype)
        
    Returns:
    --------
    str : Path to the created GDS file
    """
    # Get the directory and base name from the mat file path
    mat_file_dir = os.path.dirname(os.path.abspath(mat_file_path))
    mat_file_base = os.path.splitext(os.path.basename(mat_file_path))[0]
    
    # Create output path
    gds_output_path = os.path.join(mat_file_dir, f"{mat_file_base}_grating.gds")
    
    # Create cell name from mat file base name
    cell_name = mat_file_base.replace('.', '_').replace('-', '_')
    
    return create_gds_from_structures(
        taper_structures, 
        etch_structures,
        gds_output_path,
        taper_layer,
        etch_layer,
        cell_name
    )
