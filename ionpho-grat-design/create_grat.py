import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from scipy.io import savemat
import hdf5storage

# Add path to modules directory
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, "modules")
if module_path not in sys.path:
    sys.path.append(module_path)

# Import modules
from modules.config_loader import load_config
from modules.beam_propagation import calculate_beam_propagation
from modules.lut_handler import load_and_process_lut, load_and_process_csv_lut, plot_lut_2d
from modules.longitudinal_design import create_longitudinal_design, plot_design_results, plot_grating_design_analysis
from modules.tidy3d_simulation import simulate_grating, create_simulation
from modules.transverse_design import create_transverse_design, plot_transverse_design

def main():
    """Main function to run the grating design process"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AIM Grating Design')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to grating configuration file (required)')
    parser.add_argument('--lut_file', type=str, required=True,
                        help='Path to LUT file (.mat) (required)')
    parser.add_argument('--sim_config', type=str, 
                        help='Path to simulation configuration JSON file')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots instead of saving them')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--skip_transverse', action='store_true',
                        help='Skip transverse design calculation')
    
    # Tidy3D simulation arguments
    tidy3d_group = parser.add_argument_group('Tidy3D Simulation Options')
    tidy3d_group.add_argument('--run_tidy3d', action='store_true',
                        help='Run Tidy3D simulation after design is complete')
    tidy3d_group.add_argument('--no_simulation', action='store_true',
                        help='Set up Tidy3D simulation but do not run it (only used with --run_tidy3d)')
    tidy3d_group.add_argument('--tidy3d_output_dir', type=str, default=None,
                        help='Directory to save Tidy3D simulation output (default: {output_dir}/tidy3d)')
    tidy3d_group.add_argument('--material_json', type=str, 
                        help='Path to material stack JSON file (overrides any in sim_config)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up Tidy3D output directory if requested
    tidy3d_output_dir = args.tidy3d_output_dir if args.tidy3d_output_dir else os.path.join(args.output_dir, 'tidy3d')
    if args.run_tidy3d and not os.path.exists(tidy3d_output_dir):
        os.makedirs(tidy3d_output_dir)
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    p, p2 = load_config(args.config)
    
    # Add LUT file path to parameters
    p['lut_file_path'] = args.lut_file
    if p2 is not None:
        p2['lut_file_path'] = args.lut_file
    
    # Step 1: Calculate beam propagation
    print("Calculating beam propagation...")
    if p["two_beams"]:
        # Handle the two beams case
        beam_results = calculate_beam_propagation(p, p2)
        phi_fs, EfieldAmp, phi_fsChip, EfieldAmpChip, theta0, Xc, Yc, Xwg, Ywg = beam_results[:9]
        phi_fs2, EfieldAmp2, phi_fsChip2, EfieldAmpChip2, theta02, Xc2, Yc2, Xwg2, Ywg2 = beam_results[9:]
        
        # Calculate beam waist information
        waist_info = {
            "XwgWaists": beam_results[8]["XwgWaists"],
            "YwgWaists": beam_results[8]["YwgWaists"],
            "XcWaists": beam_results[8]["XcWaists"],
            "YcWaists": beam_results[8]["YcWaists"]
        }
        p["waist_info"] = waist_info
    else:
        # Single beam case
        phi_fs, EfieldAmp, phi_fsChip, EfieldAmpChip, theta0, Xc, Yc, Xwg, Ywg, waist_info = calculate_beam_propagation(p)
        p["waist_info"] = waist_info
        p["theta0"] = theta0
    
    # Step 2: Load and process LUT
    print(f"Loading and processing LUT: {args.lut_file}")
    gp, dc, LUTalph, LUTthet, LUTnEffs = load_and_process_csv_lut(args.lut_file, p)
    
    # Note: LUT plots with design overlay will be generated after longitudinal design
    
    # Step 3: Create longitudinal design
    print("Creating longitudinal grating design...")
    longit_design = create_longitudinal_design(p, phi_fs, EfieldAmp, theta0, Xwg, Ywg, waist_info, gp, dc, LUTalph, LUTthet)
    
    # Step 3.5: Generate LUT 2D plots with design overlay
    print("Generating LUT 2D plots with design overlay...")
    lut_figures = plot_lut_2d(gp, dc, LUTalph, LUTthet, LUTnEffs, p, show_plots=False, 
                              combined_only=True, design_data=longit_design)
    
    # Step 4: Plot results
    print("Generating plots...")
    figures = plot_design_results(p, longit_design)
    
    # Step 4.5: Generate comprehensive grating design analysis plots
    print("Generating comprehensive grating design analysis plots...")
    analysis_figures = plot_grating_design_analysis(p, longit_design, gp, dc, LUTalph, LUTthet, LUTnEffs)
    
    # Always save plots, and optionally show them
    # Save LUT plots first (with design overlay)
    for i, fig in enumerate(lut_figures):
        # Since combined_only=True by default, there should only be one combined plot
        plot_name = 'lut_combined_with_design' if i == 0 else f'lut_plot_{i+1}'
        fig_path = os.path.join(args.output_dir, f'{plot_name}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved LUT plot with design overlay: {fig_path}")
        if not args.show_plots:
            plt.close(fig)
    
    # Save design plots
    for i, fig in enumerate(figures):
        fig_path = os.path.join(args.output_dir, f'figure_{i+1}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved design plot: {fig_path}")
        if not args.show_plots:
            plt.close(fig)
    
    # Save analysis plots
    plot_names = ['design_params_analysis', 'lut_contours_with_design', 'design_params_for_paper']
    for i, fig in enumerate(analysis_figures):
        fig_path = os.path.join(args.output_dir, f'{plot_names[i]}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved analysis plot: {fig_path}")
        if not args.show_plots:
            plt.close(fig)
    
    # Show plots if requested
    if args.show_plots:
        plt.show()
    
    # Save design data
    design_data_path = os.path.join(args.output_dir, 'design_data.npz')
    np.savez(design_data_path,
             XwgCenter=longit_design["XwgCenter"],
             desDCxx=longit_design["desDCxx"],
             desperxx=longit_design["desperxx"],
             alpha_limited=longit_design["alpha_limited"],
             E2=longit_design["E2"])
    
    print(f"Design data saved to {design_data_path}")
    print(f"Design complete. Coupling efficiency: {longit_design['nuLim']:.4f}")
    
    # Step 5: Run Tidy3D simulation if requested
    if args.run_tidy3d:
        print("\n======= Running Tidy3D Simulation =======\n")
        
        # Check if sim_config is provided when using --run_tidy3d
        if not args.sim_config:
            print("Error: --sim_config is required when using --run_tidy3d")
            print("Please provide a simulation configuration file with required parameters:")
            print("  --sim_config path/to/sim_config.json")
            print("You can use the provided sample_sim_config.json as a template.")
            sys.exit(1)
        
        # Convert the design results to dictionary for the simulation
        design_results = {
            "XwgCenter": longit_design["XwgCenter"],
            "desDCxx": longit_design["desDCxx"],
            "desperxx": longit_design["desperxx"],
            "alpha_limited": longit_design["alpha_limited"],
            "E2": longit_design["E2"]
        }
        
        # Load simulation parameters from sim_config file
        try:
            print(f"Loading simulation configuration from {args.sim_config}")
            with open(args.sim_config, 'r') as f:
                sim_params = json.load(f)
        except Exception as e:
            print(f"Error loading simulation config file: {e}")
            sys.exit(1)
        
        # Check for required simulation parameters
        required_params = ['wavelength', 'material', 'material_index', 'oxide_index']
        missing_params = [param for param in required_params if param not in sim_params]
        if missing_params:
            print(f"Error: Missing required parameters in simulation config: {', '.join(missing_params)}")
            print("Please ensure your simulation config includes all required parameters:")
            for param in required_params:
                print(f"  - {param}")
            sys.exit(1)
        
        # Override material_json if provided via command line
        if args.material_json:
            material_json_path = args.material_json
            if not os.path.isabs(material_json_path):
                material_json_path = os.path.join(os.getcwd(), material_json_path)
            
            if os.path.exists(material_json_path):
                sim_params['material_json'] = material_json_path
                print(f"Using material JSON from command line: {material_json_path}")
            else:
                raise FileNotFoundError(f"Material JSON file not found: {material_json_path}")
        
        # Generate a task name for the simulation
        wavelength_um = int(round(sim_params["wavelength"]))
        material = sim_params["material"]
        task_name = f"grating_sim_{material}_{wavelength_um}um"
        
        # Check if material parameters are provided
        if not any(k in sim_params for k in ["material_index", "material_csv"]):
            print("\nNote: No material refractive index properties provided in simulation config.")
            print("Using default materials from Tidy3D material library.")
        
        # Create tidy3d output directory if needed
        if not os.path.exists(tidy3d_output_dir):
            os.makedirs(tidy3d_output_dir)
        
        # Run the simulation or just set it up
        print(f"Running Tidy3D simulation...")
        # Run the simulation
        sim_params['lut_file_path'] = args.lut_file
        results = simulate_grating(
            design_results=design_results,
            p=sim_params,
            output_dir=tidy3d_output_dir,
            task_name=task_name,
            show_plots=args.show_plots,
            run_sim=not args.no_simulation
        )
        
        # Always save simulation plots
        if 'setup_figures' in results:
            print("Saving simulation visualization plots...")
            for i, fig in enumerate(results['setup_figures']):
                fig_path = os.path.join(tidy3d_output_dir, f'sim_setup_{i+1}.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {fig_path}")
                if not args.show_plots:
                    plt.close(fig)
                
        if 'results_figures' in results:
            for i, fig in enumerate(results['results_figures']):
                fig_path = os.path.join(tidy3d_output_dir, f'sim_results_{i+1}.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {fig_path}")
                if not args.show_plots:
                    plt.close(fig)
            
        # Report any errors
        if 'error' in results:
            print(f"\nError in simulation: {results['error']}")
            sys.exit(1)
            
        # Ensure plots are displayed if show_plots is True
        if args.show_plots:
            print("\nDisplaying plots. Close plot windows to continue.")
            plt.show(block=True)  # Block until all plot windows are closed
    
    # Step 6: Run transverse design if not skipped
    if not args.skip_transverse:
        print("\n======= Running Transverse Design =======\n")
        
        # Get the required parameters for transverse design
        tapStart = p.get("tapStart", 0)
        tapang = np.arctan((2.844 * waist_info["YwgWaists"][1] / 2) / (waist_info["XwgWaists"][2] - tapStart))
        p["tapang"] = tapang
        
        # Extract indices from LUT - create a properly sized array to match XwgCenter
        XwgCenter = longit_design["XwgCenter"]
        nPts = len(XwgCenter)
        
        # Create a properly sized nEffsFromFit array - can use linear values or extract from LUT
        # For now, use a simple approximation of linear values between 1.5 and 1.7
        nEffsFromFit = np.linspace(1.5, 1.7, nPts)
        
        # Print some debug info
        print(f"nEffsFromFit shape: {nEffsFromFit.shape}, XwgCenter shape: {XwgCenter.shape}")
        print(f"tapStart: {tapStart}, tapang: {tapang * 180 / np.pi} degrees")
        
        # Calculate transverse design
        transverse_results = create_transverse_design(
            Xwg=Xwg,
            Ywg=Ywg,
            phi_fs=phi_fs,
            XwgCenter=XwgCenter,
            XwgWaists=waist_info["XwgWaists"],
            YwgWaists=waist_info["YwgWaists"],
            nEffsFromFit=nEffsFromFit,
            tapStart=tapStart,
            tapang=tapang,
            k0=p["k0"],
            p=p,
            planar_wavefronts=p.get("planar_wavefronts", False),
            desperxx=longit_design["desperxx"],
            desDCxx=longit_design["desDCxx"]
        )
        
        # Get transverse design plots
        trans_figures = plot_transverse_design(transverse_results)
        
        # Always save transverse design plots
        transverse_dir = os.path.join(args.output_dir, 'transverse')
        if not os.path.exists(transverse_dir):
            os.makedirs(transverse_dir)
        
        for i, fig in enumerate(trans_figures):
            fig_path = os.path.join(transverse_dir, f'transverse_design_{i+1}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {fig_path}")
            if not args.show_plots:
                plt.close(fig)
        
        # Show transverse plots if requested
        if args.show_plots:
            plt.show()
        
        # Save transverse design parameters
        trans_data_path = os.path.join(args.output_dir, 'transverse_design.npz')
        np.savez(trans_data_path,
                 r0=transverse_results["r0"],
                 dr=transverse_results["dr"],
                 tapang=transverse_results["tapang"])
        
        print(f"Transverse design data saved to {trans_data_path}")
        print(f"Transverse design complete. r0={transverse_results['r0']:.2f}, dr={transverse_results['dr']:.4f}")
        
        # Prepare and save perDC_interp_initial.mat files
        # Function to format values for filenames
        def forfile(val):
            return str(val).replace('.', 'p').replace('-', 'm')
        
        # Create a workspace dictionary with design parameters
        wavelength_nm = round(p["lambda"] * 1e3)
        thetaIncDegrees = p.get("thetaIncDegrees", 0)
        w0x = p.get("w0x", 0)
        w0y = p.get("w0y", 0)
        z0 = p.get("z0", 0)
        zOffx = p.get("zOffx", 0)
        zOffy = p.get("zOffy", 0)
        
        # Generate filename string
        pString_small = f"lam{wavelength_nm}nm_thet{forfile(thetaIncDegrees)}_w0x{forfile(w0x)}_w0y{forfile(w0y)}_z0{forfile(z0)}um"
        pString_small += f"_zOffXY{forfile(zOffx)}_{forfile(zOffy)}_tapStart{forfile(tapStart)}"
        
        # Add additional parameters if present
        if "note" in p:
            pString_small += p["note"]
        
        # Prepare workspace data
        workspace_data = {
            "lambda": p["lambda"],
            "nOx": p.get("nOx", 1.44),
            "tOx": p.get("tOx", 0),
            "w0x": w0x,
            "w0y": w0y,
            "x0": p.get("x0", 0),
            "z0": z0,
            "thetaIncDegrees": thetaIncDegrees,
            "tapl": XwgCenter[0] - tapStart,  # Taper length
            "lgrat": XwgCenter[-1] - XwgCenter[0],  # Grating length
            "hgrat": 2.844 * waist_info["YwgWaists"][1],  # Grating height
            "r0": transverse_results["r0"],
            "dr": transverse_results["dr"],
            "xPtsRadsQ": transverse_results["xPtsRadsQ"],
            "yPtsRadsQ_R": transverse_results["yPtsRadsQ_R"],
            "yPtsRadsQ_A": transverse_results["yPtsRadsQ_A"],
            "tapang": tapang,
            "pper": np.polyfit(XwgCenter, longit_design["desperxx"], p.get("fitorder", 7)),
            "pDC": np.polyfit(XwgCenter, longit_design["desDCxx"], p.get("fitorder", 7)),
            "tapStart": tapStart,
            "xPtsBothFull": np.linspace(np.min(XwgCenter), np.max(XwgCenter), 200),
            "p": p
        }
        
        # Calculate interpolated values
        interp_per = np.interp(workspace_data["xPtsBothFull"], XwgCenter, longit_design["desperxx"] * 1e3)
        interp_dc = np.interp(workspace_data["xPtsBothFull"], XwgCenter, longit_design["desDCxx"])
        
        workspace_data["yPtsPerFull"] = interp_per
        workspace_data["yPtsDCFull"] = interp_dc
        workspace_data["nEffsFromFit"] = nEffsFromFit
        
        # Create a transverse design directory
        transverse_dir = os.path.join(args.output_dir, 'transverse')
        if not os.path.exists(transverse_dir):
            os.makedirs(transverse_dir)
        
        # Save as .mat files
        mat_path = os.path.join(transverse_dir, f'perDC_interp_initial_{pString_small}.mat')
        mat_path_v7p3 = os.path.join(transverse_dir, f'perDC_interp_initial_v7p3_{pString_small}.mat')
        
        print(f"Saving perDC_interp_initial files...")
        print(transverse_results)
        print(f"Saving perDC_interp_initial files to {mat_path}")
        print(f"Saving perDC_interp_initial files to {mat_path_v7p3}")
        
        # Save using scipy.io.savemat (standard method)
        savemat(mat_path, workspace_data)
        
        # Save v7.3 compatible file with better error handling
        try:
            # Remove existing file if it exists to avoid corruption issues
            if os.path.exists(mat_path_v7p3):
                os.remove(mat_path_v7p3)
                print(f"Removed existing file: {mat_path_v7p3}")
            
            # Write new file using hdf5storage
            hdf5storage.write(workspace_data, path='/', filename=mat_path_v7p3, 
                            matlab_compatible=True, store_python_metadata=False)
            print(f"Successfully saved v7.3 file: {mat_path_v7p3}")
            
        except Exception as e:
            print(f"Error saving v7.3 file with hdf5storage: {e}")
            print("Attempting to save with scipy.io.savemat as fallback...")
            try:
                # Fallback to scipy savemat without compression
                savemat(mat_path_v7p3, workspace_data, do_compression=False)
                print(f"Successfully saved v7.3 file using fallback method: {mat_path_v7p3}")
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")
                print("Only the standard .mat file was saved successfully.")
        
        print(f"perDC_interp_initial files saved to:")
        print(f"  - {mat_path}")
        print(f"  - {mat_path_v7p3}")

if __name__ == "__main__":
    main() 