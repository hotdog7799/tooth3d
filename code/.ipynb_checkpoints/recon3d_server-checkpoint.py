"""
Interactive PSF Selection for 3D Reconstruction

This script matches your original workflow:
1. Load PSF image files from directory
2. Display them with indices 
3. Interactive selection (like psf_list[25:25+15+15+15:15])
4. Process and cache the PSF stack
5. Run reconstruction with selectable regularizers

Perfect match to your original notebook workflow with added regularizer selection!
"""

from psf_preprocessing import PSFPreprocessor
from admm_3d_refactored import ADMM3D
from regularizers import TV3DRegularizer, L1Regularizer, CenterWeightedRegularizer, AnisotropicDiffusionRegularizer
import os

def interactive_psf_selection():
    """Interactive PSF selection matching your original workflow."""
    
    print("=== Interactive PSF Selection for 3D Reconstruction ===")
    
    # Step 1: Set up paths (modify these for your setup)
    psf_directory = input("Enter PSF image directory path (or press Enter for '/mnt/NAS/Grants/24_AIOBIO/2501_data/calib/mask_1/whole_psf'): ").strip()
    if not psf_directory:
        psf_directory = "/mnt/NAS/Grants/24_AIOBIO/2501_data/calib/mask_1/whole_psf/"
    
    print(f"\nUsing PSF directory: {psf_directory}")
    
    # Step 2: Initialize preprocessor
    preprocessor = PSFPreprocessor(
        psf_directory=psf_directory,
        dummy_strategy="symmetric_padding",
        dummy_layers_between=0,
        dummy_layers_boundary=0
    )
    
    # Step 3: Discover PSF image files
    print("\nDiscovering PSF image files...")
    try:
        psf_files = preprocessor.discover_psf_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    if not psf_files:
        print("No PSF image files found!")
        print("Supported formats: .jpg, .jpeg, .png, .tiff, .tif")
        return None
    
    # Step 4: Display PSFs nicely (like your original code)
    print(f"\nFound {len(psf_files)} PSF image files:")
    preprocessor.print_file_with_num(num_per_line=5)
    
    # Step 5: Interactive selection
    print("\n" + "="*60)
    print("PSF SELECTION OPTIONS:")
    print("="*60)
    print("1. Use preset pattern (like your psf_list[25:25+15+15+15:15])")
    print("2. Enter custom range (e.g., '0,5,10' or '25:40')")
    print("3. Enter slice notation (e.g., '25:40', '0:10:2', '25:25+45:15')")
    print("4. Quick test with first few PSFs")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        # Preset pattern
        if len(psf_files) >= 50:
            # Your typical pattern: psf_list[25:25+15+15+15:15]
            start = 25
            step = 15
            selected_indices = []
            for i in range(3):  # 3 groups of 15
                group_start = start + i * step
                group_end = group_start + step
                selected_indices.extend(range(group_start, min(group_end, len(psf_files))))
            print(f"Using preset pattern: 3 groups of 15 starting from index 25")
        else:
            selected_indices = list(range(min(3, len(psf_files))))
            print(f"Not enough PSFs for preset pattern, using first {len(selected_indices)}")
    
    elif choice == "2":
        # Custom range
        indices_str = input("Enter PSF indices (comma-separated, e.g., '0,5,10,15'): ").strip()
        try:
            selected_indices = [int(x.strip()) for x in indices_str.split(',')]
            # Validate indices
            selected_indices = [i for i in selected_indices if 0 <= i < len(psf_files)]
        except ValueError:
            print("Invalid input. Using first 3 PSFs.")
            selected_indices = list(range(min(3, len(psf_files))))
    
    elif choice == "3":
        # Slice notation
        slice_str = input("Enter slice notation (e.g., '25:40', '0:15:3', '25:25+45:15'): ").strip()
        try:
            # Handle your special notation like '25:25+45:15'
            if '+' in slice_str:
                # Parse something like '25:25+45:15'
                parts = slice_str.split(':')
                start = int(parts[0])
                if '+' in parts[1]:
                    # Handle '25+45'
                    end_parts = parts[1].split('+')
                    end = int(end_parts[0]) + int(end_parts[1])
                else:
                    end = int(parts[1])
                step = int(parts[2]) if len(parts) > 2 else 1
            else:
                # Standard slice notation
                parts = slice_str.split(':')
                start = int(parts[0])
                end = int(parts[1]) if len(parts) > 1 else len(psf_files)
                step = int(parts[2]) if len(parts) > 2 else 1
            
            selected_indices = list(range(start, min(end, len(psf_files)), step))
            
        except (ValueError, IndexError):
            print("Invalid slice notation. Using first 3 PSFs.")
            selected_indices = list(range(min(3, len(psf_files))))
    
    elif choice == "4":
        # Quick test
        selected_indices = list(range(min(3, len(psf_files))))
        print("Using first 3 PSFs for quick test")
    
    else:
        print("Invalid choice. Using first 3 PSFs.")
        selected_indices = list(range(min(3, len(psf_files))))
    
    # Step 6: Show selection
    print(f"\nSelected {len(selected_indices)} PSFs:")
    for i, idx in enumerate(selected_indices):
        print(f"  {i}: {os.path.basename(psf_files[idx])} (index {idx})")
    
    confirm = input(f"\nProceed with these {len(selected_indices)} PSFs? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Selection cancelled.")
        return None
    
    # Step 7: Process PSF stack (with caching)
    print("\nProcessing PSF stack...")
    psf_stack_path, labels = preprocessor.process_psf_stack(
        selected_indices=selected_indices,
        use_cache=True,
        force_rebuild=False
    )
    
    print(f"✓ PSF stack ready: {psf_stack_path}")
    print(f"✓ Generated {len(labels)} total layers (including dummies)")
    
    return psf_stack_path, labels, preprocessor

def run_reconstruction_with_selected_psfs(psf_stack_path, labels):
    """Run reconstruction with the selected and processed PSF stack."""
    
    print("\n" + "="*60)
    print("RECONSTRUCTION SETUP")
    print("="*60)
    
    # Get raw image path
    raw_image_path = input("Enter raw image path (or press Enter for '../raw_image/simulation/simulated_raw_sum_type_2_0629_2354.png'): ").strip()
    if not raw_image_path:
        raw_image_path = "../raw_image/simulation/simulated_raw_sum_type_2_0629_2354.png"
    
    print(f"Using raw image: {raw_image_path}")
    
    # Check if file exists
    if not os.path.exists(raw_image_path):
        print(f"Warning: Raw image file not found: {raw_image_path}")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed not in ['y', 'yes']:
            return None
    
    # Select regularizer for tooth reconstruction
    reg_type = select_regularizer()
    
    # Ask about figure display/save preferences
    print("\n" + "="*40)
    print("FIGURE DISPLAY OPTIONS")
    print("="*40)
    print("1. Save figures only (no display) - Recommended for batch processing")
    print("2. Display and save figures - Good for monitoring progress")
    print("3. No figures at all - Fastest processing")
    
    fig_choice = input("Choose figure option (1-3, default=1): ").strip()
    if fig_choice == "2":
        disp_figs = 200  # Show every 200 iterations
        save_fig = True
        show_figs = True
        print("✓ Will display AND save figures every 200 iterations")
    elif fig_choice == "3":
        disp_figs = 0    # Never display
        save_fig = False
        show_figs = False
        print("✓ No figures will be generated")
    else:  # Default option 1
        disp_figs = 200  # Set interval for saving (but not displaying)
        save_fig = True
        show_figs = False  # This is the key fix!
        print("✓ Will save figures every 200 iterations (no display)")
    
    # Ask about number of iterations
    max_iter_input = input("Enter max iterations (default=1000): ").strip()
    try:
        max_iter = int(max_iter_input) if max_iter_input else 1000
    except ValueError:
        max_iter = 1000
        print("Invalid input, using default 1000 iterations")
    
    print(f"✓ Will run for {max_iter} iterations")
    
    # ADMM Configuration
    config = {
        # File paths
        'path_ref': 0,
        'psf_file': psf_stack_path,
        'img_file': raw_image_path,
    
        'save_dir': "../recon_result/",
        # 'save_dir': "/mnt/NAS/Grants/24_AIOBIO/2503_data/3d_recon/ph2/step2/",  # if path_ref=1, start with /
        'save_every': 100,  # Save image stack as .mat every N iterations. Use 0 to never save (except for at the end);
        'save_fig': save_fig,
        'show_figs': show_figs,
        # Data Setup
        'color_to_process': 'mono',  # 'red','green','blue', or 'mono'. If raw file is mono, this is ignored
        'image_bias': 0,  # If camera has bias, subtract from measurement file.
        'psf_bias': 0,  # if PSF needs sensor bias removed, put that here.
        
        'raw_bias': 0,
        
        'lateral_downsample': 8,  # down sample image
        'axial_downsample': 1,  # Axial averageing of impulse stack. Must be multiple of 2 and >= 1.
        'start_z': 0,  # First plane to reconstruct. 1 indexed, as is tradition.
        'end_z': 0,  # Last plane to reconstruct. If set to 0, use last plane in file.
        # GPU setup
        'useGPU': True,
        'numGPU': 0,
        # Recon Parameters
        'max_iter': max_iter,
        'disp_figs': disp_figs,
        'print_interval': 100, # Print cost every N iterations. Default 1. If set to 0, don't print.
        'regularizer': '3dtvz',
        # Optimization Parameters
        'mu1': 0.5,  # 0.25
        'mu2': 0.68,  # 0.68,
        'mu3': 4.0,  # 3.5
        'tau': 6.0e-4,  # 6.0e-4 0.008 sparsity parameter for TV
        'tau_z' : 6.0e-3,#6.0e-6
        'tau_n': 0.06,  # sparsity param for native sparsity
        # Tuning Parameter
        'autotune': 1,  # 1:auto-find mu every step. 0:defined values. If set to N>1, tune for N steps then stop.
        'mu_inc': 1.2,  # Inrement and decrement values for mu during autotune.
        'mu_dec': 1.2,  #
        'resid_tol': 1.5,  # Primal/dual gap tolerance.
        # Display setup
        'roih': 600,
        'roiw': 600,
        'display_norm_method': 'log',
        'beta_z': 10
    }
    
    print("\nStarting ADMM reconstruction...")
    
    # Create regularizer
    print(f"Initializing {reg_type} regularizer...")
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    regularizer = create_regularizer(reg_type, device)
    
    # Initialize and run ADMM with selected regularizer
    solver = ADMM3D(config, regularizer=regularizer)
    
    # Pass PSF labels to solver for better visualization
    solver.psf_labels = labels
    
    # Set the show_figs attribute to control display behavior
    solver.show_figs = show_figs
    
    final_reconstruction = solver.admm()
    
    print("\n" + "="*60)
    print("RECONSTRUCTION COMPLETE!")
    print("="*60)
    print(f"Results saved in: {solver.save_path}")
    
    # Automatically save convergence metrics without asking
    if hasattr(solver, 'plot_iteration_metrics'):
        print("Saving convergence metrics...")
        solver.plot_iteration_metrics()  # Display metrics
    
    return final_reconstruction, solver

def select_regularizer():
    """Interactive regularizer selection for tooth reconstruction."""
    
    print("\n" + "="*60)
    print("REGULARIZER SELECTION FOR TOOTH RECONSTRUCTION")
    print("="*60)
    print("Choose a regularizer optimized for tooth structures:")
    print()
    print("1. 3D Total Variation (3DTV) - Standard edge-preserving regularization")
    print("   • Good for: General smooth structures with edges")
    print("   • Characteristics: Preserves edges while smoothing regions")
    print()
    print("2. Center-Weighted TV - Tooth-optimized spatial regularization")
    print("   • Good for: Tooth at center with smooth edges")
    print("   • Characteristics: Less regularization at center, more at edges")
    print("   • Recommended for: Occlusal plane tooth reconstruction")
    print()
    print("3. L1 Sparsity - Promotes sparse reconstructions")
    print("   • Good for: Clear boundaries and sparse structures")
    print("   • Characteristics: Promotes sparsity, good for distinct features")
    print()
    print("4. Anisotropic Diffusion - Advanced edge-preserving smoothing")
    print("   • Good for: Smooth tooth surfaces with sharp edges")
    print("   • Characteristics: Adaptive smoothing that preserves edges")
    print("   • Best for: High-quality tooth reconstruction")
    print()
    print("5. Hybrid (3DTV + L1) - Combines edge preservation with sparsity")
    print("   • Good for: Complex tooth structures")
    print("   • Characteristics: Edge preservation + sparsity promotion")
    
    choice = input("\nChoose regularizer (1-5, default=2 for tooth): ").strip()
    
    if choice == "1":
        print("✓ Selected: 3D Total Variation")
        return "3dtv"
    elif choice == "3":
        print("✓ Selected: L1 Sparsity")
        return "l1"
    elif choice == "4":
        print("✓ Selected: Anisotropic Diffusion")
        return "anisotropic"
    elif choice == "5":
        print("✓ Selected: Hybrid (3DTV + L1)")
        return "hybrid"
    else:  # Default choice 2
        print("✓ Selected: Center-Weighted TV (Recommended for tooth)")
        return "center_weighted"

def create_regularizer(reg_type, device):
    """Create the selected regularizer with improved parameters."""
    
    if reg_type == "3dtv":
        return TV3DRegularizer(
            device=device,
            tau=6e-4,      # Standard lateral regularization
            tau_z=6e-10    # Lighter axial regularization
        )
    
    elif reg_type == "l1":
        return L1Regularizer(
            device=device,
            tau=1e-3       # L1 sparsity weight
        )
    
    elif reg_type == "center_weighted":
        return CenterWeightedRegularizer(
            device=device,
            tau=6e-4,          # Base lateral regularization
            tau_z=6e-10,       # Base axial regularization
            center_weight=0.7, # FIXED: More conservative weighting
            edge_weight=1.0    # Higher regularization at edges
        )
    
    elif reg_type == "anisotropic":
        return AnisotropicDiffusionRegularizer(
            device=device,
            tau=6e-4,           # Base lateral regularization
            tau_z=6e-10,        # Base axial regularization
            edge_threshold=0.1  # Edge detection threshold
        )
    
    elif reg_type == "hybrid":
        # FIXED: Use standard 3DTV instead of problematic center-weighted
        return TV3DRegularizer(
            device=device,
            tau=6e-4,      # Standard parameters
            tau_z=6e-10    
        )
    
    else:
        # Default to standard 3DTV for reliability
        return TV3DRegularizer(
            device=device,
            tau=6e-4,
            tau_z=6e-10
        )

def main():
    """Main interactive workflow."""
    
    # PSF selection
    result = interactive_psf_selection()
    if result is None:
        return
    
    psf_stack_path, labels, preprocessor = result
    
    # Ask if user wants to run reconstruction
    run_recon = input("\nRun 3D reconstruction now? (y/n): ").strip().lower()
    if run_recon in ['y', 'yes']:
        reconstruction, solver = run_reconstruction_with_selected_psfs(psf_stack_path, labels)
        return reconstruction, solver, preprocessor
    else:
        print(f"PSF stack ready at: {psf_stack_path}")
        print("You can run reconstruction later using this processed PSF stack.")
        return None, None, preprocessor

if __name__ == "__main__":
    result = main()
    print("\nDone!") 