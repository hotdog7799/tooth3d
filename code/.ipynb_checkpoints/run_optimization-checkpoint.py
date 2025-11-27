"""
Complete 3D Reconstruction Optimization Pipeline

This script combines PSF selection and parameter optimization into one smooth workflow.
Perfect for running on servers - just set your paths and let it run!
"""

import os
import json
from datetime import datetime
from interactive_psf_selection import interactive_psf_selection, run_reconstruction_with_selected_psfs
from parameter_optimization_batch import ParameterOptimizer


def create_optimization_config_for_psf(psf_stack_path, raw_image_path):
    """Create optimization configuration for the selected PSF stack."""
    
    base_config = {
        'path_ref': 0,
        'psf_file': psf_stack_path,
        'img_file': raw_image_path,
        'save_every': 0,
        'save_fig': True,
        'show_figs': False,
        'color_to_process': 'mono',
        'image_bias': 0,
        'psf_bias': 0,
        'raw_bias': 0,
        'lateral_downsample': 8,
        'axial_downsample': 1,
        'start_z': 0,
        'end_z': 0,
        'useGPU': True,
        'numGPU': 0,
        'disp_figs': 0,
        'print_interval': 0,
        'regularizer': '3dtvz',
        'autotune': 1,
        'mu_inc': 1.2,
        'mu_dec': 1.2,
        'resid_tol': 1.5,
        'roih': 600,
        'roiw': 600,
        'display_norm_method': 'log',
        'beta_z': 10
    }
    
    # Optimization configuration focused on tooth reconstruction
    opt_config = {
        'method': 'grid_search',
        'parameter_space': {
            # Start with focused parameter ranges based on your experience
            'mu1': [0.5, 0.75, 1.0],
            'mu2': [0.68, 0.8, 1.0],
            'mu3': [3.0, 4.0, 5.0],
            'tau': [3e-4, 6e-4, 1e-3],
            'tau_z': [6e-5, 1e-4],
                         'regularizer': ['3dtv', 'anisotropic'],
            'max_iter': [1000, 1500],
        }
    }
    
    return base_config, opt_config


def quick_optimization_setup():
    """Quick setup for optimization with minimal user input."""
    
    print("=== Quick 3D Reconstruction Optimization Setup ===")
    print()
    
    # Default paths (modify these for your setup)
    default_psf_dir = "/mnt/NAS/Grants/24_AIOBIO/2501_data/calib/mask_1/whole_psf/"
    default_raw_image = "../raw_image/simulation/simulated_raw_sum_type_2_0629_2354.png"
    
    print("Step 1: PSF Selection")
    print("====================")
    
    # PSF directory
    psf_dir = input(f"PSF directory (Enter for default: {default_psf_dir}): ").strip()
    if not psf_dir:
        psf_dir = default_psf_dir
    
    if not os.path.exists(psf_dir):
        print(f"Error: PSF directory not found: {psf_dir}")
        return None
    
    # Raw image
    raw_image = input(f"Raw image path (Enter for default: {default_raw_image}): ").strip()
    if not raw_image:
        raw_image = default_raw_image
    
    if not os.path.exists(raw_image):
        print(f"Error: Raw image not found: {raw_image}")
        return None
    
    print(f"\nUsing PSF directory: {psf_dir}")
    print(f"Using raw image: {raw_image}")
    
    return psf_dir, raw_image


def run_full_optimization_pipeline():
    """Run the complete optimization pipeline."""
    
    # Quick setup
    setup_result = quick_optimization_setup()
    if setup_result is None:
        return
    
    psf_dir, raw_image = setup_result
    
    print("\nStep 2: PSF Processing and Selection")
    print("=====================================")
    
    # Use the interactive PSF selection but with predefined choices for server use
    from psf_preprocessing import PSFPreprocessor
    
    preprocessor = PSFPreprocessor(
        psf_directory=psf_dir,
        dummy_strategy="symmetric_padding",
        dummy_layers_between=0,
        dummy_layers_boundary=0
    )
    
    # Discover PSF files
    try:
        psf_files = preprocessor.discover_psf_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not psf_files:
        print("No PSF image files found!")
        return
    
    print(f"Found {len(psf_files)} PSF files")
    preprocessor.print_file_with_num(num_per_line=5)
    
    # PSF selection strategy
    print("\nPSF Selection Strategy:")
    print("1. Use your typical pattern (25:25+45:15)")
    print("2. Use first 15 PSFs (for testing)")
    print("3. Custom selection")
    
    choice = input("Choose strategy (1-3, default=1): ").strip()
    
    if choice == "2":
        selected_indices = list(range(min(15, len(psf_files))))
    elif choice == "3":
        indices_str = input("Enter PSF indices (comma-separated): ").strip()
        try:
            selected_indices = [int(x.strip()) for x in indices_str.split(',')]
            selected_indices = [i for i in selected_indices if 0 <= i < len(psf_files)]
        except ValueError:
            print("Invalid input. Using default strategy.")
            selected_indices = list(range(25, min(25+45, len(psf_files)), 15))
    else:
        # Default: your typical pattern
        if len(psf_files) >= 70:
            selected_indices = list(range(25, min(70, len(psf_files)), 15))
        else:
            selected_indices = list(range(min(15, len(psf_files))))
    
    print(f"Selected {len(selected_indices)} PSFs: {selected_indices}")
    
    # Process PSF stack
    print("\nProcessing PSF stack...")
    psf_stack_path, labels = preprocessor.process_psf_stack(
        selected_indices=selected_indices,
        use_cache=True,
        force_rebuild=False
    )
    
    print(f"PSF stack ready: {psf_stack_path}")
    
    print("\nStep 3: Parameter Optimization")
    print("===============================")
    
    # Choose optimization intensity
    print("Optimization intensity:")
    print("1. Quick test (few parameters, ~30 min)")
    print("2. Standard optimization (~2-4 hours)")
    print("3. Thorough optimization (~6-12 hours)")
    
    intensity = input("Choose intensity (1-3, default=2): ").strip()
    
    # Create optimization config based on intensity
    base_config, opt_config = create_optimization_config_for_psf(psf_stack_path, raw_image)
    
    if intensity == "1":
        # Quick test
                 opt_config['parameter_space'] = {
             'mu1': [0.5, 0.75],
             'mu2': [0.68, 0.8],
             'mu3': [3.0, 4.0],
             'tau': [6e-4],
             'tau_z': [6e-5],
             'regularizer': ['3dtv'],  # Quick test with basic TV
             'max_iter': [500],
         }
    elif intensity == "3":
        # Thorough
                 opt_config['parameter_space'] = {
             'mu1': [0.25, 0.5, 0.75, 1.0],
             'mu2': [0.5, 0.68, 0.8, 1.0],
             'mu3': [2.0, 3.0, 4.0, 5.0],
             'tau': [1e-4, 3e-4, 6e-4, 1e-3],
             'tau_z': [6e-6, 6e-5, 1e-4],
             'regularizer': ['3dtv', 'anisotropic'],  # TV and anisotropic only
             'max_iter': [1000, 1500, 2000],
         }
    # Standard is already set in create_optimization_config_for_psf
    
    # Calculate estimated time
    from itertools import product
    total_combinations = len(list(product(*opt_config['parameter_space'].values())))
    estimated_minutes = total_combinations * 3  # rough estimate: 3 min per experiment
    
    print(f"\nOptimization will test {total_combinations} parameter combinations")
    print(f"Estimated time: {estimated_minutes//60}h {estimated_minutes%60}min")
    
    confirm = input("\nStart optimization? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Optimization cancelled.")
        return
    
    # Run optimization
    print("\nStarting parameter optimization...")
    optimizer = ParameterOptimizer(base_config, opt_config)
    
    try:
        optimizer.run_optimization()
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED!")
        print("="*60)
        
        if optimizer.best_result:
            print(f"Best score: {optimizer.best_result['metrics']['composite_score']:.4f}")
            print("Best parameters:")
            for param, value in optimizer.best_result['parameters'].items():
                print(f"  {param}: {value}")
            
            # Run final reconstruction with best parameters
            print("\nRunning final reconstruction with best parameters...")
            final_reconstruction, final_solver = optimizer.run_with_best_parameters(display_results=False)
            
            print(f"\nAll results saved in: {optimizer.results_dir}")
            
            # Create summary file for easy access
            summary_file = os.path.join(optimizer.results_dir, 'OPTIMIZATION_SUMMARY.txt')
            with open(summary_file, 'w') as f:
                f.write("3D RECONSTRUCTION OPTIMIZATION SUMMARY\n")
                f.write("="*40 + "\n\n")
                f.write(f"PSF directory: {psf_dir}\n")
                f.write(f"Raw image: {raw_image}\n")
                f.write(f"PSF stack: {psf_stack_path}\n")
                f.write(f"Selected PSF indices: {selected_indices}\n\n")
                f.write(f"Total experiments: {len(optimizer.results)}\n")
                f.write(f"Best score: {optimizer.best_result['metrics']['composite_score']:.4f}\n\n")
                f.write("BEST PARAMETERS:\n")
                for param, value in optimizer.best_result['parameters'].items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"\nBest result experiment: {optimizer.best_result['experiment_id']}\n")
                f.write(f"Final reconstruction saved in: final_best_reconstruction/\n")
            
            print(f"Summary saved: {summary_file}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Partial results saved in: {optimizer.results_dir}")
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()


def run_auto_optimization():
    """Run optimization with predefined settings (for server/nohup execution)."""
    
    print("3D Reconstruction Auto Optimization (Server Mode)")
    print("=" * 50)
    
    # MODIFY THESE PATHS FOR YOUR SETUP
    PSF_DIR = "/mnt/NAS/Grants/24_AIOBIO/2501_data/calib/mask_1/whole_psf/"
    RAW_IMAGE = "../raw_image/simulation/simulated_raw_sum_type_2_0629_2354.png"  # UPDATE THIS PATH
    
    # AUTO SETTINGS
    PSF_SELECTION_STRATEGY = 1  # 1: typical pattern, 2: first 15, 3: custom
    OPTIMIZATION_INTENSITY = 2  # 1: quick, 2: standard, 3: thorough
    
    print(f"PSF Directory: {PSF_DIR}")
    print(f"Raw Image: {RAW_IMAGE}")
    print(f"PSF Selection: Strategy {PSF_SELECTION_STRATEGY}")
    print(f"Optimization Intensity: {OPTIMIZATION_INTENSITY}")
    print()
    
    # Check paths
    if not os.path.exists(PSF_DIR):
        print(f"ERROR: PSF directory not found: {PSF_DIR}")
        print("Please update PSF_DIR in the script.")
        return
    
    if not os.path.exists(RAW_IMAGE):
        print(f"ERROR: Raw image not found: {RAW_IMAGE}")
        print("Please update RAW_IMAGE in the script.")
        return
    
    # PSF Processing
    print("Step 1: PSF Processing and Selection")
    print("=" * 40)
    
    from psf_preprocessing import PSFPreprocessor
    
    preprocessor = PSFPreprocessor(
        psf_directory=PSF_DIR,
        dummy_strategy="symmetric_padding",
        dummy_layers_between=0,
        dummy_layers_boundary=0
    )
    
    try:
        psf_files = preprocessor.discover_psf_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not psf_files:
        print("No PSF image files found!")
        return
    
    print(f"Found {len(psf_files)} PSF files")
    
    # Automatic PSF selection based on strategy
    if PSF_SELECTION_STRATEGY == 2:
        selected_indices = list(range(min(15, len(psf_files))))
        print("Using first 15 PSFs for testing")
    elif PSF_SELECTION_STRATEGY == 3:
        # Custom selection - modify this list as needed
        custom_indices = [25, 40, 55]  # MODIFY THIS FOR CUSTOM SELECTION
        selected_indices = [i for i in custom_indices if 0 <= i < len(psf_files)]
        print(f"Using custom PSF selection: {selected_indices}")
    else:
        # Default: typical pattern (25:70:15)
        if len(psf_files) >= 56:
            selected_indices = list(range(30, min(56, len(psf_files)), 1))
        else:
            selected_indices = list(range(min(15, len(psf_files))))
        print(f"Using typical pattern: {selected_indices}")
    
    print(f"Selected {len(selected_indices)} PSFs: {selected_indices}")
    
    # Process PSF stack
    print("\nProcessing PSF stack...")
    psf_stack_path, labels = preprocessor.process_psf_stack(
        selected_indices=selected_indices,
        use_cache=True,
        force_rebuild=False
    )
    
    print(f"PSF stack ready: {psf_stack_path}")
    
    # Parameter Optimization
    print("\nStep 2: Parameter Optimization")
    print("=" * 40)
    
    base_config, opt_config = create_optimization_config_for_psf(psf_stack_path, RAW_IMAGE)
    
    # Set optimization intensity automatically
    if OPTIMIZATION_INTENSITY == 1:
        # Quick test
        opt_config['parameter_space'] = {
            'mu1': [0.5, 0.75],
            'mu2': [0.68, 0.8],
            'mu3': [3.0, 4.0],
            'tau': [6e-4],
            'tau_z': [6e-5],
            'regularizer': ['3dtv'],
            'max_iter': [500],
        }
        print("Using Quick test mode")
    elif OPTIMIZATION_INTENSITY == 3:
        # Thorough
        opt_config['parameter_space'] = {
            'mu1': [0.25, 0.5, 0.75, 1.0],
            'mu2': [0.5, 0.68, 0.8, 1.0],
            'mu3': [2.0, 3.0, 4.0, 5.0],
            'tau': [1e-4, 3e-4, 6e-4, 1e-3],
            'tau_z': [6e-6, 6e-5, 1e-4],
            'regularizer': ['3dtv', 'anisotropic'],
            'max_iter': [1000, 1500, 2000],
        }
        print("Using Thorough optimization mode")
    else:
        # Standard (already set in create_optimization_config_for_psf)
        print("Using Standard optimization mode")
    
    # Calculate estimated time
    from itertools import product
    total_combinations = len(list(product(*opt_config['parameter_space'].values())))
    estimated_minutes = total_combinations * 3
    
    print(f"Will test {total_combinations} parameter combinations")
    print(f"Estimated time: {estimated_minutes//60}h {estimated_minutes%60}min")
    print()
    
    # Start optimization
    print("Starting parameter optimization...")
    optimizer = ParameterOptimizer(base_config, opt_config)
    
    try:
        optimizer.run_optimization()
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED!")
        print("="*60)
        
        if optimizer.best_result:
            print(f"Best score: {optimizer.best_result['metrics']['composite_score']:.4f}")
            print("Best parameters:")
            for param, value in optimizer.best_result['parameters'].items():
                print(f"  {param}: {value}")
            
            # Run final reconstruction
            print("\nRunning final reconstruction with best parameters...")
            final_reconstruction, final_solver = optimizer.run_with_best_parameters(display_results=False)
            
            print(f"\nAll results saved in: {optimizer.results_dir}")
            
            # Create summary
            summary_file = os.path.join(optimizer.results_dir, 'OPTIMIZATION_SUMMARY.txt')
            with open(summary_file, 'w') as f:
                f.write("3D RECONSTRUCTION OPTIMIZATION SUMMARY\n")
                f.write("="*40 + "\n\n")
                f.write(f"PSF directory: {PSF_DIR}\n")
                f.write(f"Raw image: {RAW_IMAGE}\n")
                f.write(f"PSF stack: {psf_stack_path}\n")
                f.write(f"Selected PSF indices: {selected_indices}\n\n")
                f.write(f"Total experiments: {len(optimizer.results)}\n")
                f.write(f"Best score: {optimizer.best_result['metrics']['composite_score']:.4f}\n\n")
                f.write("BEST PARAMETERS:\n")
                for param, value in optimizer.best_result['parameters'].items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"\nBest result experiment: {optimizer.best_result['experiment_id']}\n")
                f.write(f"Final reconstruction saved in: final_best_reconstruction/\n")
            
            print(f"Summary saved: {summary_file}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Partial results saved in: {optimizer.results_dir}")
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with auto mode detection."""
    import sys
    
    # Check if running in background/nohup mode
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        run_auto_optimization()
    else:
        # Check if we're running in an interactive terminal
        try:
            if not sys.stdin.isatty():
                # Not running in interactive terminal (nohup/background)
                print("Detected background execution (nohup mode)")
                print("Switching to auto optimization mode...")
                print()
                run_auto_optimization()
            else:
                # Interactive mode
                print("3D Reconstruction Complete Optimization Pipeline")
                print("This will:")
                print("1. Help you select PSFs")
                print("2. Automatically optimize parameters")
                print("3. Run final reconstruction with best parameters")
                print("4. Save all results with detailed logs")
                print()
                run_full_optimization_pipeline()
        except:
            # Fallback to auto mode if detection fails
            print("Cannot detect input mode, using auto optimization...")
            run_auto_optimization()


if __name__ == "__main__":
    main() 