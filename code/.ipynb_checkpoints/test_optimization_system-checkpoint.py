"""
Test Script for Parameter Optimization System

This script performs a quick test to ensure the optimization system is working correctly.
Use this before running the full optimization on the server.
"""

import os
import sys
import time
import numpy as np
import torch
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from parameter_optimization_batch import ParameterOptimizer
        print("✓ parameter_optimization_batch imported successfully")
        
        from psf_preprocessing import PSFPreprocessor
        print("✓ psf_preprocessing imported successfully")
        
        from admm_3d_refactored import ADMM3D
        print("✓ admm_3d_refactored imported successfully")
        
        from regularizers import TV3DRegularizer, AnisotropicDiffusionRegularizer
        print("✓ regularizers imported successfully")
        
        import pandas as pd
        print("✓ pandas imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and memory."""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        # Test GPU memory
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"✓ GPU available: {gpu_name}")
        print(f"✓ GPU count: {gpu_count}")
        print(f"✓ GPU memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved, {memory_total:.1f}GB total")
        
        # Test basic GPU operation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✓ GPU computation test passed")
            return True
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
            return False
    else:
        print("⚠️ GPU not available - will use CPU (much slower)")
        return False

def test_file_paths():
    """Test if example file paths exist (you need to modify these)."""
    print("\nTesting file paths...")
    
    # You need to update these paths for your system
    test_psf_path = "/mnt/NAS/Grants/24_AIOBIO/2501_data/calib/mask_1/whole_psf/processed/psf_stack_c7cae0af_20250630_151240.mat"  # Update this
    test_raw_path = "../raw_image/simulation/simulated_raw_sum_type_2_0629_2354.png"  # Update this
    
    print(f"Looking for PSF file: {test_psf_path}")
    print(f"Looking for raw image: {test_raw_path}")
    
    if os.path.exists(test_psf_path):
        print("✓ Test PSF file found")
        psf_exists = True
    else:
        print("⚠️ Test PSF file not found (you need to update the path)")
        psf_exists = False
    
    if os.path.exists(test_raw_path):
        print("✓ Test raw image found")
        raw_exists = True
    else:
        print("⚠️ Test raw image not found (you need to update the path)")
        raw_exists = False
    
    return psf_exists and raw_exists

def test_directory_creation():
    """Test directory creation and file writing permissions."""
    print("\nTesting directory creation...")
    
    test_dir = f"test_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✓ Directory created: {test_dir}")
        
        # Test file writing
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test file")
        print("✓ File writing test passed")
        
        # Cleanup
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory/file test failed: {e}")
        return False

def test_optimizer_creation():
    """Test creating optimizer with minimal configuration."""
    print("\nTesting optimizer creation...")
    
    try:
        from parameter_optimization_batch import ParameterOptimizer
        
        # Minimal test configuration
        base_config = {
            'path_ref': 0,
            'psf_file': 'dummy_path.npz',
            'img_file': 'dummy_path.jpg',
            'save_every': 0,
            'save_fig': False,
            'show_figs': False,
            'useGPU': torch.cuda.is_available(),
            'max_iter': 10,
        }
        
        opt_config = {
            'method': 'grid_search',
            'parameter_space': {
                'mu1': [0.5],
                'tau': [6e-4],
                'regularizer': ['3dtv'],
                'max_iter': [10],
            }
        }
        
        optimizer = ParameterOptimizer(base_config, opt_config)
        print("✓ Optimizer created successfully")
        
        # Test parameter combination generation
        combinations = optimizer.generate_parameter_combinations()
        print(f"✓ Generated {len(combinations)} parameter combinations")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        return False

def test_csv_functionality():
    """Test CSV saving functionality."""
    print("\nTesting CSV functionality...")
    
    try:
        import pandas as pd
        
        # Create dummy data
        test_data = {
            'experiment_id': ['exp_0001', 'exp_0002'],
            'param_mu1': [0.5, 0.75],
            'param_tau': [6e-4, 1e-3],
            'metric_composite_score': [100.5, 95.2],
            'success': [True, True]
        }
        
        df = pd.DataFrame(test_data)
        
        # Test CSV saving
        test_csv = f"test_results_{datetime.now().strftime('%H%M%S')}.csv"
        df.to_csv(test_csv, index=False)
        print(f"✓ CSV file created: {test_csv}")
        
        # Test CSV reading
        df_loaded = pd.read_csv(test_csv)
        print(f"✓ CSV file loaded: {len(df_loaded)} rows")
        
        # Cleanup
        os.remove(test_csv)
        print("✓ CSV test cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV functionality test failed: {e}")
        return False

def print_system_info():
    """Print system information."""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("Pandas: Not available")
    
    print()

def main():
    """Run all tests."""
    print("="*60)
    print("PARAMETER OPTIMIZATION SYSTEM TEST")
    print("="*60)
    
    print_system_info()
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Test", test_gpu_availability), 
        ("File Path Test", test_file_paths),
        ("Directory Test", test_directory_creation),
        ("Optimizer Test", test_optimizer_creation),
        ("CSV Test", test_csv_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name}...")
        print("-"*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready for optimization.")
        print("\nNext steps:")
        print("1. Update file paths in the test if needed")
        print("2. Run: python run_optimization.py")
        print("3. Or use: nohup python run_optimization.py > optimization.log 2>&1 &")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please fix issues before running optimization.")
        
        if not any(name == "Import Test" and result for name, result in results):
            print("\n❗ Critical: Import test failed. Check if all required modules are installed.")
        if not any(name == "GPU Test" and result for name, result in results):
            print("\n⚠️ Warning: GPU test failed. Optimization will be much slower on CPU.")

if __name__ == "__main__":
    main() 