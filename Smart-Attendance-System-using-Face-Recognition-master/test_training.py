"""
Test script to diagnose model training issues
Run this to check if everything is ready for training
"""

from pathlib import Path
import sys

def test_training_setup():
    """Test if all requirements for training are met"""
    print("=" * 60)
    print("Testing Model Training Setup")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # 1. Check if people directory exists
    print("\n1. Checking 'people' directory...")
    people_dir = Path("people")
    if not people_dir.exists():
        issues.append("'people' directory does not exist")
        print("   ✗ 'people' directory not found")
    else:
        print("   ✓ 'people' directory exists")
        
        # Check for student folders
        student_dirs = [d for d in people_dir.iterdir() if d.is_dir()]
        if not student_dirs:
            issues.append("No student folders found in 'people' directory")
            print("   ✗ No student folders found")
        else:
            print(f"   ✓ Found {len(student_dirs)} student folder(s)")
            
            # Check for images
            total_images = 0
            for student_dir in student_dirs:
                images = list(student_dir.glob("*.jpg"))
                image_count = len(images)
                total_images += image_count
                if image_count == 0:
                    warnings.append(f"No images found in {student_dir.name}")
                    print(f"   ⚠ {student_dir.name}: No images")
                else:
                    print(f"   ✓ {student_dir.name}: {image_count} images")
            
            if total_images == 0:
                issues.append("No images found in any student folder")
            else:
                print(f"   ✓ Total images: {total_images}")
    
    # 2. Check for pre-trained model
    print("\n2. Checking pre-trained model...")
    pretrained_path = Path("PreTrained_model/facenet_keras.h5")
    if not pretrained_path.exists():
        issues.append(f"Pre-trained model not found: {pretrained_path}")
        print(f"   ✗ Pre-trained model not found at {pretrained_path}")
    else:
        file_size = pretrained_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   ✓ Pre-trained model found ({file_size:.1f} MB)")
    
    # 3. Check dependencies
    print("\n3. Checking dependencies...")
    
    try:
        import cv2
        print("   ✓ OpenCV (cv2) installed")
    except ImportError:
        issues.append("OpenCV not installed")
        print("   ✗ OpenCV (cv2) not installed")
    
    try:
        import numpy as np
        print(f"   ✓ NumPy installed (version: {np.__version__})")
    except ImportError:
        issues.append("NumPy not installed")
        print("   ✗ NumPy not installed")
    
    try:
        import keras
        print(f"   ✓ Keras installed (version: {keras.__version__})")
    except ImportError:
        issues.append("Keras not installed")
        print("   ✗ Keras not installed")
    
    try:
        import tensorflow as tf
        print(f"   ✓ TensorFlow installed (version: {tf.__version__})")
    except ImportError:
        issues.append("TensorFlow not installed")
        print("   ✗ TensorFlow not installed")
    
    # 4. Test imports
    print("\n4. Testing module imports...")
    
    try:
        from embedding import emb
        print("   ✓ embedding module imports successfully")
        
        # Try to load the embedding model
        try:
            embedding_model = emb()
            print("   ✓ Embedding model loads successfully")
        except Exception as e:
            issues.append(f"Failed to load embedding model: {e}")
            print(f"   ✗ Failed to load embedding model: {e}")
    except ImportError as e:
        issues.append(f"Cannot import embedding module: {e}")
        print(f"   ✗ Cannot import embedding module: {e}")
    
    try:
        from Model_architecture.modelArch import DenseArchs
        print("   ✓ Model architecture module imports successfully")
    except ImportError as e:
        issues.append(f"Cannot import model architecture: {e}")
        print(f"   ✗ Cannot import model architecture: {e}")
    
    try:
        from Model_train import Model_Training
        print("   ✓ Model training module imports successfully")
    except ImportError as e:
        issues.append(f"Cannot import training module: {e}")
        print(f"   ✗ Cannot import training module: {e}")
    
    # 5. Check Model directory
    print("\n5. Checking Model directory...")
    model_dir = Path("Model")
    if not model_dir.exists():
        print("   ⚠ Model directory does not exist (will be created)")
        warnings.append("Model directory will be created during training")
    else:
        print("   ✓ Model directory exists")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if issues:
        print("\n❌ ISSUES FOUND (must be fixed):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\nPlease fix these issues before training.")
    else:
        print("\n✓ No critical issues found!")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    if not issues:
        print("\n✓ Ready for training! You can now use the 'Model Training' button.")
    
    return len(issues) == 0

if __name__ == "__main__":
    try:
        success = test_training_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        input("\nPress Enter to exit...")


