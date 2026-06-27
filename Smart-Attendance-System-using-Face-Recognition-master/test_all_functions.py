"""
Comprehensive test script to check all functions in the Smart Attendance System
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SMART ATTENDANCE SYSTEM - FUNCTION STATUS CHECK")
print("=" * 70)

issues = []
working = []
warnings_list = []

# 1. Check Python and basic imports
print("\n1. BASIC ENVIRONMENT")
print("-" * 70)
try:
    print(f"   [OK] Python version: {sys.version.split()[0]}")
    working.append("Python environment")
except Exception as e:
    issues.append(f"Python environment: {e}")

# 2. Check required directories
print("\n2. DIRECTORY STRUCTURE")
print("-" * 70)
required_dirs = [
    "people",
    "Model",
    "PreTrained_model",
    "FaceDetection",
    "Model_architecture",
    "MongoDB",
    "Urdu_attendance",
    "English_attendance"
]

for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   [OK] {dir_name}/ exists")
        working.append(f"Directory: {dir_name}")
    else:
        print(f"   [X] {dir_name}/ missing")
        issues.append(f"Missing directory: {dir_name}")

# 3. Check required files
print("\n3. REQUIRED FILES")
print("-" * 70)
required_files = [
    "FaceDetection/faces.xml",
    "PreTrained_model/facenet_keras.h5",
    "Model_architecture/modelArch.py",
    "embedding.py",
    "Generate_Dataset.py",
    "Model_train.py",
    "Recognizer.py",
    "UI.py",
    "main.py"
]

for file_name in required_files:
    file_path = Path(file_name)
    if file_path.exists():
        print(f"   [OK] {file_name} exists")
        working.append(f"File: {file_name}")
    else:
        print(f"   [X] {file_name} missing")
        issues.append(f"Missing file: {file_name}")

# 4. Check dependencies
print("\n4. PYTHON DEPENDENCIES")
print("-" * 70)
dependencies = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "tkinter": "tkinter (built-in)",
    "pymongo": "pymongo"
}

for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"   [OK] {package} installed")
        working.append(f"Package: {package}")
    except ImportError:
        print(f"   [X] {package} NOT installed")
        issues.append(f"Missing package: {package}")

# 5. Check TensorFlow and Keras
print("\n5. MACHINE LEARNING DEPENDENCIES")
print("-" * 70)
try:
    import tensorflow as tf
    print(f"   [OK] TensorFlow {tf.__version__} installed")
    working.append("TensorFlow")
    
    # Test tf.keras
    try:
        models = tf.keras.models
        optimizers = tf.keras.optimizers
        utils = tf.keras.utils
        print(f"   [OK] tf.keras is accessible")
        working.append("tf.keras")
    except Exception as e:
        print(f"   [X] tf.keras not working: {e}")
        issues.append(f"tf.keras error: {e}")
        
except ImportError:
    print(f"   [X] TensorFlow NOT installed")
    issues.append("TensorFlow not installed")

# 6. Test module imports
print("\n6. MODULE IMPORTS")
print("-" * 70)
modules_to_test = [
    ("Generate_Dataset", "Generate_Dataset"),
    ("Model_train", "Model_train"),
    ("Recognizer", "Recognizer"),
    ("embedding", "embedding"),
    ("Model_architecture.modelArch", "Model Architecture"),
    ("FaceDetection.face_detection", "Face Detection"),
    ("MongoDB.retrieve_pymongo_data", "MongoDB"),
    ("UI", "UI"),
    ("main", "Main Dashboard")
]

for module_name, display_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"   [OK] {display_name} imports successfully")
        working.append(f"Module: {display_name}")
    except ImportError as e:
        print(f"   [X] {display_name} import failed: {e}")
        issues.append(f"{display_name} import error: {e}")
    except Exception as e:
        print(f"   [WARN] {display_name} import warning: {e}")
        warnings_list.append(f"{display_name}: {e}")

# 7. Test specific functions
print("\n7. FUNCTION AVAILABILITY")
print("-" * 70)

# Test Generate_Dataset functions
try:
    from Generate_Dataset import Generate_Data, EnrollmentError
    print("   [OK] Generate_Data function available")
    working.append("Generate_Data function")
except Exception as e:
    print(f"   [X] Generate_Data function: {e}")
    issues.append(f"Generate_Data: {e}")

# Test Model Training
try:
    from Model_train import Model_Training
    print("   [OK] Model_Training function available")
    working.append("Model_Training function")
except Exception as e:
    print(f"   [X] Model_Training function: {e}")
    issues.append(f"Model_Training: {e}")

# Test Recognition
try:
    from Recognizer import Recognition
    print("   [OK] Recognition function available")
    working.append("Recognition function")
except Exception as e:
    print(f"   [X] Recognition function: {e}")
    issues.append(f"Recognition: {e}")

# Test Embedding
try:
    from embedding import emb
    print("   [OK] emb class available")
    working.append("emb class")
except Exception as e:
    print(f"   [X] emb class: {e}")
    issues.append(f"emb class: {e}")

# 8. Check data files
print("\n8. DATA FILES")
print("-" * 70)
data_files = [
    "Students_Enrollment.csv",
    "Urdu_attendance/Urdu_Attendance.csv",
    "English_attendance/English_Attendance.csv"
]

for file_name in data_files:
    file_path = Path(file_name)
    if file_path.exists():
        print(f"   [OK] {file_name} exists")
        working.append(f"Data file: {file_name}")
    else:
        print(f"   [WARN] {file_name} doesn't exist (will be created when needed)")
        warnings_list.append(f"Data file missing: {file_name}")

# 9. Check if students are enrolled
print("\n9. STUDENT ENROLLMENT STATUS")
print("-" * 70)
enrollment_file = Path("Students_Enrollment.csv")
if enrollment_file.exists():
    try:
        import csv
        with open(enrollment_file, 'r') as f:
            reader = csv.DictReader(f)
            students = list(reader)
            if students:
                print(f"   [OK] {len(students)} student(s) enrolled")
                working.append(f"Students enrolled: {len(students)}")
            else:
                print(f"   [WARN] Enrollment file exists but no students found")
                warnings_list.append("No students enrolled")
    except Exception as e:
        print(f"   [X] Error reading enrollment file: {e}")
        issues.append(f"Enrollment file error: {e}")
else:
    print(f"   [WARN] No enrollment file found (normal if no students enrolled yet)")
    warnings_list.append("Enrollment file not created yet")

# 10. Check if images exist
print("\n10. TRAINING IMAGES")
print("-" * 70)
people_dir = Path("people")
if people_dir.exists():
    student_dirs = [d for d in people_dir.iterdir() if d.is_dir()]
    if student_dirs:
        total_images = 0
        for student_dir in student_dirs:
            images = list(student_dir.glob("*.jpg"))
            total_images += len(images)
        print(f"   [OK] {len(student_dirs)} student folder(s) with {total_images} total images")
        working.append(f"Training images: {total_images} images")
    else:
        print(f"   [WARN] No student image folders found")
        warnings_list.append("No training images captured yet")
else:
    print(f"   ⚠ people/ directory doesn't exist")
    warnings_list.append("people/ directory not created")

# 11. Check if model is trained
print("\n11. TRAINED MODEL")
print("-" * 70)
model_path = Path("Model/Face_recognition.MODEL")
if model_path.exists():
    print(f"   [OK] Trained model exists")
    working.append("Trained model available")
else:
    print(f"   ⚠ No trained model found (need to train model first)")
    warnings_list.append("Model not trained yet")

# 12. Test Dashboard initialization
print("\n12. DASHBOARD INITIALIZATION")
print("-" * 70)
try:
    import tkinter as tk
    from main import Dashboard
    
    root = tk.Tk()
    root.withdraw()  # Hide the window
    app = Dashboard(root)
    
    print("   [OK] Dashboard initializes successfully")
    print(f"   [OK] Module status:")
    for key, val in app.modules_status.items():
        status = "[OK] Available" if val else "[X] Unavailable"
        print(f"      • {key}: {status}")
        if val:
            working.append(f"Dashboard module: {key}")
        else:
            issues.append(f"Dashboard module unavailable: {key}")
    
    root.destroy()
    working.append("Dashboard initialization")
except Exception as e:
    print(f"   [X] Dashboard initialization failed: {e}")
    issues.append(f"Dashboard error: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n[OK] WORKING: {len(working)} items")
for item in working[:10]:  # Show first 10
    print(f"   • {item}")
if len(working) > 10:
    print(f"   ... and {len(working) - 10} more")

if warnings_list:
    print(f"\n⚠ WARNINGS: {len(warnings_list)} items")
    for item in warnings_list[:5]:  # Show first 5
        print(f"   • {item}")
    if len(warnings_list) > 5:
        print(f"   ... and {len(warnings_list) - 5} more")

if issues:
    print(f"\n[X] ISSUES: {len(issues)} items")
    for item in issues:
        print(f"   • {item}")
    print("\n[WARN] ACTION REQUIRED: Please fix the issues above")
else:
    print("\n[OK] NO CRITICAL ISSUES FOUND!")
    print("   All main functions should be working properly.")

print("\n" + "=" * 70)


