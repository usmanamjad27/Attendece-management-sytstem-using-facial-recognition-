import os
from pathlib import Path
from typing import Tuple

import cv2
import pandas as pd
import pymongo
from pymongo import errors as mongo_errors

import warnings

warnings.filterwarnings("ignore")

DATASET_ROOT = Path("people")
ENROLLMENT_FILE = Path("Students_Enrollment.csv")


class EnrollmentError(Exception):
    """Raised when a student cannot be enrolled safely."""


def _sanitize_name(raw_name: str) -> str:
    cleaned = "".join(ch for ch in raw_name.title() if ch.isalnum())
    return cleaned or "Student"


def _ensure_enrollment_file() -> pd.DataFrame:
    if not ENROLLMENT_FILE.exists():
        ENROLLMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["Name", "Roll Number"]).to_csv(ENROLLMENT_FILE, index=False)
    return pd.read_csv(ENROLLMENT_FILE)


def _build_dataset_path(name: str, roll_no: str) -> Path:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_name(name)
    candidate = DATASET_ROOT / f"{roll_no}_{safe_name}"
    legacy = DATASET_ROOT / f"{roll_no}{safe_name}"
    if candidate.exists():
        raise EnrollmentError(f"Dataset for roll {roll_no} already exists.")
    if legacy.exists():
        raise EnrollmentError(f"Legacy dataset for roll {roll_no} already exists.")
    return candidate


def _register_student(name: str, roll_no: str) -> None:
    df = _ensure_enrollment_file()
    if df["Roll Number"].astype(str).eq(roll_no).any():
        raise EnrollmentError(f"Roll number {roll_no} is already enrolled.")
    record = pd.DataFrame([{"Name": name, "Roll Number": roll_no}])
    df = pd.concat([df, record], ignore_index=True)
    df.to_csv(ENROLLMENT_FILE, index=False)


def _push_to_mongo(name: str, roll_no: str) -> bool:
    try:
        client = pymongo.MongoClient(
            "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
        )
        client.server_info()
    except mongo_errors.PyMongoError as exc:
        print(f"[warning] MongoDB unavailable, skipped DB sync: {exc}")
        return False

    mydb = client["students"]
    payload = {"Name": name, "Roll_number": roll_no, "Attendance": 0}
    try:
        mydb["Urdu"].update_one({"Roll_number": roll_no}, {"$setOnInsert": payload}, upsert=True)
        mydb["English"].update_one({"Roll_number": roll_no}, {"$setOnInsert": payload}, upsert=True)
    except mongo_errors.PyMongoError as exc:
        print(f"[warning] Unable to sync MongoDB: {exc}")
        return False
    print("Student synced with MongoDB (Urdu & English subjects).")
    return True


def _find_available_camera() -> int:
    """Try to find an available camera by testing indices 0-4"""
    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return camera_index
    return None


def _capture_dataset(target_dir: Path, target_images: int = 40) -> int:
    """Capture dataset using webcam with face detection"""
    import time
    
    cascade_path = Path("FaceDetection") / "faces.xml"
    if not cascade_path.exists():
        raise RuntimeError(f"Face detection cascade file not found: {cascade_path}")
    
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    
    if face_cascade.empty():
        raise RuntimeError("Failed to load face detection cascade. Please check faces.xml file.")
    
    # Try to find and open webcam
    print("Searching for available camera...")
    camera_index = _find_available_camera()
    if camera_index is None:
        raise RuntimeError(
            "No webcam found! Please:\n"
            "• Check if your webcam is connected\n"
            "• Make sure no other application is using the webcam\n"
            "• Try unplugging and replugging the webcam"
        )
    
    print(f"Found camera at index: {camera_index}")
    print("Opening webcam...")
    cap = cv2.VideoCapture(camera_index)
    
    # Give camera time to initialize
    time.sleep(0.5)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Unable to access webcam. Please check:\n"
            "• Webcam is connected\n"
            "• Webcam drivers are installed\n"
            "• No other application is using the webcam"
        )

    # Test if we can read frames
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise RuntimeError("Camera opened but cannot read frames. Please check camera permissions.")

    captured = 0
    target_dir.mkdir(parents=True, exist_ok=False)
    
    print(f"\n{'='*60}")
    print(f"Webcam opened successfully!")
    print(f"Target: {target_images} images")
    print(f"Save location: {target_dir}")
    print(f"{'='*60}")
    print("Instructions:")
    print("  • Look directly at the camera")
    print("  • Keep your face centered in the frame")
    print("  • Ensure good lighting")
    print("  • Press 'Q' to stop early")
    print(f"{'='*60}\n")

    # Create window and move it to front
    window_name = "Capture Dataset - Training Images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Try to bring window to front (Windows specific)
    try:
        import platform
        if platform.system() == 'Windows':
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    except:
        pass  # If it fails, continue anyway
    
    try:
        frame_count = 0
        no_face_count = 0
        last_capture_frame = 0
        
        print("Camera window should be visible now. Looking for faces...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                no_face_count += 1
                if no_face_count > 30:
                    print("ERROR: Cannot read frames from camera. Stopping...")
                    break
                continue
            
            no_face_count = 0  # Reset counter on successful read
            
            frame = cv2.flip(frame, 1)  # Mirror the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

            # Draw rectangle and capture when face is detected
            face_detected = False
            for (x, y, w, h) in faces:
                face_detected = True
                # Draw green rectangle around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Capture image (with delay to avoid capturing too fast)
                frame_count += 1
                # Capture every 5 frames when face is detected (slower to ensure quality)
                if frame_count - last_capture_frame >= 5:
                    cropped = frame[y : y + h, x : x + w]
                    if cropped.size > 0 and w > 50 and h > 50:  # Ensure valid image with minimum size
                        captured += 1
                        image_path = target_dir / f"{captured}.jpg"
                        success = cv2.imwrite(str(image_path), cropped)
                        if success:
                            print(f"[{captured}/{target_images}] Image saved: {image_path.name}")
                            last_capture_frame = frame_count
                            # Flash effect when capturing
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
                        else:
                            print(f"WARNING: Failed to save image {captured}")
                            captured -= 1  # Don't count failed saves
            
            # Display status information on frame
            status_text = f"Captured: {captured}/{target_images}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show face detection status
            if face_detected:
                detection_text = "Face Detected - Keep Still!"
                detection_color = (0, 255, 0)
            else:
                detection_text = "No Face Detected - Move closer to camera"
                detection_color = (0, 0, 255)
            
            cv2.putText(frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
            
            # Instructions
            if captured < target_images:
                instruction_text = "Look at the camera | Press 'Q' to stop"
                cv2.putText(frame, instruction_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                complete_text = "Capture Complete! Press 'Q' to close"
                cv2.putText(frame, complete_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow(window_name, frame)
            
            # Check for 'Q' key or if target reached
            key = cv2.waitKey(1) & 0xFF
            if captured >= target_images:
                print(f"\nTarget reached! Captured {captured} images.")
                time.sleep(1)  # Show completion message briefly
                break
            if key in (ord("q"), ord("Q")):
                print(f"\nCapture stopped by user. Captured {captured} images.")
                break
                
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        import traceback
        print(f"\nERROR during capture: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error during capture: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nWebcam closed. Total images captured: {captured}")

    if captured == 0:
        raise RuntimeError(
            "No images were captured. Please ensure:\n"
            "• Your face is visible in the camera\n"
            "• There is adequate lighting\n"
            "• The webcam is working properly\n"
            "• You are looking directly at the camera"
        )

    return captured


def Generate_Data(name: str, roll_no: str) -> Tuple[str, int]:
    """Generate dataset for a student and register them system-wide."""
    name = (name or "").strip()
    roll_no = str(roll_no or "").strip()

    if not name or not roll_no:
        raise ValueError("Name and roll number are required.")
    if not roll_no.isdigit():
        raise ValueError("Roll number must be numeric.")

    target_path = _build_dataset_path(name, roll_no)
    _register_student(name, roll_no)
    _push_to_mongo(name, roll_no)
    captured = _capture_dataset(target_path)
    print(f"Captured {captured} images for {name}.")
    return str(target_path), captured
