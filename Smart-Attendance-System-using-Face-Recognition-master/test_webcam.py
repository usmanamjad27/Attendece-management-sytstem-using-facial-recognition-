"""
Quick webcam test script
Run this to verify your webcam is working before capturing dataset
"""

import cv2
from pathlib import Path

def test_webcam():
    """Test if webcam is accessible and working"""
    print("Testing webcam...")
    print("=" * 50)
    
    # Try to find available camera
    camera_found = False
    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Webcam found at camera index: {camera_index}")
                print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                camera_found = True
                
                # Test face detection
                cascade_path = Path("FaceDetection") / "faces.xml"
                if cascade_path.exists():
                    face_cascade = cv2.CascadeClassifier(str(cascade_path))
                    if not face_cascade.empty():
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        print(f"  Face detection: Working ({len(faces)} face(s) detected in test frame)")
                    else:
                        print(f"  Face detection: Cascade file invalid")
                else:
                    print(f"  Face detection: Cascade file not found at {cascade_path}")
                
                cap.release()
                break
            cap.release()
    
    if not camera_found:
        print("✗ No webcam found!")
        print("\nTroubleshooting:")
        print("  • Check if webcam is connected")
        print("  • Make sure no other app is using the webcam")
        print("  • Try unplugging and replugging the webcam")
        print("  • Check device manager for webcam drivers")
        return False
    
    print("\n" + "=" * 50)
    print("Webcam test completed successfully!")
    print("You can now use 'Capture Dataset' to take pictures.")
    return True

if __name__ == "__main__":
    try:
        test_webcam()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        input("\nPress Enter to exit...")


