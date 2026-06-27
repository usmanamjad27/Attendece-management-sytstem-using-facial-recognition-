"""
Simple camera test - Run this to verify your camera works
"""
import cv2
import time

print("Testing camera...")
print("=" * 50)

# Try to open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    print("Please check:")
    print("  - Camera is connected")
    print("  - No other app is using camera")
    print("  - Camera drivers are installed")
    exit(1)

print("Camera opened successfully!")
print("Camera window should appear now...")
print("Press 'Q' to close")

# Give camera time to initialize
time.sleep(0.5)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break
    
    frame = cv2.flip(frame, 1)
    
    # Add text
    cv2.putText(frame, "Camera Test - Press Q to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Camera Test", frame)
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nCamera test complete! Processed {frame_count} frames.")
print("If you saw the camera window, your camera is working!")


