import os
import csv
from pathlib import Path
from typing import Dict, Tuple, Set

import cv2
import numpy as np
import pandas as pd
from FaceDetection.face_detection import face
try:
    import tensorflow as tf
    load_model = tf.keras.models.load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        from keras.models import load_model

from embedding import emb
from MongoDB.retrieve_pymongo_data import database

import warnings

warnings.filterwarnings("ignore")

PEOPLE_DIR = Path("people")
MODEL_PATH = Path("Model") / "Face_recognition.MODEL"


def _load_enrollments() -> Dict[str, str]:
    csv_path = Path("Students_Enrollment.csv")
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        roll = str(row.get("Roll Number", "")).strip()
        name = str(row.get("Name", "")).strip()
        if roll:
            mapping[roll] = name or f"Student_{roll}"
    return mapping


def _parse_directory_name(folder_name: str) -> Tuple[str, str]:
    digits = "".join(ch for ch in folder_name if ch.isdigit())
    remainder = folder_name[len(digits) :].lstrip("_- ") if digits else folder_name
    return digits or "", remainder or folder_name


def _create_labels() -> Tuple[Dict[int, str], Dict[int, str]]:
    """Create mapping from index to student name and roll number"""
    enrollment_map = _load_enrollments()
    students: Dict[int, str] = {}  # index -> name
    roll_numbers: Dict[int, str] = {}  # index -> roll number
    valid_people = [
        folder
        for folder in sorted(os.listdir(PEOPLE_DIR))
        if (PEOPLE_DIR / folder).is_dir()
    ]
    for idx, folder in enumerate(valid_people):
        roll_digits, fallback_name = _parse_directory_name(folder)
        label_name = enrollment_map.get(roll_digits, fallback_name)
        students[idx] = label_name
        roll_numbers[idx] = roll_digits or folder
    return students, roll_numbers


def _update_csv_attendance(lecture: str, roll_number: str, name: str) -> None:
    """Update attendance CSV file by incrementing attendance by +1"""
    subject = "Urdu" if str(lecture) == "1" else "English"
    csv_path = Path(f"{subject}_attendance") / f"{subject}_Attendance.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing data
    rows = []
    header = ["Roll Number", "Name", "Attendance"]
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    # Find and update the student's attendance
    found = False
    for row in rows:
        if row.get("Roll Number", "").strip() == str(roll_number).strip():
            current_attendance = int(row.get("Attendance", "0") or "0")
            row["Attendance"] = str(current_attendance + 1)
            row["Name"] = name  # Update name in case it changed
            found = True
            break
    
    # If student not found, add new entry
    if not found:
        rows.append({
            "Roll Number": str(roll_number),
            "Name": name,
            "Attendance": "1"
        })
    
    # Write back to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def Recognition(subject: int, threshold: float = 0.80) -> None:
    students, roll_numbers = _create_labels()
    if not students:
        raise RuntimeError("No registered students were found. Please enroll first.")

    embedder = emb()
    fd = face()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(str(MODEL_PATH))

    data = database()
    lecture = str(subject)
    
    # Track recognized students in current frame to avoid duplicate increments
    recognized_this_frame: Set[int] = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera for recognition.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            detections, coordinates = fd.detectFace(frame)

            # Reset recognized set for each new frame
            recognized_this_frame.clear()

            if detections is None:
                cv2.imshow('Say Cheese and Press "Q" to Quit', frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            for idx, detected in enumerate(detections):
                bbox = coordinates[idx]
                processed = cv2.resize(detected, (160, 160)).astype("float32") / 255.0
                processed = np.expand_dims(processed, axis=0)
                feed = embedder.calculate(processed)
                feed = np.expand_dims(feed, axis=0)

                prediction = model.predict(feed, verbose=0)[0]
                result = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                label = "Unknown"
                rect_color = (0, 0, 255)
                attendance_updated = False

                if result in students and confidence >= threshold:
                    label = students[result]
                    rect_color = (0, 255, 0)
                    
                    # Update attendance immediately when person is recognized (only once per frame)
                    if result not in recognized_this_frame:
                        roll_number = roll_numbers.get(result, "")
                        # Update CSV attendance file
                        _update_csv_attendance(lecture, roll_number, label)
                        
                        # Update MongoDB if connected
                        if data.connected:
                            data.update(label, lecture)
                        
                        recognized_this_frame.add(result)
                        attendance_updated = True

                text = f"{label} ({confidence*100:.1f}%)" if label != "Unknown" else label

                if attendance_updated:
                    text = f"{label} - Attendance +1"

                cv2.putText(
                    frame,
                    text,
                    (bbox[0], max(25, bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    rect_color,
                    2,
                )

            cv2.imshow('Say Cheese and Press "Q" to Quit', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Export final CSV from MongoDB if connected
    if data.connected:
        data.export_csv(lecture)
