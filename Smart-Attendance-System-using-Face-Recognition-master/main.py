"""
Smart Attendance System Using Face Recognition
Main Dashboard - Entry Point

This is the main dashboard that connects all modules:
- Student Enrollment
- Model Training  
- Take Attendance
- View Attendance
- Settings
"""

import sys
import os
import csv
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox as mess
from tkinter import ttk
from PIL import Image, ImageTk

# Suppress TensorFlow warnings and set environment
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress TensorFlow info/warning messages

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Colors for the dashboard
COLORS = {
    "bg": "#0f172a",
    "card": "#1e293b",
    "accent": "#ec4899",
    "accent_dark": "#be185d",
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "text": "#f8fafc",
}


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System - Dashboard")
        self.root.configure(bg=COLORS["bg"])
        self.root.state("zoomed")
        
        # Check dependencies
        
        self._check_dependencies()
        
        # Build UI
        self._build_dashboard()
        
    def _check_dependencies(self):
        """Check which modules are available"""
        self.modules_status = {
            "enrollment": True,  # Generate_Dataset doesn't need keras
            "training": False,
            "recognition": False,
            "ui": True,  # UI can run without keras
        }
        
        # Check for tensorflow and keras
        try:
            # Suppress warnings during import
            import warnings
            warnings.filterwarnings('ignore')
            
            import tensorflow as tf
            # Test if tf.keras is available by trying to access it
            # tf.keras uses lazy loading, so we need to actually access it
            try:
                # Try to access tf.keras.models (this will trigger lazy loading)
                models_module = tf.keras.models
                optimizers_module = tf.keras.optimizers
                utils_module = tf.keras.utils
                # If we get here, tf.keras is working
                self.modules_status["training"] = True
                self.modules_status["recognition"] = True
                print(f"[INFO] TensorFlow {tf.__version__} with tf.keras is available")
            except (AttributeError, Exception) as ke:
                # tf.keras didn't work, try standalone keras
                raise ke
        except (ImportError, AttributeError, Exception):
            # If TensorFlow or tf.keras doesn't work, try standalone keras as fallback
            try:
                import warnings
                warnings.filterwarnings('ignore')
                import keras
                # Test if keras works
                _ = keras.models
                _ = keras.optimizers
                _ = keras.utils
                self.modules_status["training"] = True
                self.modules_status["recognition"] = True
                print(f"[INFO] Standalone keras {keras.__version__} is available")
            except (ImportError, AttributeError, Exception):
                # Neither works
                print(f"[WARNING] TensorFlow/Keras not properly configured")
                pass
    
    def _build_dashboard(self):
        """Build the main dashboard interface"""
        # Header
        header = tk.Frame(self.root, bg=COLORS["bg"], height=120)
        header.pack(fill="x", padx=30, pady=(30, 20))
        
        # Title
        title_frame = tk.Frame(header, bg=COLORS["bg"])
        title_frame.pack(expand=True)
        
        tk.Label(
            title_frame,
            text="Smart Attendance System",
            font=("Helvetica", 42, "bold"),
            fg=COLORS["text"],
            bg=COLORS["bg"],
        ).pack()
        
        tk.Label(
            title_frame,
            text="Face Recognition Based Attendance Management",
            font=("Helvetica", 16),
            fg="#94a3b8",
            bg=COLORS["bg"],
        ).pack(pady=(10, 0))
        
        # Main content area
        content = tk.Frame(self.root, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # Create button grid
        self._create_action_buttons(content)
        
        # Status bar
        self._create_status_bar()
    
    def _create_action_buttons(self, parent):
        """Create action buttons for different modules"""
        buttons_frame = tk.Frame(parent, bg=COLORS["bg"])
        buttons_frame.pack(expand=True)
        
        # Button configurations
        actions = [
            {
                "text": "📝 Student Enrollment",
                "command": self._open_enrollment,
                "description": "Enroll new students and capture their face images",
                "available": self.modules_status["enrollment"],
                "row": 0,
                "col": 0,
            },
            {
                "text": "🎓 Model Training",
                "command": self._open_training,
                "description": "Train the face recognition model",
                "available": self.modules_status["training"],
                "row": 0,
                "col": 1,
            },
            {
                "text": "📸 Take Attendance",
                "command": self._open_attendance_ui,
                "description": "Open full attendance management interface",
                "available": self.modules_status["ui"],
                "row": 0,
                "col": 2,
            },
            {
                "text": "👁️ View Attendance",
                "command": self._open_view_attendance,
                "description": "View attendance records for Urdu and English",
                "available": True,
                "row": 1,
                "col": 0,
            },
            {
                "text": "🔍 Test Recognition",
                "command": self._open_test_recognition,
                "description": "Test face recognition (requires trained model)",
                "available": self.modules_status["recognition"],
                "row": 1,
                "col": 1,
            },
            {
                "text": "⚙️ Settings",
                "command": self._open_settings,
                "description": "Configure system settings",
                "available": True,
                "row": 1,
                "col": 2,
            },
            {
                "text": "🗑️ Delete Student Data",
                "command": self._open_delete_student,
                "description": "Remove student records, attendance, and images",
                "available": True,
                "row": 2,
                "col": 0,
            },
        ]
        
        # Configure grid
        for i in range(3):
            buttons_frame.columnconfigure(i, weight=1, uniform="equal")
        for i in range(3):
            buttons_frame.rowconfigure(i, weight=1, uniform="equal")
        
        # Create buttons
        for action in actions:
            self._create_action_button(buttons_frame, action)
    
    def _create_action_button(self, parent, action):
        """Create a single action button"""
        btn_frame = tk.Frame(
            parent,
            bg=COLORS["card"],
            relief="flat",
            bd=0,
        )
        btn_frame.grid(
            row=action["row"],
            column=action["col"],
            padx=15,
            pady=15,
            sticky="nsew",
        )
        btn_frame.grid_propagate(False)
        
        # Button
        btn = tk.Button(
            btn_frame,
            text=action["text"],
            font=("Helvetica", 18, "bold"),
            bg=COLORS["primary"] if action["available"] else "#475569",
            fg="white",
            activebackground=COLORS["primary_dark"] if action["available"] else "#64748b",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=30,
            pady=20,
            cursor="hand2" if action["available"] else "arrow",
            command=action["command"] if action["available"] else lambda: self._show_unavailable(action["text"]),
        )
        btn.pack(fill="both", expand=True)
        
        # Description
        desc = tk.Label(
            btn_frame,
            text=action["description"],
            font=("Helvetica", 11),
            fg="#cbd5e1",
            bg=COLORS["card"],
            wraplength=250,
        )
        desc.pack(pady=(0, 15))
        
        # Status indicator
        status_text = "✓ Available" if action["available"] else "✗ Requires Setup"
        status_color = COLORS["success"] if action["available"] else COLORS["warning"]
        status = tk.Label(
            btn_frame,
            text=status_text,
            font=("Helvetica", 10, "bold"),
            fg=status_color,
            bg=COLORS["card"],
        )
        status.pack()
    
    def _create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = tk.Frame(self.root, bg=COLORS["accent"], height=50)
        status_frame.pack(fill="x", side="bottom")
        
        self.status_label = tk.Label(
            status_frame,
            text="Dashboard Ready | All systems operational",
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg=COLORS["accent"],
            padx=20,
            pady=12,
        )
        self.status_label.pack(side="left")
        
        quit_btn = tk.Button(
            status_frame,
            text="Exit",
            command=self.root.quit,
            bg=COLORS["danger"],
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            padx=25,
            pady=12,
            cursor="hand2",
        )
        quit_btn.pack(side="right", padx=20)
    
    def _open_enrollment(self):
        """Open student enrollment module"""
        try:
            from Generate_Dataset import Generate_Data, EnrollmentError
            
            # Create enrollment window
            enroll_window = tk.Toplevel(self.root)
            enroll_window.title("Student Enrollment")
            enroll_window.configure(bg=COLORS["bg"])
            enroll_window.geometry("500x400")
            
            frame = tk.Frame(enroll_window, bg=COLORS["card"], padx=30, pady=30)
            frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            tk.Label(
                frame,
                text="Student Enrollment",
                font=("Helvetica", 20, "bold"),
                fg=COLORS["text"],
                bg=COLORS["card"],
            ).pack(pady=(0, 20))
            
            # Roll Number
            tk.Label(frame, text="Roll Number:", font=("Helvetica", 12), fg=COLORS["text"], bg=COLORS["card"]).pack(anchor="w", pady=5)
            roll_entry = tk.Entry(frame, font=("Helvetica", 12), width=30)
            roll_entry.pack(pady=(0, 15), fill="x")
            
            # Name
            tk.Label(frame, text="Student Name:", font=("Helvetica", 12), fg=COLORS["text"], bg=COLORS["card"]).pack(anchor="w", pady=5)
            name_entry = tk.Entry(frame, font=("Helvetica", 12), width=30)
            name_entry.pack(pady=(0, 20), fill="x")
            
            def submit_registration():
                """Register student without capturing images"""
                roll = roll_entry.get().strip()
                name = name_entry.get().strip()
                if not roll or not name:
                    mess.showerror("Error", "Please enter both roll number and name")
                    return
                if not roll.isdigit():
                    mess.showerror("Invalid Roll Number", "Roll number must be numeric.")
                    return
                
                try:
                    # Update Students_Enrollment.csv
                    enrollment_path = Path("Students_Enrollment.csv")
                    header = ["Name", "Roll Number"]
                    rows = []
                    if enrollment_path.exists():
                        with open(enrollment_path, newline="") as fh:
                            rows = list(csv.DictReader(fh))
                    
                    exists = False
                    for row in rows:
                        if row.get("Roll Number", "") == roll:
                            row["Name"] = name  # Update name if exists
                            exists = True
                            break
                    if not exists:
                        rows.append({"Name": name, "Roll Number": roll})
                    
                    with open(enrollment_path, "w", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=header)
                        writer.writeheader()
                        writer.writerows(rows)
                    
                    # Update attendance files for both subjects with 0 attendance
                    for subject in ["Urdu", "English"]:
                        attendance_path = Path(f"{subject}_attendance") / f"{subject}_Attendance.csv"
                        attendance_path.parent.mkdir(parents=True, exist_ok=True)
                        att_header = ["Roll Number", "Name", "Attendance"]
                        att_rows = []
                        if attendance_path.exists():
                            with open(attendance_path, newline="") as fh:
                                att_rows = list(csv.DictReader(fh))
                        
                        att_exists = False
                        for row in att_rows:
                            if row.get("Roll Number", "") == roll:
                                row["Name"] = name
                                att_exists = True
                                break
                        if not att_exists:
                            att_rows.append({"Roll Number": roll, "Name": name, "Attendance": "0"})
                        
                        with open(attendance_path, "w", newline="") as fh:
                            writer = csv.DictWriter(fh, fieldnames=att_header)
                            writer.writeheader()
                            writer.writerows(att_rows)
                    
                    mess.showinfo("Success", f"Student {name} (Roll: {roll}) registered successfully!")
                    self._set_status(f"Student {name} registered successfully")
                    # Clear fields
                    roll_entry.delete(0, tk.END)
                    name_entry.delete(0, tk.END)
                except Exception as e:
                    mess.showerror("Error", f"Registration failed: {e}")
                    self._set_status(f"Registration error: {e}")
            
            def start_capture():
                roll = roll_entry.get().strip()
                name = name_entry.get().strip()
                if not roll or not name:
                    mess.showerror("Error", "Please enter both roll number and name")
                    return
                if not roll.isdigit():
                    mess.showerror("Invalid Roll Number", "Roll number must be numeric.")
                    return
                
                # Show instructions before opening webcam
                response = mess.askyesno(
                    "Start Capture",
                    f"Ready to capture training images for {name} (Roll: {roll}).\n\n"
                    "Instructions:\n"
                    "• Webcam will open in a new window\n"
                    "• Look directly at the camera\n"
                    "• Keep your face in the frame\n"
                    "• System will capture 40 images automatically\n"
                    "• Press 'Q' to stop early if needed\n\n"
                    "Continue?"
                )
                
                if not response:
                    return
                
                try:
                    enroll_window.destroy()
                    self._set_status(f"Opening webcam to capture images for {name}...")
                    # Give a moment for window to close
                    self.root.update()
                    
                    # Run capture in a way that ensures window appears
                    import threading
                    def run_capture():
                        try:
                            result = Generate_Data(name, roll)
                            self.root.after(0, lambda: mess.showinfo(
                                "Success", 
                                f"Successfully captured {result[1]} images for {name}!\n\n"
                                f"Images saved to: {result[0]}\n\n"
                                "You can now train the model using the 'Model Training' button."
                            ))
                            self.root.after(0, lambda: self._set_status(f"Enrollment completed: {result[1]} images captured for {name}"))
                        except Exception as e:
                            import traceback
                            error_msg = str(e)
                            error_details = traceback.format_exc()
                            self.root.after(0, lambda: mess.showerror(
                                "Capture Error", 
                                f"Failed to capture images:\n\n{error_msg}\n\n"
                                "Please check:\n"
                                "• Webcam is connected and working\n"
                                "• No other application is using the webcam\n"
                                "• Camera permissions are granted\n"
                                "• Sufficient lighting\n"
                                "• Your face is visible to the camera"
                            ))
                            self.root.after(0, lambda: self._set_status(f"Enrollment error: {error_msg}"))
                            print(f"Capture error details:\n{error_details}")
                    
                    # Run in separate thread so UI doesn't freeze
                    threading.Thread(target=run_capture, daemon=True).start()
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    mess.showerror("Error", f"Failed to start capture: {str(e)}\n\nDetails:\n{error_details}")
                    self._set_status(f"Enrollment error: {e}")
            
            # Button frame for two buttons
            button_frame = tk.Frame(frame, bg=COLORS["card"])
            button_frame.pack(pady=10, fill="x")
            
            # Submit Registration button (register without capturing)
            tk.Button(
                button_frame,
                text="Submit Registration",
                command=submit_registration,
                bg=COLORS["primary"],
                fg="white",
                font=("Helvetica", 12, "bold"),
                padx=20,
                pady=10,
                cursor="hand2",
            ).pack(side="left", expand=True, fill="x", padx=(0, 10))
            
            # Start Capture button (register and capture images)
            tk.Button(
                button_frame,
                text="Start Capture",
                command=start_capture,
                bg=COLORS["accent"],
                fg="white",
                font=("Helvetica", 12, "bold"),
                padx=20,
                pady=10,
                cursor="hand2",
            ).pack(side="left", expand=True, fill="x")
            
        except ImportError as e:
            mess.showerror("Error", f"Cannot import enrollment module: {e}")
    
    def _get_student_by_roll(self, roll_number: str) -> tuple:
        """Get student name by roll number from enrollment CSV"""
        enrollment_path = Path("Students_Enrollment.csv")
        if not enrollment_path.exists():
            return None, None
        
        with open(enrollment_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("Roll Number", "").strip() == str(roll_number).strip():
                    return row.get("Name", "").strip(), str(roll_number).strip()
        return None, None
    
    def _open_training(self):
        """Open model training module with roll number options"""
        if not self.modules_status["training"]:
            mess.showerror(
                "Dependencies Missing",
                "Model training requires keras and tensorflow.\n\n"
                "Install them using:\npip install keras tensorflow"
            )
            return
        
        try:
            from Model_train import Model_Training
            from Generate_Dataset import Generate_Data, EnrollmentError
            
            # Create training window
            train_window = tk.Toplevel(self.root)
            train_window.title("Model Training")
            train_window.configure(bg=COLORS["bg"])
            train_window.geometry("550x500")
            
            frame = tk.Frame(train_window, bg=COLORS["card"], padx=30, pady=30)
            frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            tk.Label(
                frame,
                text="Model Training",
                font=("Helvetica", 20, "bold"),
                fg=COLORS["text"],
                bg=COLORS["card"],
            ).pack(pady=(0, 20))
            
            # Roll Number section
            roll_frame = tk.Frame(frame, bg=COLORS["card"])
            roll_frame.pack(fill="x", pady=(0, 20))
            
            tk.Label(
                roll_frame,
                text="Roll Number:",
                font=("Helvetica", 12, "bold"),
                fg=COLORS["text"],
                bg=COLORS["card"],
            ).pack(anchor="w", pady=5)
            
            roll_entry = tk.Entry(roll_frame, font=("Helvetica", 12), width=30)
            roll_entry.pack(pady=(0, 10), fill="x")
            
            # Info label
            info_label = tk.Label(
                roll_frame,
                text="Enter roll number to capture images or train model for specific student",
                font=("Helvetica", 9),
                fg="#94a3b8",
                bg=COLORS["card"],
                wraplength=450,
                justify="left",
            )
            info_label.pack(anchor="w", pady=(0, 10))
            
            # Buttons frame
            buttons_frame = tk.Frame(frame, bg=COLORS["card"])
            buttons_frame.pack(fill="x", pady=10)
            
            def capture_for_roll():
                """Capture images for specific roll number"""
                roll = roll_entry.get().strip()
                if not roll:
                    mess.showerror("Error", "Please enter a roll number")
                    return
                if not roll.isdigit():
                    mess.showerror("Invalid Roll Number", "Roll number must be numeric.")
                    return
                
                # Find student by roll number
                name, roll_found = self._get_student_by_roll(roll)
                if not name:
                    mess.showerror(
                        "Student Not Found",
                        f"Student with Roll Number {roll} is not registered.\n\n"
                        "Please register the student first using 'Student Enrollment'."
                    )
                    return
                
                # Check if images already exist
                people_dir = Path("people")
                image_folder = None
                if people_dir.exists():
                    for folder in people_dir.iterdir():
                        if folder.is_dir() and (folder.name.startswith(roll + "_") or folder.name.startswith(roll)):
                            image_folder = folder
                            break
                
                if image_folder and list(image_folder.glob("*.jpg")):
                    response = mess.askyesno(
                        "Images Already Exist",
                        f"Images already exist for {name} (Roll: {roll}).\n\n"
                        "Do you want to capture new images?\n"
                        "(This will add to existing images)"
                    )
                    if not response:
                        return
                
                # Show instructions
                response = mess.askyesno(
                    "Start Capture",
                    f"Ready to capture training images for:\n\n"
                    f"Name: {name}\n"
                    f"Roll Number: {roll}\n\n"
                    "Instructions:\n"
                    "• Webcam will open in a new window\n"
                    "• Look directly at the camera\n"
                    "• Keep your face in the frame\n"
                    "• System will capture 40 images automatically\n"
                    "• Press 'Q' to stop early if needed\n\n"
                    "Continue?"
                )
                
                if not response:
                    return
                
                try:
                    train_window.destroy()
                    self._set_status(f"Capturing images for {name} (Roll: {roll})...")
                    # Give a moment for window to close
                    self.root.update()
                    
                    # Run capture in separate thread
                    import threading
                    def run_capture():
                        try:
                            result = Generate_Data(name, roll)
                            self.root.after(0, lambda: mess.showinfo(
                                "Success",
                                f"Successfully captured {result[1]} images for {name}!\n\n"
                                f"Images saved to: {result[0]}\n\n"
                                "You can now train the model."
                            ))
                            self.root.after(0, lambda: self._set_status(f"Capture completed: {result[1]} images for {name}"))
                        except EnrollmentError as e:
                            self.root.after(0, lambda: mess.showerror("Enrollment Error", str(e)))
                            self.root.after(0, lambda: self._set_status(f"Capture error: {e}"))
                        except Exception as e:
                            import traceback
                            error_msg = str(e)
                            error_details = traceback.format_exc()
                            self.root.after(0, lambda: mess.showerror(
                                "Capture Error",
                                f"Failed to capture images:\n\n{error_msg}\n\n"
                                "Please check:\n"
                                "• Webcam is connected and working\n"
                                "• No other application is using the webcam\n"
                                "• Camera permissions are granted\n"
                                "• Sufficient lighting\n"
                                "• Your face is visible to the camera"
                            ))
                            self.root.after(0, lambda: self._set_status(f"Capture error: {error_msg}"))
                            print(f"Capture error details:\n{error_details}")
                    
                    threading.Thread(target=run_capture, daemon=True).start()
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    mess.showerror("Error", f"Failed to start capture: {str(e)}\n\nDetails:\n{error_details}")
                    self._set_status(f"Capture error: {e}")
            
            def train_for_roll():
                """Train model for specific roll number or all students"""
                roll = roll_entry.get().strip()
                
                if roll:
                    # Train for specific roll number
                    if not roll.isdigit():
                        mess.showerror("Invalid Roll Number", "Roll number must be numeric.")
                        return
                    
                    name, roll_found = self._get_student_by_roll(roll)
                    if not name:
                        mess.showerror(
                            "Student Not Found",
                            f"Student with Roll Number {roll} is not registered."
                        )
                        return
                    
                    # Check if images exist
                    people_dir = Path("people")
                    image_folder = None
                    if people_dir.exists():
                        for folder in people_dir.iterdir():
                            if folder.is_dir() and (folder.name.startswith(roll + "_") or folder.name.startswith(roll)):
                                image_folder = folder
                                break
                    
                    if not image_folder or not list(image_folder.glob("*.jpg")):
                        mess.showerror(
                            "No Images Found",
                            f"No images found for {name} (Roll: {roll}).\n\n"
                            "Please capture images first using 'Capture Images' button."
                        )
                        return
                    
                    response = mess.askyesno(
                        "Train Model",
                        f"Train model for:\n\n"
                        f"Name: {name}\n"
                        f"Roll Number: {roll}\n\n"
                        "Note: Model will be trained for ALL enrolled students.\n"
                        "Training will take several minutes.\n\n"
                        "Continue?"
                    )
                else:
                    # Train for all students
                    response = mess.askyesno(
                        "Train Model",
                        "Train model for ALL enrolled students.\n\n"
                        "Model training will take several minutes.\n"
                        "Make sure you have captured images for students.\n\n"
                        "Continue?"
                    )
                
                if not response:
                    return
                
                self._set_status("Starting model training...")
                train_window.destroy()
                
                # Run in separate thread to avoid blocking
                import threading
                def train():
                    try:
                        Model_Training()
                        self.root.after(0, lambda: mess.showinfo(
                            "Success", 
                            "Model training completed successfully!\n\n"
                            "You can now use the face recognition feature for attendance."
                        ))
                        self.root.after(0, lambda: self._set_status("Model training completed successfully"))
                    except FileNotFoundError as e:
                        error_msg = str(e)
                        self.root.after(0, lambda: mess.showerror("File Not Found", error_msg))
                        self.root.after(0, lambda: self._set_status(f"Training error: {error_msg}"))
                    except RuntimeError as e:
                        error_msg = str(e)
                        self.root.after(0, lambda: mess.showerror("Training Error", error_msg))
                        self.root.after(0, lambda: self._set_status(f"Training error: {error_msg}"))
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        error_msg = f"Training failed: {str(e)}\n\nDetails:\n{error_details}"
                        self.root.after(0, lambda: mess.showerror("Training Failed", error_msg))
                        self.root.after(0, lambda: self._set_status(f"Training error: {str(e)}"))
                
                threading.Thread(target=train, daemon=True).start()
            
            # Capture Images button
            tk.Button(
                buttons_frame,
                text="📸 Capture Images",
                command=capture_for_roll,
                bg=COLORS["accent"],
                fg="white",
                font=("Helvetica", 12, "bold"),
                padx=20,
                pady=12,
                cursor="hand2",
            ).pack(fill="x", pady=(0, 10))
            
            # Train Model button
            tk.Button(
                buttons_frame,
                text="🎓 Train Model",
                command=train_for_roll,
                bg=COLORS["primary"],
                fg="white",
                font=("Helvetica", 12, "bold"),
                padx=20,
                pady=12,
                cursor="hand2",
            ).pack(fill="x", pady=(0, 10))
            
            # Train All button
            tk.Button(
                buttons_frame,
                text="🎓 Train Model (All Students)",
                command=lambda: (roll_entry.delete(0, tk.END), train_for_roll()),
                bg=COLORS["success"],
                fg="white",
                font=("Helvetica", 11, "bold"),
                padx=20,
                pady=10,
                cursor="hand2",
            ).pack(fill="x")
            
        except ImportError as e:
            mess.showerror("Error", f"Cannot import training module: {e}")
    
    def _open_attendance_ui(self):
        """Open the full attendance UI"""
        try:
            # Use existing Tk mainloop and open UI in a child Toplevel instead of
            # spawning a second Tk instance (which can silently fail on some systems)
            from UI import AttendanceApp

            ui_window = tk.Toplevel(self.root)
            AttendanceApp(ui_window)
            self._set_status("Attendance Management Interface opened.")
            
        except ImportError as e:
            mess.showerror("Error", f"Cannot import UI module: {e}")
    
    def _open_view_attendance(self):
        """Open attendance viewer"""
        try:
            import csv
            from pathlib import Path
            from tkinter import ttk
            
            view_window = tk.Toplevel(self.root)
            view_window.title("View Attendance")
            view_window.configure(bg=COLORS["bg"])
            view_window.geometry("800x600")
            
            # Subject selection
            subject_frame = tk.Frame(view_window, bg=COLORS["card"], padx=20, pady=15)
            subject_frame.pack(fill="x", padx=20, pady=(20, 10))
            
            tk.Label(
                subject_frame,
                text="Select Subject:",
                font=("Helvetica", 14, "bold"),
                fg=COLORS["text"],
                bg=COLORS["card"],
            ).pack(side="left", padx=10)
            
            subject_var = tk.StringVar(value="Urdu")
            tk.Radiobutton(
                subject_frame,
                text="Urdu",
                variable=subject_var,
                value="Urdu",
                font=("Helvetica", 12),
                fg=COLORS["text"],
                bg=COLORS["card"],
                selectcolor=COLORS["card"],
                command=lambda: self._load_attendance_table(view_window, subject_var.get()),
            ).pack(side="left", padx=10)
            
            tk.Radiobutton(
                subject_frame,
                text="English",
                variable=subject_var,
                value="English",
                font=("Helvetica", 12),
                fg=COLORS["text"],
                bg=COLORS["card"],
                selectcolor=COLORS["card"],
                command=lambda: self._load_attendance_table(view_window, subject_var.get()),
            ).pack(side="left", padx=10)
            
            # Table frame
            table_frame = tk.Frame(view_window, bg=COLORS["bg"])
            table_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
            
            # Create treeview
            columns = ("Roll Number", "Name", "Attendance")
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=200, anchor="center")
            
            scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Store tree reference
            view_window.tree = tree
            view_window.subject_var = subject_var
            
            # Load initial data
            self._load_attendance_table(view_window, "Urdu")
            
        except Exception as e:
            mess.showerror("Error", f"Cannot open attendance viewer: {e}")
    
    def _load_attendance_table(self, window, subject):
        """Load attendance data into table"""
        try:
            tree = window.tree
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Load CSV
            csv_path = Path(f"{subject}_attendance/{subject}_Attendance.csv")
            if not csv_path.exists():
                tree.insert("", "end", values=("No data", "No data", "No data"))
                return
            
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tree.insert(
                        "",
                        "end",
                        values=(
                            row.get("Roll Number", ""),
                            row.get("Name", ""),
                            row.get("Attendance", "0"),
                        ),
                    )
        except Exception as e:
            mess.showerror("Error", f"Error loading attendance: {e}")
    
    def _open_test_recognition(self):
        """Open test recognition"""
        if not self.modules_status["recognition"]:
            mess.showerror(
                "Dependencies Missing",
                "Recognition requires keras and tensorflow.\n\n"
                "Install them using:\npip install keras tensorflow"
            )
            return
        
        mess.showinfo("Test Recognition", "This will open the recognition window.\nPress Q to quit.")
        try:
            from Recognizer import Recognition
            # Run in separate thread
            import threading
            def recognize():
                try:
                    Recognition(1, 0.80)  # Urdu subject, 0.80 threshold
                except Exception as e:
                    self.root.after(0, lambda: mess.showerror("Error", f"Recognition failed: {e}"))
            
            threading.Thread(target=recognize, daemon=True).start()
            self._set_status("Recognition window opened. Press Q to quit.")
        except Exception as e:
            mess.showerror("Error", f"Cannot start recognition: {e}")
    
    def _open_settings(self):
        """Open settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.configure(bg=COLORS["bg"])
        settings_window.geometry("500x400")
        
        frame = tk.Frame(settings_window, bg=COLORS["card"], padx=30, pady=30)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(
            frame,
            text="System Settings",
            font=("Helvetica", 20, "bold"),
            fg=COLORS["text"],
            bg=COLORS["card"],
        ).pack(pady=(0, 20))
        
        # System info
        info_text = f"""
System Information:
- Python: {sys.version.split()[0]}
- Modules Status:
  • Enrollment: {'✓' if self.modules_status['enrollment'] else '✗'}
  • Training: {'✓' if self.modules_status['training'] else '✗'}
  • Recognition: {'✓' if self.modules_status['recognition'] else '✗'}
  • UI: {'✓' if self.modules_status['ui'] else '✗'}
        """
        
        tk.Label(
            frame,
            text=info_text,
            font=("Helvetica", 11),
            fg=COLORS["text"],
            bg=COLORS["card"],
            justify="left",
        ).pack(anchor="w", pady=20)
    
    def _open_delete_student(self):
        """Open delete student data window"""
        delete_window = tk.Toplevel(self.root)
        delete_window.title("Delete Student Data")
        delete_window.configure(bg=COLORS["bg"])
        delete_window.geometry("600x500")
        
        frame = tk.Frame(delete_window, bg=COLORS["card"], padx=30, pady=30)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(
            frame,
            text="Delete Student Data",
            font=("Helvetica", 20, "bold"),
            fg=COLORS["text"],
            bg=COLORS["card"],
        ).pack(pady=(0, 20))
        
        # Warning label
        warning_label = tk.Label(
            frame,
            text="⚠️ WARNING: This will permanently delete all data for the student!",
            font=("Helvetica", 11, "bold"),
            fg=COLORS["danger"],
            bg=COLORS["card"],
            wraplength=500,
        )
        warning_label.pack(pady=(0, 20))
        
        # Roll Number
        tk.Label(frame, text="Roll Number:", font=("Helvetica", 12), fg=COLORS["text"], bg=COLORS["card"]).pack(anchor="w", pady=5)
        roll_entry = tk.Entry(frame, font=("Helvetica", 12), width=30)
        roll_entry.pack(pady=(0, 15), fill="x")
        
        # Info label
        info_label = tk.Label(
            frame,
            text="This will delete:\n• Student enrollment record\n• All attendance records (Urdu & English)\n• All captured images from people/ folder",
            font=("Helvetica", 10),
            fg="#94a3b8",
            bg=COLORS["card"],
            justify="left",
        )
        info_label.pack(pady=(0, 20), anchor="w")
        
        def delete_student():
            roll = roll_entry.get().strip()
            if not roll:
                mess.showerror("Error", "Please enter a roll number")
                return
            
            # Confirm deletion
            response = mess.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete ALL data for Roll Number: {roll}?\n\n"
                "This action cannot be undone!\n\n"
                "This will delete:\n"
                "• Enrollment record\n"
                "• All attendance records\n"
                "• All captured images\n\n"
                "Continue?",
                icon="warning"
            )
            
            if not response:
                return
            
            try:
                deleted_items = []
                
                # 1. Delete from Students_Enrollment.csv
                enrollment_path = Path("Students_Enrollment.csv")
                if enrollment_path.exists():
                    rows = []
                    student_name = None
                    with open(enrollment_path, newline="") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            if row.get("Roll Number", "").strip() != roll:
                                rows.append(row)
                            else:
                                student_name = row.get("Name", "")
                    
                    if student_name:
                        with open(enrollment_path, "w", newline="") as fh:
                            writer = csv.DictWriter(fh, fieldnames=["Name", "Roll Number"])
                            writer.writeheader()
                            writer.writerows(rows)
                        deleted_items.append("Enrollment record")
                
                # 2. Delete from attendance CSV files
                for subject in ["Urdu", "English"]:
                    attendance_path = Path(f"{subject}_attendance") / f"{subject}_Attendance.csv"
                    if attendance_path.exists():
                        rows = []
                        with open(attendance_path, newline="") as fh:
                            reader = csv.DictReader(fh)
                            for row in reader:
                                if row.get("Roll Number", "").strip() != roll:
                                    rows.append(row)
                        
                        with open(attendance_path, "w", newline="") as fh:
                            writer = csv.DictWriter(fh, fieldnames=["Roll Number", "Name", "Attendance"])
                            writer.writeheader()
                            writer.writerows(rows)
                        deleted_items.append(f"{subject} attendance")
                
                # 3. Delete image folders from people/ directory
                people_dir = Path("people")
                if people_dir.exists():
                    # Find folders that start with the roll number
                    for folder in people_dir.iterdir():
                        if folder.is_dir():
                            folder_name = folder.name
                            # Check if folder name starts with roll number
                            if folder_name.startswith(roll + "_") or folder_name.startswith(roll):
                                shutil.rmtree(folder)
                                deleted_items.append(f"Image folder: {folder_name}")
                
                # 4. Try to delete from MongoDB if connected
                try:
                    from MongoDB.retrieve_pymongo_data import database
                    data = database()
                    if data.connected:
                        # Delete from both Urdu and English collections
                        data.db.Urdu.delete_one({"Roll_number": roll})
                        data.db.English.delete_one({"Roll_number": roll})
                        deleted_items.append("MongoDB records")
                except Exception:
                    pass  # MongoDB not connected or error
                
                if deleted_items:
                    mess.showinfo(
                        "Success",
                        f"Student data for Roll Number {roll} has been deleted!\n\n"
                        f"Deleted:\n" + "\n".join(f"• {item}" for item in deleted_items)
                    )
                    self._set_status(f"Deleted student data for Roll Number: {roll}")
                    roll_entry.delete(0, tk.END)
                else:
                    mess.showinfo("Info", f"No data found for Roll Number: {roll}")
                    
            except Exception as e:
                mess.showerror("Error", f"Failed to delete student data: {e}")
                self._set_status(f"Delete error: {e}")
        
        # Delete button
        button_frame = tk.Frame(frame, bg=COLORS["card"])
        button_frame.pack(pady=20, fill="x")
        
        tk.Button(
            button_frame,
            text="Delete Student Data",
            command=delete_student,
            bg=COLORS["danger"],
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=20,
            pady=10,
            cursor="hand2",
        ).pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        tk.Button(
            button_frame,
            text="Cancel",
            command=delete_window.destroy,
            bg=COLORS["card"],
            fg=COLORS["text"],
            font=("Helvetica", 12),
            padx=20,
            pady=10,
            cursor="hand2",
        ).pack(side="left", expand=True, fill="x")
    
    def _show_unavailable(self, feature):
        """Show message for unavailable features"""
        mess.showinfo(
            "Feature Unavailable",
            f"{feature} is currently unavailable.\n\n"
            "This feature requires additional dependencies.\n"
            "Please install keras and tensorflow to enable ML features."
        )
    
    def _set_status(self, message):
        """Update status bar"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)


def main():
    """Main entry point"""
    print("=" * 60)
    print("Smart Attendance System Using Face Recognition")
    print("=" * 60)
    print("\nStarting Dashboard...")
    print("Note: Make sure MongoDB is running if you want database features.")
    print("Press Ctrl+C to exit.\n")
    
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
