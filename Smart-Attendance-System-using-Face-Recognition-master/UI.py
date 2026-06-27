import csv
import datetime as dt
import time
import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import messagebox as mess
from tkinter import ttk
from typing import Optional

from PIL import Image, ImageTk

# Heavy deps loaded lazily inside handlers so the UI can open even if TF/Keras missing.
try:
    from Generate_Dataset import Generate_Data, EnrollmentError
except Exception:
    Generate_Data = None
    class EnrollmentError(Exception):
        pass


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


class AttendanceApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("Smart Attendance System Using Facial Recognition")
        self.master.configure(bg=COLORS["bg"])
        self.master.state("zoomed")

        self.selection = tk.IntVar(value=1)
        self.recognition_threshold = tk.DoubleVar(value=0.80)
        self.auto_train = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Welcome! Enroll a student to get started.")
        self.clock_var = tk.StringVar()
        self.progress_var = tk.IntVar(value=0)
        self.progress_label_var = tk.StringVar(value="Training idle")
        self.filter_var = tk.StringVar()
        self.total_attendance_var = tk.StringVar(value="Total attendance: 0")

        self.training_thread: Optional[Thread] = None
        self.hero_photo = None
        self.status_frame: Optional[tk.Frame] = None

        self._configure_style()
        self._build_layout()
        self._tick_clock()

    # UI BUILDERS -----------------------------------------------------------------
    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Card.TLabelframe", background=COLORS["card"], foreground=COLORS["text"])
        style.configure("Card.TLabelframe.Label", font=("Helvetica", 16, "bold"), foreground=COLORS["text"])
        style.configure("Card.TCheckbutton", background=COLORS["card"], foreground=COLORS["text"])
        style.map(
            "Card.TCheckbutton",
            background=[("active", COLORS["card"])],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure("Accent.TButton", background=COLORS["accent"], foreground="white")
        style.map("Accent.TButton", background=[("active", COLORS["accent_dark"])])
        style.configure("Primary.TButton", background=COLORS["primary"], foreground="white")
        style.map("Primary.TButton", background=[("active", COLORS["primary_dark"])])
        style.configure("Danger.TButton", background=COLORS["danger"], foreground="white")
        style.map("Danger.TButton", background=[("active", "#b91c1c")])

    def _build_layout(self) -> None:
        header = tk.Frame(self.master, bg=COLORS["bg"])
        header.pack(fill="x", padx=24, pady=(16, 8))

        self._inject_hero(header)

        title_container = tk.Frame(header, bg=COLORS["bg"])
        title_container.pack(side="left", padx=20)
        tk.Label(
            title_container,
            text="Smart Attendance System",
            font=("Helvetica", 32, "bold"),
            fg=COLORS["text"],
            bg=COLORS["bg"],
        ).pack(anchor="w")
        tk.Label(
            title_container,
            text="Touchless enrollment • Instant recognition • Mongo synced",
            font=("Helvetica", 14),
            fg="#94a3b8",
            bg=COLORS["bg"],
        ).pack(anchor="w", pady=(6, 0))

        clock_frame = tk.Frame(header, bg=COLORS["card"], bd=0, relief="flat")
        clock_frame.pack(side="right", padx=10)
        tk.Label(
            clock_frame,
            textvariable=self.clock_var,
            font=("Helvetica", 16, "bold"),
            fg=COLORS["text"],
            bg=COLORS["card"],
            padx=24,
            pady=12,
        ).pack()

        content = tk.Frame(self.master, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        left_card = ttk.Labelframe(content, text="Enrollment & Training", style="Card.TLabelframe")
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right_card = ttk.Labelframe(content, text="Attendance & Insights", style="Card.TLabelframe")
        right_card.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self._build_enrollment_card(left_card)
        self._build_attendance_card(right_card)
        self._build_status_bar()

    def _inject_hero(self, container: tk.Frame) -> None:
        try:
            hero = Image.open("landscape.png").resize((220, 110))
            self.hero_photo = ImageTk.PhotoImage(hero)
            tk.Label(container, image=self.hero_photo, bg=COLORS["bg"]).pack(side="left", padx=(0, 10))
        except Exception:
            pass

    def _build_enrollment_card(self, parent: ttk.Labelframe) -> None:
        for index in range(2):
            parent.columnconfigure(index, weight=1)

        form_frame = tk.Frame(parent, bg=COLORS["card"])
        form_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        tk.Label(form_frame, text="Student Roll Number", bg=COLORS["card"], fg=COLORS["text"], font=("Helvetica", 12)).grid(
            row=0, column=0, sticky="w"
        )
        self.roll_entry = ttk.Entry(form_frame, font=("Helvetica", 12))
        self.roll_entry.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        tk.Label(form_frame, text="Student Full Name", bg=COLORS["card"], fg=COLORS["text"], font=("Helvetica", 12)).grid(
            row=2, column=0, sticky="w"
        )
        self.name_entry = ttk.Entry(form_frame, font=("Helvetica", 12))
        self.name_entry.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        form_frame.columnconfigure(0, weight=1)

        button_frame = tk.Frame(parent, bg=COLORS["card"])
        button_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
        ttk.Button(
            button_frame, text="Capture Dataset", style="Accent.TButton", command=self._handle_capture
        ).pack(side="left", expand=True, fill="x", padx=(0, 10))
        ttk.Button(
            button_frame, text="Train Model", style="Primary.TButton", command=self._trigger_training
        ).pack(side="left", expand=True, fill="x")

        quick_frame = tk.Frame(parent, bg=COLORS["card"])
        quick_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        ttk.Button(
            quick_frame,
            text="Submit Enrollment (0% + show)",
            style="Primary.TButton",
            command=self._handle_submit_enroll,
        ).pack(fill="x")

        options_frame = tk.Frame(parent, bg=COLORS["card"])
        options_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        ttk.Checkbutton(
            options_frame,
            text="Auto-train after capture",
            variable=self.auto_train,
            style="Card.TCheckbutton",
        ).pack(side="left")

        threshold_frame = tk.Frame(parent, bg=COLORS["card"])
        threshold_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        tk.Label(
            threshold_frame,
            text="Recognition Confidence",
            fg=COLORS["text"],
            bg=COLORS["card"],
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")
        scale = ttk.Scale(
            threshold_frame,
            from_=0.60,
            to=0.95,
            orient="horizontal",
            variable=self.recognition_threshold,
            command=lambda _: self._update_threshold_label(),
        )
        scale.pack(fill="x", pady=6)
        self.threshold_label = tk.Label(
            threshold_frame,
            text="80%",
            fg="#e2e8f0",
            bg=COLORS["card"],
            font=("Helvetica", 12, "bold"),
        )
        self.threshold_label.pack(anchor="e")

        progress_frame = tk.Frame(parent, bg=COLORS["card"])
        progress_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        ttk.Progressbar(
            progress_frame, orient="horizontal", mode="determinate", variable=self.progress_var
        ).pack(fill="x", pady=(0, 6))
        tk.Label(
            progress_frame,
            textvariable=self.progress_label_var,
            fg="#cbd5f5",
            bg=COLORS["card"],
            font=("Helvetica", 11),
        ).pack(anchor="w")

    def _build_attendance_card(self, parent: ttk.Labelframe) -> None:
        parent.columnconfigure(0, weight=1)

        class_frame = tk.Frame(parent, bg=COLORS["card"])
        class_frame.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")
        tk.Label(
            class_frame, text="Active Lecture", fg=COLORS["text"], bg=COLORS["card"], font=("Helvetica", 12, "bold")
        ).pack(anchor="w")
        lectures = tk.Frame(class_frame, bg=COLORS["card"])
        lectures.pack(anchor="w", pady=6)
        ttk.Radiobutton(lectures, text="Urdu", variable=self.selection, value=1).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(lectures, text="English", variable=self.selection, value=2).pack(side="left")

        action_frame = tk.Frame(parent, bg=COLORS["card"])
        action_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ttk.Button(
            action_frame,
            text="Start Live Recognition",
            style="Accent.TButton",
            command=self._start_recognition,
        ).pack(fill="x", pady=(0, 6))
        ttk.Button(
            action_frame,
            text="Refresh Attendance",
            style="Primary.TButton",
            command=self._load_attendance,
        ).pack(fill="x")

        filter_frame = tk.Frame(parent, bg=COLORS["card"])
        filter_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        tk.Label(filter_frame, text="Quick Filter", fg=COLORS["text"], bg=COLORS["card"]).grid(row=0, column=0, sticky="w")
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        filter_entry.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(filter_frame, text="Apply", command=self._load_attendance).grid(row=1, column=1, padx=(6, 0))
        ttk.Button(filter_frame, text="Clear Table", style="Danger.TButton", command=self._clear_tree).grid(
            row=1, column=2, padx=(6, 0)
        )
        filter_frame.columnconfigure(0, weight=1)

        tree_frame = tk.Frame(parent, bg=COLORS["card"])
        tree_frame.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="nsew")
        parent.rowconfigure(3, weight=1)
        columns = ("roll_no", "name", "attendance")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        for col, label in zip(columns, ("Roll Number", "Name", "Attendance")):
            self.tree.heading(col, text=label)
            self.tree.column(col, anchor="center", width=150)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        total_frame = tk.Frame(parent, bg=COLORS["card"])
        total_frame.grid(row=4, column=0, padx=20, pady=(0, 6), sticky="ew")
        tk.Label(
            total_frame,
            textvariable=self.total_attendance_var,
            fg=COLORS["text"],
            bg=COLORS["card"],
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")

        log_frame = tk.Frame(parent, bg=COLORS["card"])
        log_frame.grid(row=5, column=0, padx=20, pady=(0, 12), sticky="ew")
        tk.Label(log_frame, text="Activity Log", fg=COLORS["text"], bg=COLORS["card"]).pack(anchor="w")
        self.log_text = tk.Text(
            log_frame,
            height=6,
            bg="#0b1120",
            fg="#e2e8f0",
            relief="flat",
            font=("Consolas", 10),
        )
        self.log_text.pack(fill="both", expand=True, pady=6)
        self.log_text.config(state="disabled")

    def _build_status_bar(self) -> None:
        self.status_frame = tk.Frame(self.master, bg=COLORS["accent"])
        self.status_frame.pack(fill="x", side="bottom")
        self.status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            fg="white",
            bg=COLORS["accent"],
            font=("Helvetica", 12, "bold"),
            padx=12,
            pady=6,
        )
        self.status_label.pack(side="left", fill="x", expand=True)
        self.quit_button = tk.Button(
            self.status_frame,
            text="Quit",
            command=self.master.destroy,
            bg=COLORS["danger"],
            fg="white",
            bd=0,
            relief="flat",
            padx=20,
            pady=6,
        )
        self.quit_button.pack(side="right")

    # EVENT HANDLERS ----------------------------------------------------------------
    def _update_threshold_label(self) -> None:
        self.threshold_label.config(text=f"{int(self.recognition_threshold.get()*100)}%")

    def _tick_clock(self) -> None:
        now = dt.datetime.now().strftime("%A, %d %B %Y  |  %I:%M:%S %p")
        self.clock_var.set(now)
        self.master.after(1000, self._tick_clock)

    def _handle_capture(self) -> None:
        name = self.name_entry.get().strip()
        roll = self.roll_entry.get().strip()
        if not name or not roll:
            mess.showerror("Missing data", "Please provide both roll number and full name.")
            return

        self._log(f"Capture started for {name} ({roll})")
        self._set_status("Capturing dataset, please look at the camera...", "info")

        def task():
            if Generate_Data is None:
                from Generate_Dataset import Generate_Data as _GD
                return _GD(name, roll)
            return Generate_Data(name, roll)

        self._run_async(task, on_success=self._on_capture_complete, context="Dataset capture")

    def _on_capture_complete(self, result) -> None:
        path, total = result
        self._log(f"Captured {total} images → {path}")
        self._set_status("Dataset captured successfully.", "success")
        mess.showinfo("Capture complete", f"Stored {total} images inside:\n{path}")
        self._load_attendance()
        if self.auto_train.get():
            self._trigger_training()

    def _handle_submit_enroll(self) -> None:
        """Submit enrollment, initialize 0 attendance, and show in the right-side table."""
        name = self.name_entry.get().strip()
        roll = self.roll_entry.get().strip()
        if not name or not roll:
            mess.showerror("Missing data", "Please provide both roll number and full name.")
            return
        if not roll.isdigit():
            mess.showerror("Invalid roll number", "Roll number must be numeric.")
            return

        try:
            self._upsert_enrollment_records(name, roll)
            # Ensure the student exists in both subject attendance files with 0 total attendance
            self._upsert_attendance_file(Path("Urdu_attendance/Urdu_Attendance.csv"), roll, name)
            self._upsert_attendance_file(Path("English_attendance/English_Attendance.csv"), roll, name)
        except Exception as exc:
            self._handle_error("Enrollment", exc)
            return

        self._set_status(f"Student {name} ({roll}) enrolled with 0 total attendance.", "success")
        self._log(f"Enrollment submitted → {name} ({roll})")
        # Refresh current subject view so the new student appears immediately
        self._load_attendance()

    def _trigger_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            self._set_status("Training already running...", "warning")
            return
        self.progress_var.set(0)
        self.progress_label_var.set("Training in progress...")
        self._set_status("Training face recognition model...", "info")
        self.training_thread = Thread(target=self._run_training, daemon=True)
        self.training_thread.start()
        self._animate_progress()

    def _run_training(self) -> None:
        try:
            from Model_train import Model_Training
            Model_Training()
        except Exception as exc:
            self.master.after(0, lambda: self._handle_error("Training", exc))
        else:
            self.master.after(0, self._on_training_complete)

    def _on_training_complete(self) -> None:
        self.progress_var.set(100)
        self.progress_label_var.set("Training completed")
        self._set_status("Model trained successfully.", "success")
        self._log("Model training completed.")

    def _animate_progress(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            new_value = min(self.progress_var.get() + 3, 90)
            self.progress_var.set(new_value)
            self.master.after(350, self._animate_progress)
        else:
            if self.progress_var.get() < 100:
                self.progress_var.set(100)
                self.progress_label_var.set("Training completed")

    def _start_recognition(self) -> None:
        lecture = self.selection.get()
        threshold = round(float(self.recognition_threshold.get()), 2)
        self._set_status("Starting recognition... Press Q to close the camera window.", "info")
        self._log(f"Recognition launched (threshold={threshold})")

        def task():
            from Recognizer import Recognition
            Recognition(lecture, threshold)

        self._run_async(
            task,
            on_success=lambda _: (
                self._set_status("Recognition session closed.", "success"),
                self._load_attendance(),
            ),
            context="Recognition",
        )

    def _load_attendance(self) -> None:
        subject = (
            "Urdu_attendance/Urdu_Attendance.csv"
            if self.selection.get() == 1
            else "English_attendance/English_Attendance.csv"
        )
        csv_path = Path(subject)
        if not csv_path.exists():
            self._handle_error("Attendance", FileNotFoundError(f"Attendance file not found: {csv_path}"))
            self.total_attendance_var.set("Total attendance: 0")
            return
        self._clear_tree()
        query = self.filter_var.get().strip().lower()
        total_att = 0
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                roll = row.get("Roll Number", "")
                name = row.get("Name", "")
                attendance = row.get("Attendance", "")
                try:
                    total_att += int(str(attendance or "0").strip() or 0)
                except ValueError:
                    pass
                if query and query not in roll.lower() and query not in name.lower():
                    continue
                self.tree.insert("", "end", values=(roll, name, attendance))
        self._log(f"Attendance refreshed from {csv_path.name}")
        self._set_status(f"Loaded attendance for {csv_path.name}", "success")
        self.total_attendance_var.set(f"Total attendance: {total_att}")

    def _clear_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    # HELPERS ----------------------------------------------------------------------
    def _upsert_enrollment_records(self, name: str, roll: str) -> None:
        """Ensure the student is present in Students_Enrollment.csv."""
        path = Path("Students_Enrollment.csv")
        header = ["Name", "Roll Number"]
        rows = []
        if path.exists():
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))

        exists = False
        for row in rows:
            if row.get("Roll Number", "") == roll:
                row["Name"] = name  # keep latest name
                exists = True
                break
        if not exists:
            rows.append({"Name": name, "Roll Number": roll})

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def _upsert_attendance_file(self, path: Path, roll: str, name: str) -> None:
        """Ensure the student exists in a subject attendance CSV with 0 attendance."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ["Roll Number", "Name", "Attendance"]
        rows = []
        if path.exists():
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))

        exists = False
        for row in rows:
            if row.get("Roll Number", "") == roll:
                row["Name"] = name
                exists = True
                break
        if not exists:
            rows.append({"Roll Number": roll, "Name": name, "Attendance": "0"})

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def _run_async(self, func, on_success=None, context: str = "Operation") -> None:
        def runner():
            try:
                result = func()
            except (EnrollmentError, ValueError) as exc:
                self.master.after(0, lambda: self._handle_error(context, exc))
            except Exception as exc:
                self.master.after(0, lambda: self._handle_error(context, exc))
            else:
                if on_success:
                    self.master.after(0, lambda: on_success(result))

        Thread(target=runner, daemon=True).start()

    def _handle_error(self, context: str, exc: Exception) -> None:
        message = f"{context} failed: {exc}"
        self._set_status(message, "danger")
        self._log(f"ERROR: {message}")
        mess.showerror(context, message)

    def _set_status(self, message: str, level: str = "info") -> None:
        palette = {
            "info": COLORS["accent"],
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "danger": COLORS["danger"],
        }
        color = palette.get(level, COLORS["accent"])
        if self.status_frame and self.status_label:
            self.status_frame.configure(bg=color)
            self.status_label.configure(bg=color)
        self.status_var.set(message)

    def _log(self, entry: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {entry}"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")


def main():
    window = tk.Tk()
    AttendanceApp(window)
    window.mainloop()


if __name__ == "__main__":
    main()

