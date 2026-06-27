
from pathlib import Path
from typing import Optional

import pandas as pd
from pymongo import MongoClient
from pymongo import errors as mongo_errors


class database:
    def __init__(self, uri: str = "mongodb://localhost:27017/", timeout_ms: int = 5000):
        self.client = None
        self.db = None
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            self.client.server_info()
            self.db = self.client.students
        except mongo_errors.PyMongoError as exc:
            print(f"[warning] Unable to connect to MongoDB: {exc}")

    @property
    def connected(self) -> bool:
        return self.db is not None

    def update(self, name: str, lecture: str) -> None:
        if not self.connected:
            return
        collection = self.db.Urdu if str(lecture) == "1" else self.db.English
        collection.update_one({"Name": name}, {"$inc": {"Attendance": 1}}, upsert=True)

    def _export_path(self, lecture: str) -> Path:
        directory = Path("Urdu_attendance") if str(lecture) == "1" else Path("English_attendance")
        directory.mkdir(parents=True, exist_ok=True)
        subject = "Urdu" if str(lecture) == "1" else "English"
        return directory / f"{subject}_Attendance.csv"

    def export_csv(self, lecture: str) -> Optional[Path]:
        if not self.connected:
            print("[info] Skipped attendance export because MongoDB is offline.")
            return None

        collection = self.db.Urdu if str(lecture) == "1" else self.db.English
        records = list(collection.find())
        if not records:
            df = pd.DataFrame(columns=["Roll Number", "Name", "Attendance"])
        else:
            df = pd.DataFrame(
                [
                    {
                        "Roll Number": rec.get("Roll_number", ""),
                        "Name": rec.get("Name", ""),
                        "Attendance": rec.get("Attendance", 0),
                    }
                    for rec in records
                ]
            )

        export_path = self._export_path(lecture)
        df.to_csv(export_path, index=False)
        return export_path
