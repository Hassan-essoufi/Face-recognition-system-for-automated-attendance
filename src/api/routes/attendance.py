from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import csv
from src.api.schemas import AttendanceRead
from src.api.dependencies import get_db
from src.database.models import Attendance
from src.services.attendance_service import mark_attendance_service
from src.recognition.webcam_attendance import real_time_att
from src.recognition.video_recognition import VideoFaceRecognition

router = APIRouter()

@router.post("/real_time")
def run_real_time_attendance(
    csv_file: str = Form(...),
    threshold: float = Form(0.6),
    db: Session = Depends(get_db)
):
    real_time_att(
        embeddings_path='embeddings.npy',
        csv_file=csv_file,
        threshold=threshold
    )

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mark_attendance_service(
                db=db,
                person_name=row["Name"],
                confidence=None,
                source="webcam",
                status=row["Status"]
            )

    return {"status": "completed", "file": csv_file}


# Video recognition

@router.post("/video_recognition")
def run_video_recognition(
    input_video: UploadFile = File(...),
    output_video_path: str = Form("output.mp4"),
    attendance_csv: str = Form("attendance_video.csv"),
    db: Session = Depends(get_db)
):
    temp_video_path = f"temp_{input_video.filename}"
    with open(temp_video_path, "wb") as f:
        f.write(input_video.file.read())

    recognizer = VideoFaceRecognition()
    recognizer.process_video(
        input_path=temp_video_path,
        output_path=output_video_path,
        attendance_file=attendance_csv
    )

    # Store attendance in DB
    import csv
    from datetime import datetime

    with open(attendance_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mark_attendance_service(
                db=db,
                person_name=row["Name"],
                confidence=None,
                source="video",
                status=row["Status"]
            )

    return {"status": "success", "output_video": output_video_path, "attendance_csv": attendance_csv}

# Get all attendance
@router.get("/all", response_model=List[AttendanceRead])
def get_all_attendance(db: Session = Depends(get_db)):
    return db.query(Attendance).all()


# Get absent records
@router.get("/absent", response_model=List[AttendanceRead])
def get_absent(db: Session = Depends(get_db)):
    return db.query(Attendance).filter(Attendance.status == "absent").all()
