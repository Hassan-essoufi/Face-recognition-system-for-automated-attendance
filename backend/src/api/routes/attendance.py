import os
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List
import csv
from backend.src.api.schemas import AttendanceRead
from backend.src.api.dependencies import get_db
from backend.src.database.models import Attendance
from backend.src.services.attendance_service import mark_attendance_service
from backend.src.recognition.webcam_attendance import (
    real_time_att_stream,
    stop_real_time_att_stream,
    consume_stream_recognitions,
)
from backend.src.recognition.video_recognition import VideoFaceRecognition
import uuid
from werkzeug.utils import secure_filename

router = APIRouter()


@router.get("/stats_from_csv")
def get_attendance_stats_from_csv(db: Session = Depends(get_db)):
    now = datetime.now()
    start_day = now - timedelta(days=6)
    week_days = [start_day + timedelta(days=idx) for idx in range(7)]

    week_labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    week_map = {
        day.date(): {
            "day": f"{week_labels[day.weekday()]} {day.strftime('%d/%m')}",
            "present": 0,
            "absent": 0,
        }
        for day in week_days
    }

    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    month_keys = []
    for step in range(5, -1, -1):
        month_cursor = now.month - step
        year_cursor = now.year
        while month_cursor <= 0:
            month_cursor += 12
            year_cursor -= 1
        while month_cursor > 12:
            month_cursor -= 12
            year_cursor += 1
        month_keys.append((year_cursor, month_cursor))

    month_map = {
        key: {
            "month": month_labels[key[1] - 1],
            "present": 0,
            "total": 0,
        }
        for key in month_keys
    }

    records = db.query(Attendance).all()

    for record in records:
        try:
            row_date = datetime.strptime(record.date, "%Y-%m-%d")
        except (TypeError, ValueError):
            continue

        status = (record.status or "").strip().lower()
        if status not in {"present", "absent"}:
            continue

        row_day = row_date.date()
        if row_day in week_map:
            week_map[row_day][status] += 1

        month_key = (row_date.year, row_date.month)
        if month_key in month_map:
            month_map[month_key]["total"] += 1
            if status == "present":
                month_map[month_key]["present"] += 1

    week_data = [week_map[day.date()] for day in week_days]
    trend_data = []
    for key in month_keys:
        month_stat = month_map[key]
        total = month_stat["total"]
        rate = round((month_stat["present"] / total) * 100, 1) if total > 0 else 0.0
        trend_data.append({"month": month_stat["month"], "taux": rate})

    return {
        "week_data": week_data,
        "trend_data": trend_data,
    }


@router.get("/real_time_stream")
def stream_real_time_attendance(threshold: float = 0.6):
    return StreamingResponse(
        real_time_att_stream(
            embeddings_path='backend/embeddings/embeddings.npy',
            threshold=threshold
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/real_time_stream/stop")
def stop_real_time_attendance_stream(threshold: float = Form(0.6), db: Session = Depends(get_db)):
    stop_real_time_att_stream()
    detections = consume_stream_recognitions(threshold=threshold)

    for detection in detections:
        mark_attendance_service(
            db=db,
            person_name=detection["name"],
            confidence=float(detection["confidence"]),
            source="webcam",
            threshold=threshold,
        )

    return {"status": "stopped", "detections": detections}


# Video recognition
@router.post("/video_recognition")
def run_video_recognition(
    input_video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    video_path = os.path.join("backend/video_db", "input", f"{secure_filename(input_video.filename)}")

    with open(video_path, "wb") as f:
        f.write(input_video.file.read())
    output_video_path = os.path.join("backend/video_db", "output", f"{uuid.uuid4().hex}.mp4")
    attendance_csv = os.path.join("backend/attendance", "videos", f"{uuid.uuid4().hex}.csv")
    recognizer = VideoFaceRecognition()
    recognizer.process_video(
        input_path=video_path,
        output_path=output_video_path,
        attendance_file=attendance_csv
    )

    # Store attendance in DB

    with open(attendance_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mark_attendance_service(
                db=db,
                person_name=row["Name"],
                confidence=float(row["confidence"]),
                source="video"
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


@router.get("/export_csv")
def export_attendance_csv(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Format de date invalide. Utilisez YYYY-MM-DD")

    attendance_dirs = [
        os.path.join("backend", "attendance", "videos"),
        os.path.join("backend", "attendance", "webcam"),
    ]

    matching_files: list[tuple[str, float]] = []

    for directory in attendance_dirs:
        if not os.path.isdir(directory):
            continue

        for file_name in os.listdir(directory):
            if not file_name.lower().endswith(".csv"):
                continue

            file_path = os.path.join(directory, file_name)
            if not os.path.isfile(file_path):
                continue

            file_mtime = os.path.getmtime(file_path)
            file_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
            if file_date == date:
                matching_files.append((file_path, file_mtime))

    if not matching_files:
        raise HTTPException(status_code=404, detail="Non existence de la présence pour ce jour")

    latest_file_path = max(matching_files, key=lambda item: item[1])[0]
    return FileResponse(
        path=latest_file_path,
        media_type="text/csv",
        filename=f"presence_{date}.csv"
    )
