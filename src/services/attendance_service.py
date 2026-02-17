from datetime import datetime
from sqlalchemy.orm import Session
from src.database.models import Attendance

CONFIDENCE_THRESHOLD = 0.7  

def mark_attendance_service(db: Session, person_name: str, confidence: float, source: str = "webcam"):
    """
    Marks attendance with this rules:
    - Prevent duplicates per day
    - Use confidence to decide present/absent
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    existing = db.query(Attendance).filter(
        Attendance.person_name == person_name,
        Attendance.date == today
    ).first()
    if existing:
        return {"status": "already_marked", "id": existing.id}

    status = "present" if confidence >= CONFIDENCE_THRESHOLD else "absent"

    # Create record
    record = Attendance(
        person_name=person_name,
        date=today,
        time=now_time,
        status=status,
        confidence=confidence,
        source=source
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return {"status": status, "id": record.id}
