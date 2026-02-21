from datetime import datetime
from sqlalchemy.orm import Session
from src.database.models import Attendance

def mark_attendance_service(
    db: Session,
    person_name: str,
    confidence: float,
    source: str = "webcam",
    threshold: float = 0.7
):
    """
    Marks attendance:
    - Always updates existing record for today
    - If no record exists, creates a new one
    """

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    confidence = float(confidence)

    # Check if attendance already exists for today
    existing = db.query(Attendance).filter(
        Attendance.person_name == person_name,
        Attendance.date == today
    ).first()

    status = "present" if confidence >= threshold else "absent"

    if existing:
        # Overwrite existing record
        existing.confidence = confidence
        existing.status = status
        existing.time = now_time
        existing.source = source
        db.commit()
        db.refresh(existing)
        return {"status": "updated", "id": existing.id}

    # Create new record if none exists
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
