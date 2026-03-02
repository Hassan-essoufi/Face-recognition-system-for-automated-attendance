import cv2
import csv
import numpy as np
from numpy.linalg import norm
from datetime import datetime
from threading import Event

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN


STREAM_STOP_EVENT = Event()
STREAM_RECOGNITIONS = {}
STREAM_THRESHOLD = 0.6


def stop_real_time_att_stream():
    STREAM_STOP_EVENT.set()


def consume_stream_recognitions(threshold=0.6):
    if not STREAM_RECOGNITIONS:
        return []

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items = []
    for name, conf in STREAM_RECOGNITIONS.items():
        confidence = float(conf)
        status = "Present" if confidence >= threshold else "Absent"
        items.append(
            {
                "name": name,
                "status": status,
                "confidence": confidence,
                "time": time_now,
            }
        )

    STREAM_RECOGNITIONS.clear()
    return items


def real_time_att_stream(embeddings_path, threshold=0.6):
    global STREAM_THRESHOLD
    STREAM_THRESHOLD = threshold
    STREAM_STOP_EVENT.clear()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    data = np.load(embeddings_path, allow_pickle=True).item()
    known_names = list(data.keys())
    known_embeddings = np.array(list(data.values()))
    known_embeddings = known_embeddings / norm(known_embeddings, axis=1, keepdims=True)
    STREAM_RECOGNITIONS.clear()
    for name in known_names:
        STREAM_RECOGNITIONS[name] = 0.0

    cap = cv2.VideoCapture(0)

    try:
        while not STREAM_STOP_EVENT.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)

            if boxes is not None:
                faces = mtcnn.extract(rgb, boxes, save_path=None)

                for box, face in zip(boxes, faces):
                    face = face.unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = facenet(face).cpu().numpy()[0]

                    embedding = embedding / norm(embedding)
                    scores = np.dot(known_embeddings, embedding)

                    best_idx = np.argmax(scores)
                    best_score = float(scores[best_idx])

                    if best_score >= threshold:
                        name = known_names[best_idx]
                        STREAM_RECOGNITIONS[name] = max(
                            STREAM_RECOGNITIONS.get(name, 0.0),
                            best_score,
                        )
                        color = (0, 255, 0)
                        label = f"{name} ({best_score:.2f})"
                    else:
                        color = (0, 0, 255)
                        label = f"Unknown ({best_score:.2f})"

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()
        STREAM_STOP_EVENT.clear()

def real_time_att(
    embeddings_path,
    csv_file,
    threshold=0.6):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    data = np.load(embeddings_path, allow_pickle=True).item()
    known_names = list(data.keys())
    known_embeddings = np.array(list(data.values()))

    known_embeddings = known_embeddings / norm(known_embeddings, axis=1, keepdims=True)

    attendance = {name: "Absent" for name in known_names}
    confidence = {name: 0.0 for name in known_names}

    cap = cv2.VideoCapture(0)
    print("Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            faces = mtcnn.extract(rgb, boxes, save_path=None)

            for box, face in zip(boxes, faces):
                face = face.unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = facenet(face).cpu().numpy()[0]

                embedding = embedding / norm(embedding)

                scores = np.dot(known_embeddings, embedding)

                best_idx = np.argmax(scores)
                best_score = scores[best_idx]

                if best_score >= threshold:
                    name = known_names[best_idx]
                    attendance[name] = "Present"
                    confidence[name] = max(confidence[name], float(best_score))
                    color = (0, 255, 0)
                    label = f"{name} ({best_score:.2f})"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    label = f"Unknown ({best_score:.2f})"

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        cv2.imshow("FaceNet Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Status", "confidence", "Time"])
        for name, status in attendance.items():
            conf = round(confidence.get(name, 0.0), 3)
            writer.writerow([name, status, conf, time_now])

    print("Attendance saved to", csv_file)

