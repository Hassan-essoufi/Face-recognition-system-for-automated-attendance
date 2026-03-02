# 🚀 FacePresence

### AI-Powered Real-Time Attendance Management Platform

## Overview
FacePresence is a full-stack attendance platform with:
- Frontend dashboard (React + Vite)
- FastAPI backend
- Real-time and video-based face recognition workflows
- Attendance export to CSV
---
### Backend:
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-009688)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)


---
## Main Features
- Person management (upload or camera capture)
- Face crop before augmentation (camera flow)
- Attendance dashboard with analytics
- Weekly attendance and attendance-rate charts
- Video recognition pipeline
- CSV export for attendance by selected date
---

# 🖥 Live System Overview

## 🔷 Dashboard

📊 Real-time attendance metrics
📈 Weekly trend visualization
📉 Attendance rate analysis
📋 Absentee tracking

### 📸 Screenshot – Dashboard

`docs/screenshots/dashboard.png`

<img src="docs/screenshots/dashboard.png" width="800"/>

---

## 👤 Person Management

* Register new individuals
* Upload or capture face images
* Automatic face cropping
* Embedding generation and storage

### 📸 Screenshot – Person Management

`docs/screenshots/persons.png`

<img src="docs/screenshots/person-management.png" width="800"/>

---

## 🎥 Real-Time Recognition

* Live webcam detection
* Face embedding extraction
* Cosine similarity matching
* Automatic attendance marking

### 📸 Screenshot – Real-Time Stream

`docs/screenshots/real-time-recognition.png`

<img src="docs/screenshots/real-time-recognition.png" width="800"/>

---

## 📹 Video Recognition Pipeline

* Upload recorded video
* Frame-by-frame face detection
* Recognition across video stream
* Batch attendance generation

### 📸 Screenshot – Video Processing

`docs/screenshots/video-recognition.png`

<img src="docs/screenshots/video-recognition.png" width="800"/>

---

## 📁 CSV Export & Analytics

* Export attendance by selected date
* Generate structured CSV reports
* Dashboard statistics from CSV

### 📸 Screenshot – CSV Export

`docs/screenshots/csv-export.png`

<img src="docs/screenshots/csv-export.png" width="800"/>

---

# 🧠 Machine Learning Pipeline

### Step 1 – Face Detection

MTCNN detects and aligns faces.

### Step 2 – Face Embedding

FaceNet generates 128-d embedding vectors using:

* PyTorch
* OpenCV

### Step 3 – Recognition Logic

* Cosine similarity comparison
* Threshold-based validation
* Confidence scoring

---

# 🏗 Architecture

```
    Frontend
        ↓ 
FastAPI Backend
        ↓
Face Recognition Engine
        ↓
SQL Database (SQLAlchemy ORM)
```

# 🔌 API Documentation

Interactive Swagger UI available at:

```
/docs
```

### Main Endpoints

```
GET    /attendance/all
GET    /attendance/absent
POST   /attendance/real_time_stream
POST   /attendance/real_time_stream/stop
POST   /attendance/video_recognition
GET    /attendance/export_csv
GET    /attendance/stats_from_csv
```

---

# 🚀 How to Run

## Backend

```bash
pip install -r requirements.txt
uvicorn backend.src.api.main:app --reload
```

## Frontend

```bash
npm install
npm run dev
```

---

## Notes
- Frontend package versions are listed in `package.json`.
- Backend dependencies are pinned in `requirements.txt`.
---
# 📊 Engineering Decisions

* Model loaded once at startup
* Modular service-layer architecture
* CSV-driven lightweight analytics
* Controlled stream lifecycle
* Separation of ML pipeline and API layer

---


# 👤 Author

Essoufi Hassan
ML & data Engineer

---

# Notice

⭐ If you find this project useful or inspiring, consider giving it a star to support the work!

---

## Screenshots (placeholders)

### Frontend Screenshots
> Replace the image paths below with your real screenshots.

![Frontend - Dashboard](docs/screenshots/frontend-dashboard.png)
![Frontend - Person Management](docs/screenshots/frontend-person-management.png)

### FastAPI Screenshots
![FastAPI - Swagger UI](docs/screenshots/fastapi-swagger.png)
![FastAPI - Endpoint Test](docs/screenshots/fastapi-endpoint-test.png)

### Video Recognition Example
![Video Recognition Example](docs/screenshots/video-recognition-example.png)

### CSV Generation Example
![CSV Generation Example](docs/screenshots/csv-generation-example.png)

