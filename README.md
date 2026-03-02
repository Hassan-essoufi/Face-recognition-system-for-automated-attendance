<h1 align="center">
  MPresence IA: Real-Time Attendance Management System
</h1>


---


# 🧠 Overview

MPresence IA is a AI-powered attendance management system built with modern with deep learning.

It combines:

- ⚡ High-performance FastAPI backend  
- 🎯 Real-time face recognition  
- 🎥 Video-based recognition pipeline  
- 📊 Interactive analytics dashboard  
- 📁 CSV attendance export  

---

# 🛠 Tech Stack

## 🔷 Backend
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?logo=pytorch&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-red)

## 🔷 Frontend
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-Fast%20Build-646CFF?logo=vite&logoColor=white)
![MUI](https://img.shields.io/badge/MUI-UI%20Framework-007FFF?logo=mui&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-Data%20Visualization-orange)

---

# ✨ Core Features

- 👤 Person registration (upload or camera capture)
- ✂ Automatic face cropping before embedding
- 🎥 Real-time webcam recognition
- 📹 Video file recognition pipeline
- 📊 Dashboard attendance analytics
- 📉 Attendance rate visualization
- 📁 CSV export by selected date
- 📋 Absentee tracking

---

# 🖥 Live System Overview

---
## 🔌 API Interface (FastAPI)

Interactive Swagger documentation available at:

```
http://localhost:8000/docs
```

<img src="docs/screenshots/Face_recognition_api.png" width="800"/>


---

## 👤 Person Management

- Register new individuals  
- Upload or capture face images  
- Automatic face cropping  
- Embedding generation & storage  

<img src="docs/screenshots/persons_gestion.png" width="800"/>

---

## 🎥 Real-Time Recognition

- Live webcam detection  
- Face embedding extraction  
- Cosine similarity matching  
- Automatic attendance marking  

<img src="docs/screenshots/live_recognition.png" width="800"/>
<br>
<img src="docs/screenshots/result_live_rcognition.png" width="800"/>

---

## 📹 Video Recognition Pipeline

- Upload recorded video  
- Frame-by-frame detection  
- Recognition across stream  
- Batch attendance generation  

<img src="docs/screenshots/video_recognition.png" width="800"/>

<img src="docs/screenshots/result_video.png" width="800"/>

---
## 📊 Dashboard

- Real-time attendance metrics  
- Weekly trend visualization  
- Attendance rate analysis  
- Absentee tracking  

<img src="docs/screenshots/dashboard1.png" width="800"/>

<img src="docs/screenshots/dashboard2.png" width="800"/>

---
# 🧠 Machine Learning Pipeline

### 1️⃣ Face Detection
- MTCNN-based face detection & alignment

### 2️⃣ Face Embedding
- FaceNet architecture
- 128-d embedding vectors
- Implemented with PyTorch + OpenCV

### 3️⃣ Recognition Logic
- Cosine similarity comparison
- Threshold-based validation
- Confidence scoring
- Attendance trigger logic

---

# 🏗 System Architecture

```
    Frontend 
        ↓
FastAPI Backend
        ↓
Recognition Service Layer
        ↓
Face Embedding Engine (PyTorch)
        ↓
SQL Database (SQLAlchemy ORM)
```
---

# 📡 API Endpoints

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

# 🚀 Getting Started

## 1️⃣ Backend

```bash
pip install -r requirements.txt
uvicorn backend.src.api.main:app --reload
```

Backend runs at:
```
http://localhost:8000
```

---

## 2️⃣ Frontend

```bash
npm install
npm run dev
```

Frontend runs at:
```
http://localhost:5173
```

---

# 👤 Author

**Essoufi Hassan**  
Machine Learning & Data Engineer  

---

# ⭐ Note

If you find this project useful or inspiring, consider giving it a ⭐ to support the work.
