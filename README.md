# 🧍‍♂️ Crowd Monitoring & People Counting System

A real-time **AI-based crowd monitoring system** built using **Python, Flask, YOLOv8, and DeepSORT**.
The application detects, tracks, and counts people from **live webcam feeds or uploaded videos**, supports **zone-based monitoring**, and generates **alerts** for crowd density and dwell time.

---

## 🚀 Features

* 🔍 Real-time **person detection** using YOLOv8
* 🆔 **Stable person tracking** with DeepSORT
* 📊 **Live people counting** and population history
* 📦 **Zone-based monitoring** (manual zone drawing)
* ⏱️ **Dwell time tracking** per zone
* 🚨 **Crowd & dwell-time alerts**
* 🎥 Supports **webcam** and **video upload**
* 👤 **User authentication** (login & registration)
* 🗄️ SQLite database for users, cameras, and zones
* 📈 Live dashboard with heatmap & analytics

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **AI / ML:** YOLOv8, DeepSORT
* **Computer Vision:** OpenCV
* **Database:** SQLite
* **Frontend:** HTML, CSS, Bootstrap, JavaScript

---

## 📂 Project Structure

```
project/
│── app.py
│── create_db.py
│── database.db
│── templates/
│── static/
│   └── uploads/
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

### 5️⃣ Open in browser

```
http://localhost:5000
```

---

## 🧪 How It Works (Simple Explanation)

1. User logs in or registers
2. Selects **webcam** or uploads a **video file**
3. YOLOv8 detects people in each frame
4. DeepSORT tracks people with stable IDs
5. Zones are drawn manually on the video
6. System counts people per zone
7. Alerts are generated if thresholds are exceeded

---

## 🚨 Alert Types

* High crowd density alert
* Long dwell time in a zone
* Restricted zone monitoring

---

## 📌 Use Cases

* Shopping malls
* Railway stations
* Airports
* Colleges & campuses
* Public events & gatherings

---

## 🔮 Future Enhancements

* Email / SMS alerts
* Cloud database support
* Role-based access
* Multi-camera dashboard
* Deployment with Docker

---

## 👨‍💻 Author

**Sanjivanee Jarhad**
MCA Student | Python & AI Developer

---

## 📜 License

This project is for **educational and learning purposes*

Just tell me 👌
