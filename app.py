from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
import sqlite3, os, json, threading, time, random, math
from collections import deque
from werkzeug.security import generate_password_hash, check_password_hash
from create_db import DB_NAME, init_db, get_db
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- Initialize DB ----------------
init_db()

# ---------------- Flask App ----------------
app = Flask(__name__)
app.secret_key = "your_secret_key_2024"
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Models ----------------
yolo_model = YOLO("yolov8n.pt")

# Optimized tracker for stable IDs
tracker = DeepSort(
    max_age=60,          # Keep tracks longer
    n_init=3,            # Require 3 confirmations
    max_iou_distance=0.7, # Stricter matching
    max_cosine_distance=0.4,
    nn_budget=50
)

# ---------------- Shared Data ----------------
zone_counts = {}
population_history = []
activity_points = []
zone_boxes = {}
lock = threading.Lock()
processing_thread = None
stop_processing = False
show_tracking_ids = False
persons_data = []
total_people = 0

# ---------------- Stable ID System with Trajectories ----------------
class StableIDManager:
    def __init__(self):
        self.next_id = 1
        self.id_map = {}  # tracker_id -> stable_id
        self.last_positions = {}  # stable_id -> (x, y)
        self.last_seen = {}  # stable_id -> timestamp
        self.trajectories = {}  # stable_id -> deque of positions (for trails)
        self.max_trajectory_length = 20  # Keep last 20 positions
        
    def get_stable_id(self, tracker_id, current_position, current_time):
        # If we already know this tracker_id
        if tracker_id in self.id_map:
            stable_id = self.id_map[tracker_id]
            self.last_positions[stable_id] = current_position
            self.last_seen[stable_id] = current_time
            
            # Add to trajectory history
            if stable_id not in self.trajectories:
                self.trajectories[stable_id] = deque(maxlen=self.max_trajectory_length)
            self.trajectories[stable_id].append({
                'x': current_position[0],
                'y': current_position[1],
                'time': current_time
            })
                
            return stable_id
            
        # Try to match with recently lost person
        best_match_id = None
        best_distance = 40  # max pixels to match
        
        for sid, last_time in list(self.last_seen.items()):
            # Only check if disappeared recently (last 3 seconds)
            if current_time - last_time > 3.0:
                continue
                
            if sid in self.last_positions:
                last_pos = self.last_positions[sid]
                distance = math.sqrt(
                    (current_position[0] - last_pos[0])**2 + 
                    (current_position[1] - last_pos[1])**2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = sid
        
        # If found match, reuse that ID
        if best_match_id is not None:
            self.id_map[tracker_id] = best_match_id
            self.last_positions[best_match_id] = current_position
            self.last_seen[best_match_id] = current_time
            
            # Add to trajectory history
            if best_match_id not in self.trajectories:
                self.trajectories[best_match_id] = deque(maxlen=self.max_trajectory_length)
            self.trajectories[best_match_id].append({
                'x': current_position[0],
                'y': current_position[1],
                'time': current_time
            })
                
            return best_match_id
            
        # Otherwise, create new stable ID
        stable_id = self.next_id
        self.id_map[tracker_id] = stable_id
        self.next_id += 1
        self.last_positions[stable_id] = current_position
        self.last_seen[stable_id] = current_time
        self.trajectories[stable_id] = deque(maxlen=self.max_trajectory_length)
        self.trajectories[stable_id].append({
            'x': current_position[0],
            'y': current_position[1],
            'time': current_time
        })
        return stable_id
        
    def cleanup_old_ids(self, current_time, max_age=10):
        # Remove IDs not seen for a while
        for sid in list(self.last_seen.keys()):
            if current_time - self.last_seen[sid] > max_age:
                del self.last_seen[sid]
                if sid in self.last_positions:
                    del self.last_positions[sid]
                if sid in self.trajectories:
                    del self.trajectories[sid]
                
        # Clean up old tracker mappings
        for track_id in list(self.id_map.keys()):
            if self.id_map[track_id] not in self.last_seen:
                del self.id_map[track_id]
    
    def get_trajectory(self, stable_id):
        """Get trajectory history for a stable ID"""
        if stable_id in self.trajectories:
            return list(self.trajectories[stable_id])
        return []
    
    def get_velocity(self, stable_id):
        """Calculate velocity from last 2 positions"""
        if stable_id in self.trajectories and len(self.trajectories[stable_id]) >= 2:
            points = list(self.trajectories[stable_id])
            p2 = points[-1]  # Current
            p1 = points[-2]  # Previous
            
            # Calculate time difference
            dt = p2['time'] - p1['time']
            if dt > 0:
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                return [dx/dt, dy/dt]
        return [0, 0]

id_manager = StableIDManager()

# Smoothing for display
_last_positions = {}

# Keep last processed frame size
last_frame_size = (800, 600)

# Canvas size used in frontend
CANVAS_W = 800
CANVAS_H = 600

# ---------------- Alert System ----------------
alert_config = {
    "crowd_threshold": 20,
    "dwell_time_threshold": 300,
    "restricted_zone_alerts": True,
    "email_notifications": False
}

person_dwell_times = {}
active_alerts = []

# ---------------- Helpers ----------------
def _smooth_position(track_id, cx, cy, alpha=0.3):
    now = time.time()
    prev = _last_positions.get(track_id)
    if prev is None:
        _last_positions[track_id] = (cx, cy, now)
        return int(cx), int(cy)
    px, py, pt = prev
    sx = int(px * (1 - alpha) + cx * alpha)
    sy = int(py * (1 - alpha) + cy * alpha)
    _last_positions[track_id] = (sx, sy, now)
    return sx, sy

# ---------------- Video Processing ----------------
def process_video(video_source):
    global zone_counts, population_history, activity_points, stop_processing
    global show_tracking_ids, persons_data, person_dwell_times, active_alerts
    global _last_positions, total_people, last_frame_size, id_manager

    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    
    # Reset for new video
    id_manager = StableIDManager()
    _last_positions.clear()

    while not stop_processing:
        ret, frame = cap.read()
        if not ret:
            if isinstance(video_source, str) and video_source and os.path.exists(video_source):
                cap.release()
                cap = cv2.VideoCapture(video_source)
                continue
            else:
                break

        frame_h, frame_w = frame.shape[:2]
        last_frame_size = (frame_w, frame_h)
        current_time = time.time()

        # ---------------- Prepare scaled zones ----------------
        scaled_zone_boxes = {}
        try:
            sx = frame_w / float(CANVAS_W)
            sy = frame_h / float(CANVAS_H)
            for label, (zx1, zy1, zx2, zy2) in zone_boxes.items():
                zx1_f = int(zx1 * sx)
                zy1_f = int(zy1 * sy)
                zx2_f = int(zx2 * sx)
                zy2_f = int(zy2 * sy)
                scaled_zone_boxes[label] = (zx1_f, zy1_f, zx2_f, zy2_f)
        except Exception:
            scaled_zone_boxes = {label: (zx1, zy1, zx2, zy2) for label, (zx1, zy1, zx2, zy2) in zone_boxes.items()}

        # ---------------- YOLO detection ----------------
        detections = []
        try:
            results = yolo_model(frame, stream=True, classes=[0])
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                xy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), conf, cls in zip(xy, confs, clss):
                    x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
                    w = x2_f - x1_f
                    h = y2_f - y1_f
                    if w <= 0 or h <= 0 or conf < 0.45:  # Higher confidence for stability
                        continue
                    detections.append(([x1_f, y1_f, w, h], float(conf), int(cls)))
        except Exception as e:
            print("YOLO inference error:", e)
            detections = []

        # ---------------- Tracker update ----------------
        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print("Tracker update error:", e)
            tracks = []

        counts = {z: 0 for z in zone_boxes.keys()}
        new_points = []
        current_persons = []

        # Clean old IDs
        id_manager.cleanup_old_ids(current_time)

        for trk in tracks:
            # Only use confirmed tracks
            confirmed = getattr(trk, "is_confirmed", lambda: True)()
            time_since_update = getattr(trk, "time_since_update", 0)
            
            if not confirmed or time_since_update > 2:
                continue

            try:
                ltrb = trk.to_ltrb()
            except Exception:
                continue

            x1, y1, x2, y2 = map(int, ltrb)
            track_id = getattr(trk, "track_id", getattr(trk, "trackId", None))
            if track_id is None:
                continue

            # Calculate center
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get smoothed position for display
            s_cx, s_cy = _smooth_position(track_id, center_x, center_y)
            
            # GET STABLE ID
            stable_id = id_manager.get_stable_id(
                track_id, 
                (s_cx, s_cy), 
                current_time
            )

            # Get trajectory and velocity for trails
            trajectory = id_manager.get_trajectory(stable_id)
            velocity = id_manager.get_velocity(stable_id)

            # Simple age group
            height = y2 - y1
            if height < 100:
                age_group = "child"
            elif height < 160:
                age_group = "young"
            else:
                age_group = "adult"
            
            movement_speed = "normal"

            person_data = {
                "id": stable_id,  # Use stable ID
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": s_cx, "cy": s_cy,
                "age_group": age_group,
                "movement_speed": movement_speed,
                "zone": None,
                "trajectory": trajectory,  # For drawing trails
                "velocity": velocity  # For movement arrows
            }

            # Zone detection
            for label, (zx1_f, zy1_f, zx2_f, zy2_f) in scaled_zone_boxes.items():
                if zx1_f <= s_cx <= zx2_f and zy1_f <= s_cy <= zy2_f:
                    counts[label] = counts.get(label, 0) + 1
                    person_data["zone"] = label

                    # Dwell time tracking
                    dwell_key = f"{stable_id}_{label}"
                    if dwell_key not in person_dwell_times:
                        person_dwell_times[dwell_key] = {"enter_time": current_time, "zone": label}
                    else:
                        dwell_time = current_time - person_dwell_times[dwell_key]["enter_time"]
                        if dwell_time > alert_config["dwell_time_threshold"]:
                            alert_msg = f"Person {stable_id} in zone '{label}' for {int(dwell_time/60)} minutes"
                            if not any(a["message"] == alert_msg for a in active_alerts):
                                active_alerts.append({
                                    "type": "dwell_time",
                                    "message": alert_msg,
                                    "severity": "warning",
                                    "timestamp": current_time
                                })

                    # Activity point
                    new_points.append({
                        "x": s_cx,
                        "y": s_cy,
                        "activity": 10,
                        "age_group": age_group,
                        "movement": movement_speed
                    })

            current_persons.append(person_data)

        # Clean old dwell records
        for dwell_key in list(person_dwell_times.keys()):
            if current_time - person_dwell_times[dwell_key]["enter_time"] > 3600:
                del person_dwell_times[dwell_key]

        total_people_frame = len(current_persons)

        # Crowd density alert
        if total_people_frame > alert_config["crowd_threshold"]:
            alert_msg = f"High crowd density: {total_people_frame} people detected"
            if not any(a["message"] == alert_msg for a in active_alerts):
                active_alerts.append({
                    "type": "crowd",
                    "message": alert_msg,
                    "severity": "danger",
                    "timestamp": current_time
                })

        # Keep recent alerts only
        active_alerts = [a for a in active_alerts if current_time - a["timestamp"] < 300]

        # Thread-safe updates
        with lock:
            for k, v in counts.items():
                zone_counts[k] = v

            total_people = total_people_frame

            population_history.append(total_people)
            if len(population_history) > 50:
                population_history.pop(0)

            activity_points.extend(new_points)
            if len(activity_points) > 50:
                activity_points = activity_points[-50:]

            if show_tracking_ids:
                # Sort by stable ID
                persons_data = sorted(current_persons, key=lambda x: x["id"])
            else:
                persons_data = []

        time.sleep(0.03)
        frame_count += 1

    cap.release()

# ---------------- Routes ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# ----------- Authentication -----------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        email = request.form["email"]
        password = request.form["password"]
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email=?", (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["name"] = user["name"]
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid credentials","danger")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password", "")
        
        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return render_template("register.html")
            
        hashed_pw = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)",
                               (name,email,hashed_pw))
                conn.commit()
            flash("Registration successful! Please login.","success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists","danger")
    return render_template("register.html")

@app.route("/logout")
def logout():
    global stop_processing
    stop_processing = True
    if processing_thread:
        processing_thread.join()
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

# ----------- Dashboard -----------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Get user stats
    conn = get_db()
    cursor = conn.cursor()
    
    # Get camera count
    cursor.execute("SELECT COUNT(*) as count FROM cameras WHERE user_id=?", (session["user_id"],))
    camera_count = cursor.fetchone()["count"]
    
    # Get zone count
    cursor.execute("SELECT COUNT(*) as count FROM zones WHERE user_id=?", (session["user_id"],))
    zone_count = cursor.fetchone()["count"]
    
    conn.close()
    
    # Get active alerts count
    with lock:
        alert_count = len(active_alerts)
    
    return render_template("dashboard.html", 
                         camera_count=camera_count,
                         zone_count=zone_count,
                         alert_count=alert_count,
                         people_count=total_people)

# ----------- Camera Management -----------
@app.route("/cameras")
def cameras():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras WHERE user_id=?", (session["user_id"],))
    user_cameras = cursor.fetchall()
    conn.close()
    
    return render_template("cameras.html", cameras=user_cameras)

@app.route("/add_camera", methods=["POST"])
def add_camera():
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    name = request.form.get("name")
    source_type = request.form.get("source_type")
    source_value = request.form.get("source_value")
    
    if source_type == "webcam":
        source_value = "0"
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO cameras (user_id, name, source_type, source_value, status) VALUES (?, ?, ?, ?, 'stopped')",
            (session["user_id"], name, source_type, source_value)
        )
        conn.commit()
        conn.close()
        flash("Camera added successfully!", "success")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/delete_camera/<int:camera_id>", methods=["POST"])
def delete_camera(camera_id):
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras WHERE id=? AND user_id=?", (camera_id, session["user_id"]))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/start_camera_management/<int:camera_id>")
def start_camera_management(camera_id):
    global processing_thread, stop_processing
    
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    # Stop current processing
    stop_processing = True
    if processing_thread:
        processing_thread.join()
    
    # Get camera details
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras WHERE id=? AND user_id=?", (camera_id, session["user_id"]))
    camera = cursor.fetchone()
    
    if not camera:
        conn.close()
        return jsonify({"success": False, "error": "Camera not found"})
    
    # Update camera status to running
    cursor.execute("UPDATE cameras SET status='running' WHERE id=?", (camera_id,))
    conn.commit()
    conn.close()
    
    # Start processing with new camera
    stop_processing = False
    video_source = 0 if camera["source_type"] == "webcam" else camera["source_value"]
    
    processing_thread = threading.Thread(target=process_video, args=(video_source,), daemon=True)
    processing_thread.start()
    
    # Update session
    session["current_camera_id"] = camera_id
    session["current_camera_name"] = camera["name"]
    session["use_camera"] = True
    session.pop("video_file", None)
    
    return jsonify({"success": True, "camera_name": camera["name"], "status": "running"})

@app.route("/stop_camera")
def stop_camera():
    global processing_thread, stop_processing
    
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    # Stop processing
    stop_processing = True
    if processing_thread:
        processing_thread.join()
    
    # Update camera status to stopped
    current_camera_id = session.get("current_camera_id")
    if current_camera_id:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE cameras SET status='stopped' WHERE id=?", (current_camera_id,))
        conn.commit()
        conn.close()
    
    # Clear session
    session.pop("current_camera_id", None)
    session.pop("current_camera_name", None)
    session.pop("use_camera", None)
    
    return jsonify({"success": True, "status": "stopped"})

@app.route("/camera_status")
def camera_status():
    if "user_id" not in session:
        return jsonify({"status": "stopped"})
    
    current_camera_id = session.get("current_camera_id")
    if current_camera_id:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM cameras WHERE id=?", (current_camera_id,))
        camera = cursor.fetchone()
        conn.close()
        
        if camera:
            return jsonify({
                "status": camera["status"],
                "camera_id": current_camera_id,
                "camera_name": session.get("current_camera_name", "Unknown Camera")
            })
    
    return jsonify({"status": "stopped"})

# ----------- Alert System -----------
@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        # Handle form submission
        alert_config.update({
            "crowd_threshold": int(request.form.get("crowd_threshold", 20)),
            "dwell_time_threshold": int(request.form.get("dwell_time_threshold", 300)),
            "restricted_zone_alerts": request.form.get("restricted_zone_alerts") == "on",
            "email_notifications": request.form.get("email_notifications") == "on"
        })
        flash("Alert settings updated successfully!", "success")
        return redirect(url_for("alerts"))
    
    return render_template("alerts.html", config=alert_config)
@app.route("/api/alerts")
def api_alerts():
    if "user_id" not in session:
        return jsonify({"alerts": []})
    
    with lock:
        return jsonify({"alerts": active_alerts[-10:]})

# ----------- Zones -----------
@app.route("/zones", methods=["GET","POST"])
def zones():
    global zone_boxes, zone_counts
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    conn = get_db()
    cursor = conn.cursor()
    
    if request.method=="POST":
        zones_data = request.form.get("zonesData")
        if zones_data:
            try:
                zones_json = json.loads(zones_data)
                zone_boxes = {}
                cursor.execute("DELETE FROM zones WHERE user_id=?", (session["user_id"],))
                for z in zones_json:
                    cursor.execute(
                        "INSERT INTO zones (user_id,label,x1,y1,x2,y2) VALUES (?,?,?,?,?,?)",
                        (session["user_id"], z["label"], z["x"], z["y"], z["x"]+z["w"], z["y"]+z["h"])
                    )
                    zone_boxes[z["label"]] = (z["x"], z["y"], z["x"]+z["w"], z["y"]+z["h"])
                    if z["label"] not in zone_counts:
                        zone_counts[z["label"]] = 0
                conn.commit()
                flash("Zones saved successfully!", "success")
            except Exception as e:
                print("Error saving zones:", e)
                flash("Error saving zones", "danger")
    
    cursor.execute("SELECT id,label,x1,y1,x2,y2 FROM zones WHERE user_id=?", (session["user_id"],))
    zones = cursor.fetchall()
    conn.close()
    
    # Load zone boxes from database if not already loaded
    if not zone_boxes and zones:
        for zone in zones:
            zone_boxes[zone["label"]] = (zone["x1"], zone["y1"], zone["x2"], zone["y2"])
            if zone["label"] not in zone_counts:
                zone_counts[zone["label"]] = 0
    
    video_file = session.get("video_file")
    use_camera = session.get("use_camera", False)
    
    return render_template("zones.html", zones=zones, video_file=video_file, use_camera=use_camera)

@app.route("/toggle_tracking")
def toggle_tracking():
    global show_tracking_ids
    value = request.args.get("id", "0")
    show_tracking_ids = value == "1"
    return jsonify({"success": True, "show_tracking": show_tracking_ids})

# ----------- Live Dashboard -----------
@app.route("/live_dashboard")
def live_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("live_dashboard.html")

# ----------- API Endpoints -----------
@app.route("/api/zone_data")
def api_zone_data():
    if "user_id" not in session:
        return jsonify({"zones": [], "population": [], "peak": 0, "heatmap": [], "total": 0, "active_zones": 0})
    
    with lock:
        # Get user's zones from database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT label FROM zones WHERE user_id=?", (session["user_id"],))
        user_zones = [row["label"] for row in cursor.fetchall()]
        conn.close()

        zones_list = [{"label": l, "count": zone_counts.get(l, 0)} for l in user_zones]
        total_population = total_people if total_people is not None else sum(zone_counts.values())
        active_zones_count = sum(1 for v in zone_counts.values() if v > 0)

        # Scale activity_points to CANVAS coords
        frame_w, frame_h = last_frame_size
        heatmap_points = []
        if activity_points:
            for p in activity_points[-20:]:
                try:
                    x_frame = p.get("x", 0)
                    y_frame = p.get("y", 0)
                    x_canvas = int(x_frame * (CANVAS_W / float(frame_w))) if frame_w else int(x_frame)
                    y_canvas = int(y_frame * (CANVAS_H / float(frame_h))) if frame_h else int(y_frame)
                    heatmap_points.append({
                        "x": x_canvas,
                        "y": y_canvas,
                        "activity": p.get("activity", 5),
                        "age_group": p.get("age_group"),
                        "movement": p.get("movement")
                    })
                except Exception:
                    continue

        pop_hist = [{"time": idx, "total": val} for idx, val in enumerate(population_history[-15:])] if population_history else []

        return jsonify({
            "zones": zones_list,
            "population": pop_hist,
            "peak": max(population_history) if population_history else 0,
            "heatmap": heatmap_points,
            "total": total_population,
            "active_zones": active_zones_count
        })

@app.route("/api/tracking_data")
def api_tracking_data():
    with lock:
        if show_tracking_ids:
            return jsonify({"persons": persons_data})
        else:
            return jsonify({"persons": []})

# ----------- Video Upload -----------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global processing_thread, stop_processing
    
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    if request.method == "GET":
        return render_template("upload.html")
    
    if request.method == "POST":
        stop_processing = True
        if processing_thread: 
            processing_thread.join()
        
        if "video" not in request.files:
            flash("No file selected!", "danger")
            return redirect(url_for("dashboard"))
        
        file = request.files["video"]
        if file.filename == "":
            flash("No file selected!", "danger")
            return redirect(url_for("dashboard"))
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        session["video_file"] = file.filename
        session["use_camera"] = False
        
        stop_processing = False
        processing_thread = threading.Thread(target=process_video, args=(filepath,), daemon=True)
        processing_thread.start()
        
        flash("Video uploaded and processing started!", "success")
        return redirect(url_for("zones"))

@app.route("/start_camera")
def start_camera():
    global processing_thread, stop_processing
    
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    stop_processing = True
    if processing_thread:
        processing_thread.join()

    session.pop("video_file", None)
    session["use_camera"] = True

    # Update default camera status if it exists in database
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM cameras WHERE user_id=? AND source_type='webcam' LIMIT 1", (session["user_id"],))
    webcam = cursor.fetchone()

    if webcam:
        cursor.execute("UPDATE cameras SET status='running' WHERE id=?", (webcam["id"],))
        session["current_camera_id"] = webcam["id"]
        session["current_camera_name"] = "Default Webcam"
    conn.commit()
    conn.close()

    stop_processing = False
    processing_thread = threading.Thread(target=process_video, args=(0,), daemon=True)
    processing_thread.start()
    
    flash("Webcam started successfully!", "success")
    return redirect(url_for("zones"))

# ----------- Static Files -----------
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ----------- Error Handlers -----------
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500

if __name__=="__main__":
    print("=" * 50)
    print("People Counter Application")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("Login with your credentials")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    
    app.run(debug=True, host="0.0.0.0", port=5000)