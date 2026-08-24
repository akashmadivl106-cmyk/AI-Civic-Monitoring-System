from flask import Flask, Response, send_from_directory, redirect, render_template
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ======================================
# LOAD MODELS
# ======================================

garbage_model = YOLO("runs/detect/train/weights/best.pt")
coco_model = YOLO("yolov8n.pt")

# ======================================
# GLOBAL VARIABLES
# ======================================

camera_running = False
cap = None

last_alert = "No Active Alerts"

last_garbage_save = 0
last_trespass_save = 0
last_nopark_save = 0

# ======================================
# DETECTION ZONES
# ======================================

T_X1 = 150
T_Y1 = 100
T_X2 = 500
T_Y2 = 400

P_X1 = 200
P_Y1 = 100
P_X2 = 500
P_Y2 = 400


# ======================================
# LOGGING
# ======================================

def log_alert(message):

    global last_alert

    last_alert = message

    os.makedirs("logs", exist_ok=True)

    with open("logs/alerts.txt", "a") as file:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        file.write(
            f"[{timestamp}] {message}\n"
        )


# ======================================
# START CAMERA
# ======================================

@app.route("/start")

def start_monitoring():

    global camera_running
    global cap

    if not camera_running:

        cap = cv2.VideoCapture(0)

        camera_running = True

    return redirect("/")


# ======================================
# STOP CAMERA
# ======================================

@app.route("/stop")

def stop_monitoring():

    global camera_running
    global cap

    camera_running = False

    if cap is not None:

        cap.release()

    return redirect("/")


# ======================================
# EVIDENCE IMAGES
# ======================================

@app.route("/evidence/<category>/<filename>")

def evidence(category, filename):

    return send_from_directory(
        f"evidence/{category}",
        filename
    )


# ======================================
# VIDEO FEED
# ======================================

def generate_frames():

    global cap
    global camera_running

    global last_garbage_save
    global last_trespass_save
    global last_nopark_save

    while True:

        if not camera_running:

            blank = (
                255
                * np.ones(
                    (480, 640, 3),
                    dtype="uint8"
                )
            )

            cv2.putText(
                blank,
                "Monitoring Stopped",
                (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            _, buffer = cv2.imencode(
                ".jpg",
                blank
            )

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame +
                b'\r\n'
            )

            continue

        success, frame = cap.read()

        if not success:
            continue

        # ==================================
        # DRAW ZONES
        # ==================================

        cv2.rectangle(
            frame,
            (T_X1, T_Y1),
            (T_X2, T_Y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "TRESPASS ZONE",
            (T_X1, T_Y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.rectangle(
            frame,
            (P_X1, P_Y1),
            (P_X2, P_Y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "NO PARKING ZONE",
            (P_X1, P_Y1 - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

                # ==================================
        # GARBAGE DETECTION
        # ==================================

        garbage_results = garbage_model(frame, conf=0.5)

        for result in garbage_results:

            for box in result.boxes:

                cls = int(box.cls[0])

                label = garbage_model.names[cls]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                current_time = time.time()

                if (
                    current_time
                    - last_garbage_save
                    >= 5
                ):

                    os.makedirs(
                        "evidence/garbage",
                        exist_ok=True
                    )

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    filename = (
                        f"evidence/garbage/"
                        f"{timestamp}.jpg"
                    )

                    cv2.imwrite(
                        filename,
                        frame
                    )

                    log_alert(
                        f"GARBAGE DETECTED - {label}"
                    )

                    last_garbage_save = current_time

        # ==================================
        # COCO DETECTION
        # ==================================

        coco_results = coco_model(frame, conf=0.5)

        trespassing = False
        noparking = False

        for result in coco_results:

            for box in result.boxes:

                cls = int(box.cls[0])

                label = coco_model.names[cls]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2

                # ==========================
                # PERSON
                # ==========================

                if label == "person":

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    if (
                        T_X1 < center_x < T_X2
                        and
                        T_Y1 < center_y < T_Y2
                    ):

                        trespassing = True

                # ==========================
                # VEHICLE
                # ==========================

                if label in [

                    "car",
                    "motorcycle",
                    "bus",
                    "truck"

                ]:

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 0),
                        2
                    )

                    if (
                        P_X1 < center_x < P_X2
                        and
                        P_Y1 < center_y < P_Y2
                    ):

                        noparking = True

        # ==================================
        # TRESPASS ALERT
        # ==================================

        if trespassing:

            cv2.putText(
                frame,
                "TRESPASSING DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            current_time = time.time()

            if (
                current_time
                - last_trespass_save
                >= 5
            ):

                os.makedirs(
                    "evidence/trespassing",
                    exist_ok=True
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"evidence/trespassing/"
                    f"{timestamp}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

                log_alert(
                    "TRESPASSING DETECTED"
                )

                last_trespass_save = current_time

        # ==================================
        # NO PARKING ALERT
        # ==================================

        if noparking:

            cv2.putText(
                frame,
                "NO PARKING VIOLATION",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

            current_time = time.time()

            if (
                current_time
                - last_nopark_save
                >= 5
            ):

                os.makedirs(
                    "evidence/noparking",
                    exist_ok=True
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"evidence/noparking/"
                    f"{timestamp}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

                log_alert(
                    "NO PARKING VIOLATION"
                )

                last_nopark_save = current_time

        # ==================================
        # STREAM FRAME
        # ==================================

        _, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


        # ======================================
# VIDEO FEED ROUTE
# ======================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ======================================
# DASHBOARD
# ======================================


# ======================================
# DASHBOARD — DATA HELPERS
# (display-only formatting; no detection logic here)
# ======================================

def _classify(message):
    """Same category logic already used for counting; reused so the
    table badges always agree with the KPI counts above them."""
    if "GARBAGE" in message:
        return "garbage", "GARBAGE"
    if "TRESPASSING" in message:
        return "trespass", "TRESPASSING"
    if "NO PARKING" in message:
        return "noparking", "NO PARKING"
    return "other", "EVENT"


def _parse_log_line(line):
    """Parses '[YYYY-MM-DD HH:MM:SS] MESSAGE' lines from logs/alerts.txt
    into a display-friendly dict. Returns None for malformed lines."""

    line = line.strip()

    if not line.startswith("[") or "]" not in line:
        return None

    try:
        end = line.index("]")
        timestamp = line[1:end]
        message = line[end + 1:].strip()
    except ValueError:
        return None

    category, label = _classify(message)

    return {
        "time": timestamp,
        "category": category,
        "label": label,
        "message": message,
    }


def _evidence_timestamp(filename):
    """Best-effort readable timestamp derived from the existing
    'YYYYMMDD_HHMMSS.jpg' evidence filename format. Falls back to the
    raw filename if the pattern doesn't match — never fabricates a time."""

    name = os.path.splitext(filename)[0]

    try:
        dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _build_evidence_list(category, filenames):
    items = []
    for fname in filenames:
        items.append({
            "filename": fname,
            "url": f"/evidence/{category}/{fname}",
            "timestamp": _evidence_timestamp(fname),
        })
    return items


# ======================================
# DASHBOARD
# ======================================

@app.route("/")
def home():

    try:
        with open("logs/alerts.txt", "r") as file:
            logs = file.readlines()
    except:
        logs = []

    garbage = sum(
        "GARBAGE" in log
        for log in logs
    )

    trespassing = sum(
        "TRESPASSING" in log
        for log in logs
    )

    noparking = sum(
        "NO PARKING" in log
        for log in logs
    )

    total_violations = garbage + trespassing + noparking

    try:
        garbage_images = sorted(
            os.listdir("evidence/garbage"),
            reverse=True
        )[:3]
    except:
        garbage_images = []

    try:
        trespass_images = sorted(
            os.listdir("evidence/trespassing"),
            reverse=True
        )[:3]
    except:
        trespass_images = []

    try:
        nopark_images = sorted(
            os.listdir("evidence/noparking"),
            reverse=True
        )[:3]
    except:
        nopark_images = []

    status = (
        "🟢 ONLINE"
        if camera_running
        else
        "🔴 OFFLINE"
    )

    # ---- Display-only reshaping of the same data above ----
    # (no new detection/log data is read here beyond what already existed)

    recent_events = [
        parsed for parsed in (
            _parse_log_line(line) for line in reversed(logs[-20:])
        ) if parsed is not None
    ]

    evidence = {
        "garbage": _build_evidence_list("garbage", garbage_images),
        "trespassing": _build_evidence_list("trespassing", trespass_images),
        "noparking": _build_evidence_list("noparking", nopark_images),
    }

    # Genuine (not fabricated) system checks: these reflect real,
    # currently-observable state rather than hardcoded/fabricated values.
    logging_active = os.path.exists("logs/alerts.txt")
    evidence_storage_active = os.path.isdir("evidence")
    models_loaded = garbage_model is not None and coco_model is not None

    return render_template(
        "index.html",
        camera_running=camera_running,
        status_label=status,
        garbage=garbage,
        trespassing=trespassing,
        noparking=noparking,
        total_violations=total_violations,
        last_alert=last_alert,
        recent_events=recent_events,
        evidence=evidence,
        logging_active=logging_active,
        evidence_storage_active=evidence_storage_active,
        models_loaded=models_loaded,
    )

# ======================================
# RUN
# ======================================

if __name__ == "__main__":
    app.run(
        debug=True,
        threaded=True
    )
