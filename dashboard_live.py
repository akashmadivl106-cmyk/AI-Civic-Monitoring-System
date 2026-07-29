from flask import Flask, Response, send_from_directory
from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

app = Flask(__name__)

# ==========================
# LOAD MODELS
# ==========================

garbage_model = YOLO("runs/detect/train/weights/best.pt")
coco_model = YOLO("yolov8n.pt")

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

last_garbage_save = 0
last_trespass_save = 0
last_nopark_save = 0

# ==========================
# ZONES
# ==========================

T_X1, T_Y1 = 150, 100
T_X2, T_Y2 = 500, 400

P_X1, P_Y1 = 200, 100
P_X2, P_Y2 = 500, 400

# ==========================
# LOG ALERT
# ==========================

def log_alert(message):

    os.makedirs("logs", exist_ok=True)

    with open("logs/alerts.txt", "a") as file:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        file.write(
            f"[{timestamp}] {message}\n"
        )

# ==========================
# EVIDENCE ROUTE
# ==========================

@app.route("/evidence/<category>/<filename>")
def evidence(category, filename):

    return send_from_directory(
        f"evidence/{category}",
        filename
    )

# ==========================
# LIVE VIDEO ROUTE
# ==========================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ==========================
# LIVE DETECTION ENGINE
# ==========================

def generate_frames():

    global last_garbage_save
    global last_trespass_save
    global last_nopark_save

    while True:

        success, frame = cap.read()

        if not success:
            break

        # --------------------------
        # DRAW ZONES
        # --------------------------

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
            (P_X1, P_Y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # --------------------------
        # GARBAGE DETECTION
        # --------------------------

        garbage_results = garbage_model(frame)

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

        # --------------------------
        # PERSON + VEHICLE DETECTION
        # --------------------------

        coco_results = coco_model(frame)

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

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # PERSON

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

                # VEHICLES

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

        # --------------------------
        # ALERTS
        # --------------------------

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

        # --------------------------
        # STREAM FRAME
        # --------------------------

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


# ==========================
# DASHBOARD HOME PAGE
# ==========================

@app.route("/")
def home():

    try:
        with open("logs/alerts.txt", "r") as file:
            logs = file.readlines()
    except:
        logs = []

    garbage = sum("GARBAGE" in log for log in logs)
    trespassing_count = sum("TRESPASSING" in log for log in logs)
    noparking_count = sum("NO PARKING" in log for log in logs)

    html = f"""
    <!DOCTYPE html>
    <html>

    <head>

    <meta http-equiv="refresh" content="10">

    <title>AI Civic Monitoring System</title>

    <style>

    body {{
        margin:0;
        font-family: Arial;
        background:#0f172a;
        color:white;
    }}

    .banner {{
        background:#1e293b;
        padding:25px;
        text-align:center;
        box-shadow:0 2px 10px black;
    }}

    .banner h1 {{
        margin:0;
        color:#38bdf8;
    }}

    .banner p {{
        margin-top:10px;
        color:#cbd5e1;
    }}

    .live {{
        text-align:center;
        margin-top:20px;
    }}

    .live img {{
        width:80%;
        border-radius:15px;
        border:4px solid #38bdf8;
    }}

    .cards {{
        display:flex;
        justify-content:center;
        gap:20px;
        margin-top:30px;
        flex-wrap:wrap;
    }}

    .card {{
        width:250px;
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
    }}

    .g {{
        background:#f59e0b;
    }}

    .t {{
        background:#dc2626;
    }}

    .n {{
        background:#2563eb;
    }}

    .section {{
        background:#1e293b;
        margin:20px;
        padding:20px;
        border-radius:15px;
    }}

    pre {{
        color:white;
        font-size:14px;
    }}

    footer {{
        text-align:center;
        padding:20px;
        color:#94a3b8;
    }}

    </style>

    </head>

    <body>

    <div class="banner">

        <h1>🏙 AI Civic Monitoring System</h1>

        <p>
        Smart Surveillance using YOLOv8
        </p>

    </div>

    <div class="live">

        <h2>📹 Live Monitoring Feed</h2>

        <img src="/video_feed">

    </div>

    <div class="cards">

        <div class="card g">
            Garbage Alerts
            <br><br>
            {garbage}
        </div>

        <div class="card t">
            Trespassing Alerts
            <br><br>
            {trespassing_count}
        </div>

        <div class="card n">
            No Parking Alerts
            <br><br>
            {noparking_count}
        </div>

    </div>

    <div class="section">

        <h2>📋 Recent Alerts</h2>

        <pre>
{''.join(logs[-20:])}
        </pre>

    </div>

    <footer>

        AI Civic Monitoring System<br>
        Major Project 2025-26

    </footer>

    </body>
    </html>
    """

    return html


# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=True)