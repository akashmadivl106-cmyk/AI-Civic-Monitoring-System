from flask import Flask, Response, send_from_directory, redirect
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

    html = f"""
<!DOCTYPE html>
<html>

<head>

<meta http-equiv="refresh" content="5">

<title>AI Civic Monitoring System</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body {{
background:#071633;
color:white;
font-family:Arial;
margin:0;
}}

.header {{
background:#182842;
padding:30px;
text-align:center;
box-shadow:0 0 20px black;
}}

.header h1 {{
font-size:55px;
margin:0;
color:#3eb3ff;
}}

.header p {{
font-size:24px;
}}

.section {{
padding:30px;
}}

.live {{
text-align:center;
}}

.live img{{
    width:50%;
    max-width:1000px;
    height:auto;
    border-radius:20px;
    border:4px solid #3eb3ff;
}}

.buttons {{
text-align:center;
margin-top:20px;
}}

.btn {{
padding:15px 30px;
font-size:20px;
border:none;
border-radius:10px;
cursor:pointer;
margin:10px;
}}

.start {{
background:#00c853;
color:white;
}}

.stop {{
background:#ff1744;
color:white;
}}

.cards {{
display:flex;
justify-content:center;
gap:30px;
flex-wrap:wrap;
}}

.card {{
width:280px;
padding:25px;
text-align:center;
border-radius:15px;
font-size:30px;
font-weight:bold;
}}

.g {{
background:orange;
}}

.t {{
background:crimson;
}}

.n {{
background:royalblue;
}}

.panel {{
background:#1a2b4a;
padding:25px;
margin-top:25px;
border-radius:15px;
}}

.alertbox {{
background:#b71c1c;
padding:20px;
border-radius:15px;
font-size:25px;
font-weight:bold;
}}

.gallery img {{
width:260px;
border-radius:10px;
margin:10px;
}}

.footer {{
margin-top:40px;
padding:25px;
text-align:center;
background:#182842;
}}

</style>

</head>

<body>

<div class="header">

<h1>🏙 AI Civic Monitoring System</h1>

<p>Smart City Surveillance Platform</p>

</div>

<div class="section live">

<h2>📷 Live Monitoring Feed</h2>

<img src="/video_feed">

<div class="buttons">

<a href="/start">
<button class="btn start">
▶ Start Monitoring
</button>
</a>

<a href="/stop">
<button class="btn stop">
■ Stop Monitoring
</button>
</a>

</div>

</div>

<div class="section">

<div class="cards">

<div class="card g">
Garbage Alerts
<br><br>
{garbage}
</div>

<div class="card t">
Trespassing Alerts
<br><br>
{trespassing}
</div>

<div class="card n">
No Parking Alerts
<br><br>
{noparking}
</div>

</div>

<div class="panel">

<h2>🚨 Active Alert</h2>

<div class="alertbox">
{last_alert}
</div>

</div>

<div class="panel">

<h2>⚙ System Status</h2>

<p>Camera Status : {status}</p>

<p>YOLO Status : 🟢 RUNNING</p>

<p>Server Status : 🟢 ACTIVE</p>

</div>



<div class="panel">

<h2>📊 Analytics Dashboard</h2>

<div style="
display:flex;
justify-content:space-between;
align-items:stretch;
gap:20px;
">

<div style="
flex:1;
height:400px;
background:#13284d;
padding:20px;
border-radius:15px;
">
<canvas id="barChart"></canvas>
</div>

<div style="
flex:1;
height:400px;
background:#13284d;
padding:20px;
border-radius:15px;
">
<canvas id="pieChart"></canvas>
</div>

</div>

</div>


<div class="panel">

<h2>📋 Recent Alerts</h2>

<pre>
{''.join(logs[-20:])}
</pre>

</div>

<div class="panel gallery">

<h2>🖼 Latest Evidence</h2>

<h3>Garbage</h3>

"""

    for img in garbage_images:

        html += f"""
<img src="/evidence/garbage/{img}">
"""

    html += "<h3>Trespassing</h3>"

    for img in trespass_images:

        html += f"""
<img src="/evidence/trespassing/{img}">
"""

    html += "<h3>No Parking</h3>"

    for img in nopark_images:

        html += f"""
<img src="/evidence/noparking/{img}">
"""

    html += f"""

</div>

</div>

<div class="footer">

<h2>AI Civic Monitoring System</h2>

<p>Major Project 2025-26</p>

</div>

<script>

new Chart(
document.getElementById('barChart'),
{{
type:'bar',
data:{{
labels:[
'Garbage',
'Trespassing',
'No Parking'
],
datasets:[{{
label:'Violations',
data:[
{garbage},
{trespassing},
{noparking}
],
backgroundColor:[
'orange',
'crimson',
'royalblue'
]
}}]
}},
options:{{
responsive:true,
maintainAspectRatio:false
}}
}}
);

new Chart(
document.getElementById('pieChart'),
{{
type:'pie',
data:{{
labels:[
'Garbage',
'Trespassing',
'No Parking'
],
datasets:[{{
data:[
{garbage},
{trespassing},
{noparking}
],
backgroundColor:[
'orange',
'crimson',
'royalblue'
]
}}]
}},
options:{{
responsive:true,
maintainAspectRatio:false
}}
}}
);

</script>

</body>
</html>
"""

    return html


# ======================================
# RUN
# ======================================

if __name__ == "__main__":
    app.run(
        debug=True,
        threaded=True
    )