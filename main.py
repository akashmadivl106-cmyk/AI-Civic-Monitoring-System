from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

# ==========================
# LOAD MODELS
# ==========================

garbage_model = YOLO("runs/detect/train/weights/best.pt")
coco_model = YOLO("yolov8n.pt")

# ==========================
# ALERT LOG
# ==========================

def log_alert(message):
    os.makedirs("logs", exist_ok=True)

    with open("logs/alerts.txt", "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] {message}\n")

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

# Trespassing Zone
T_X1, T_Y1 = 150, 100
T_X2, T_Y2 = 500, 400

# No Parking Zone
P_X1, P_Y1 = 200, 100
P_X2, P_Y2 = 500, 400



while True:

    success, frame = cap.read()

    if not success:
        break

    # ==========================
    # DRAW ZONES
    # ==========================

    cv2.rectangle(frame, (T_X1, T_Y1), (T_X2, T_Y2), (0, 0, 255), 2)
    cv2.putText(frame, "TRESPASS ZONE",
                (T_X1, T_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    cv2.rectangle(frame, (P_X1, P_Y1), (P_X2, P_Y2), (255, 0, 0), 2)
    cv2.putText(frame, "NO PARKING ZONE",
                (P_X1, P_Y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)

    # ==========================
    # GARBAGE DETECTION
    # ==========================

    garbage_results = garbage_model(frame)

    for result in garbage_results:
        for box in result.boxes:

            cls = int(box.cls[0])
            label = garbage_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 255), 2)

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

            current_time = time.time()

            if current_time - last_garbage_save >= 5:

                os.makedirs("evidence/garbage", exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = f"evidence/garbage/{timestamp}.jpg"

                cv2.imwrite(filename, frame)

                log_alert(f"GARBAGE DETECTED - {label}")

                last_garbage_save = current_time


                    # ==========================
    # PERSON + VEHICLE DETECTION
    # ==========================

    coco_results = coco_model(frame)

    trespassing = False
    noparking = False

    for result in coco_results:
        for box in result.boxes:

            cls = int(box.cls[0])
            label = coco_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # --------------------------
            # TRESPASSING
            # --------------------------

            if label == "person":

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0), 2)

                if (T_X1 < center_x < T_X2 and
                    T_Y1 < center_y < T_Y2):

                    trespassing = True

            # --------------------------
            # NO PARKING
            # --------------------------

            if label in ["car", "motorcycle", "bus", "truck"]:

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (255, 255, 0), 2)

                if (P_X1 < center_x < P_X2 and
                    P_Y1 < center_y < P_Y2):

                    noparking = True

    # ==========================
    # TRESPASS ALERT
    # ==========================

    if trespassing:

        cv2.putText(frame,
                    "TRESPASSING DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

        current_time = time.time()

        if current_time - last_trespass_save >= 5:

            os.makedirs("evidence/trespassing", exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"evidence/trespassing/{timestamp}.jpg"

            cv2.imwrite(filename, frame)

            log_alert("TRESPASSING DETECTED")

            last_trespass_save = current_time

    # ==========================
    # NO PARKING ALERT
    # ==========================

    if noparking:

        cv2.putText(frame,
                    "NO PARKING VIOLATION",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3)

        current_time = time.time()

        if current_time - last_nopark_save >= 5:

            os.makedirs("evidence/noparking", exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"evidence/noparking/{timestamp}.jpg"

            cv2.imwrite(filename, frame)

            log_alert("NO PARKING VIOLATION")

            last_nopark_save = current_time

    # ==========================
    # DISPLAY
    # ==========================

    cv2.imshow("AI Civic Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# CLOSE
# ==========================

cap.release()
cv2.destroyAllWindows()