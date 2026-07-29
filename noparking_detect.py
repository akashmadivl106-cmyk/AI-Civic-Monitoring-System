from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

def log_alert(message):
    os.makedirs("logs", exist_ok=True)

    with open("logs/alerts.txt", "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] {message}\n")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
last_saved_time = 0

# No-parking zone coordinates
ZONE_X1, ZONE_Y1 = 200, 100
ZONE_X2, ZONE_Y2 = 500, 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw no-parking zone
    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 0, 255), 2)
    cv2.putText(frame, "NO PARKING ZONE",
                (ZONE_X1, ZONE_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    violation = False

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Detect vehicles
            if label in ["car", "motorcycle", "bus", "truck"]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if (ZONE_X1 < center_x < ZONE_X2 and
                    ZONE_Y1 < center_y < ZONE_Y2):
                    violation = True

    if violation:
        cv2.putText(frame,
                    "NO PARKING VIOLATION!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)
        
        #evidance store man
        current_time = time.time()

        if current_time - last_saved_time >= 5:

            os.makedirs("evidence/noparking", exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"evidence/noparking/{timestamp}.jpg"

            cv2.imwrite(filename, frame)

            log_alert("NO PARKING VIOLATION")

            print(f"Evidence Saved: {filename}")

            last_saved_time = current_time

    cv2.imshow("No Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()