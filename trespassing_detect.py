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

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
last_saved_time = 0

# Restricted Zone Coordinates
ZONE_X1 = 150
ZONE_Y1 = 100
ZONE_X2 = 500
ZONE_Y2 = 400

while True:
    success, frame = cap.read()

    if not success:
        break

    # Draw restricted area
    cv2.rectangle(
        frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        2
    )

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw person box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Check if person is inside zone
                if (
                    ZONE_X1 < center_x < ZONE_X2
                    and ZONE_Y1 < center_y < ZONE_Y2
                ):

                    cv2.putText(
                        frame,
                        "TRESPASSING DETECTED!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    # Save evidence image
                    current_time = time.time()

                    if current_time - last_saved_time >= 5:

                        os.makedirs("evidence/trespassing", exist_ok=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        filename = f"evidence/trespassing/{timestamp}.jpg"

                        cv2.imwrite(filename, frame)

                        log_alert("TRESPASSING DETECTED")

                        print(f"Evidence Saved: {filename}")

                        last_saved_time = current_time

    cv2.imshow("Trespassing Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()