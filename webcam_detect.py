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

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Start webcam
cap = cv2.VideoCapture(0)
last_saved_time = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    # Run YOLO detection
    results = model(frame,cong=0.5)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    #evidance storage man
    current_time = time.time()

    if current_time - last_saved_time >= 5:

        os.makedirs("evidence/garbage", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"evidence/garbage/{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        log_alert("GARBAGE DETECTED")

        print(f"Garbage Evidence Saved: {filename}")

        last_saved_time = current_time

    # Show output
    cv2.imshow("AI Civic Monitoring - Real Time Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()