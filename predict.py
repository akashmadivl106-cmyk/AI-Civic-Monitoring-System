from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on test images
results = model.predict(
    source="datasets/garbage/litter.v1i.yolov8/test/images",
    save=True
)

print("Prediction completed!")