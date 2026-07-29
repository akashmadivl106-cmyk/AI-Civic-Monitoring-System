from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="datasets/garbage/litter.v1i.yolov8/data.yaml",
    epochs=10,
    imgsz=640
)