from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Test on one image
results = model("datasets/garbage/litter.v1i.yolov8/test/images")

print("YOLOv8 working successfully!")