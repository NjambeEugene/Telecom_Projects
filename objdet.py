from ultralytics import YOLO

# Use YOLOv8 nano (fast) or small
model = YOLO('yolov8n.pt')

# Train on your 2-class dataset
model.train(data='C:\\Users\\NJAMBE\\Documents\\TRAFFIC\\dataset.yaml', epochs=50, imgsz=416)