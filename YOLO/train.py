import os
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Define data path
data_path = os.path.join(os.getcwd(), "YOLO V7", "data.yaml")
print(f"Using dataset: {data_path}")
print(f"Dataset exists: {os.path.exists(data_path)}")

# Train model
results = model.train(
    data=data_path,
    epochs=25,
    imgsz=640,
    batch=8,
    patience=10,
    device='cpu',
    workers=2,
    project="runs/detect",
    name="train_v2_corrected",
    exist_ok=False,
    verbose=True
)

print("Training completed!")
print(f"Results saved to: {results.save_dir}")
