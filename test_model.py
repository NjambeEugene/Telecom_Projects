import os
from ultralytics import YOLO
from pathlib import Path
import cv2

# Load the trained model
model_path = r"runs/detect/runs/detect/train/weights/best.pt"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print("Make sure training completed successfully!")
    exit(1)

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Test on validation images
val_images_dir = r"YOLO/images/val"
print(f"\nTesting on validation images from: {val_images_dir}")

# Get list of validation images
val_images = list(Path(val_images_dir).glob("*.jpg")) + list(Path(val_images_dir).glob("*.png"))
print(f"Found {len(val_images)} validation images")

if len(val_images) > 0:
    # Test on first few images
    for img_path in val_images[:5]:
        print(f"\nTesting: {img_path.name}")
        
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=0.5,
            save=True,
            device='cpu'
        )
        
        # Print detections
        for result in results:
            if result.boxes:
                print(f"  Detections found: {len(result.boxes)}")
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"    - {class_name}: {confidence:.2f}")
            else:
                print("  No detections")

# Evaluate on full validation set
print("\n" + "="*50)
print("Running full validation evaluation...")
print("="*50)

results = model.val(
    data=r"YOLO/data.yaml",
    device='cpu'
)

print("\nValidation Results:")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

# Optional: Test on custom image
print("\n" + "="*50)
print("To test on a custom image, run:")
print("  python test_model.py --image path/to/image.jpg")
print("="*50)
