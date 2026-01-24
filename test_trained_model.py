import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# Load the trained model
# Find the latest training run
runs_dir = Path("runs/detect/runs/detect")
train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")], 
                    key=lambda x: x.stat().st_mtime, reverse=True)

if train_dirs:
    latest_model = train_dirs[0] / "weights" / "best.pt"
    if latest_model.exists():
        print(f"Loading model: {latest_model}")
        model = YOLO(str(latest_model))
        print(f"Model classes: {model.names}")
    else:
        print(f"Model not found at {latest_model}")
        print("Available weights:")
        for weight_file in (train_dirs[0] / "weights").glob("*.pt"):
            print(f"  - {weight_file}")
        exit()
else:
    print("No training runs found!")
    exit()

# Test on validation images
print("\n" + "="*60)
print("Testing on validation images...")
print("="*60)
val_images_dir = "YOLO V7/images/val"

if os.path.exists(val_images_dir):
    image_files = sorted([f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(image_files)} validation images\n")
    
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    total_detections = 0
    car_white_count = 0
    car_black_count = 0
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(val_images_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  [{i+1}/{len(image_files)}] Failed to read: {img_file}")
            continue
        
        # Run inference
        results = model(img, conf=0.4, verbose=False)
        annotated_img = results[0].plot()
        
        # Save result
        output_path = os.path.join(output_dir, f"detected_{img_file}")
        cv2.imwrite(output_path, annotated_img)
        
        # Count detections
        detections = results[0].boxes
        total_detections += len(detections)
        
        # Print detections
        line = f"  [{i+1}/{len(image_files)}] {img_file}: {len(detections)} detections"
        if len(detections) > 0:
            class_counts = {}
            for box in detections:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                # Track car colors
                if class_name == 'car_white':
                    car_white_count += 1
                elif class_name == 'car_black':
                    car_black_count += 1
            
            print(line)
            for class_name, count in class_counts.items():
                print(f"      → {class_name}: {count}")
        else:
            print(line + " (no cars)")

    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"  - car_white: {car_white_count}")
    print(f"  - car_black: {car_black_count}")
    print(f"\nResults saved to '{output_dir}' folder")
    print("="*60)
else:
    print(f"Validation images directory not found: {val_images_dir}")

# Optional: Test on train images too
print("\n" + "="*60)
print("Testing on training images...")
print("="*60)
train_images_dir = "YOLO V7/images/train"

if os.path.exists(train_images_dir):
    image_files = sorted([f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(image_files)} training images (testing first 5)\n")
    
    for i, img_file in enumerate(image_files[:5]):
        img_path = os.path.join(train_images_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        results = model(img, conf=0.4, verbose=False)
        annotated_img = results[0].plot()
        
        output_path = os.path.join(output_dir, f"train_detected_{img_file}")
        cv2.imwrite(output_path, annotated_img)
        
        detections = results[0].boxes
        print(f"  [{i+1}/5] {img_file}: {len(detections)} detections")
        if len(detections) > 0:
            for box in detections:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"      → {class_name}: {confidence:.1%}")

print("\nTest completed!")

