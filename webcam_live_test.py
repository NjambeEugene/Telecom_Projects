import cv2
from ultralytics import YOLO
from pathlib import Path
import os

# Load the trained model
runs_dir = Path("runs/detect/runs/detect")
train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")], 
                    key=lambda x: x.stat().st_mtime, reverse=True)

if train_dirs:
    latest_model = train_dirs[0] / "weights" / "best.pt"
    if latest_model.exists():
        print(f"Loading model: {latest_model}")
        model = YOLO(str(latest_model))
        print(f"Model loaded successfully!")
        print(f"Classes: {list(model.names.values())}")
    else:
        print("Model not found!")
        exit()
else:
    print("No training runs found!")
    exit()

# Open webcam
print("\nStarting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Create output directory for saved frames
output_dir = "webcam_results"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
saved_count = 0
car_detections = 0

# Frame smoothing for stable counts
smoothing_window = 5  # Average over 5 frames
detection_history = []
car_count_history = []

print("Webcam started! Press:")
print("  'q' to quit")
print("  's' to save current frame")
print("  'c' to clear saved images")
print(f"  Detection smoothing: {smoothing_window} frames")
print("\n" + "="*60)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to read frame")
        break
    
    frame_count += 1
    
    # Run inference with lower confidence for better detection
    results = model(frame, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()
    
    # Get detections
    detections = results[0].boxes
    
    # Count car detections
    car_count = 0
    for box in detections:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if 'car' in class_name.lower():
            car_count += 1
    
    # Add to history for smoothing
    detection_history.append(len(detections))
    car_count_history.append(car_count)
    
    # Keep only recent frames in history
    if len(detection_history) > smoothing_window:
        detection_history.pop(0)
    if len(car_count_history) > smoothing_window:
        car_count_history.pop(0)
    
    # Calculate smoothed (averaged) counts
    stable_detection_count = int(sum(detection_history) / len(detection_history))
    stable_car_count = int(sum(car_count_history) / len(car_count_history))
    
    if car_count > 0:
        car_detections += car_count
    
    # Add information overlay with stable counts
    info_text = f"Frame: {frame_count} | Det: {len(detections)} (Avg: {stable_detection_count}) | Cars: {car_count} (Avg: {stable_car_count})"
    cv2.putText(annotated_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add class info if objects detected
    if len(detections) > 0:
        class_counts = {}
        y_offset = 60
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        for class_name, count in class_counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
    
    # Add save status
    cv2.putText(annotated_frame, f"Saved: {saved_count}", (10, annotated_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow("Car Detection - Live Webcam (Press 'q' to quit, 's' to save)", annotated_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        filename = os.path.join(output_dir, f"webcam_frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, annotated_frame)
        print(f"Saved: {filename}")
        saved_count += 1
    elif key == ord('c'):
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            saved_count = 0
            print("Cleared saved images")

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("WEBCAM TEST SUMMARY")
print("="*60)
print(f"Total frames processed: {frame_count}")
print(f"Total car detections (cumulative): {car_detections}")
print(f"Average detections per frame: {car_detections/frame_count:.2f}" if frame_count > 0 else "No frames")
print(f"Frames saved: {saved_count}")
print(f"Results saved to: {output_dir}")
print(f"Smoothing window: {smoothing_window} frames")
print("="*60)
