import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import requests
import json

# ESP32 Configuration
ESP32_IP = "192.168.137.85"  # Change this to your ESP32's IP address
ESP32_PORT = 80
ESP32_URL = f"http://{ESP32_IP}:{ESP32_PORT}/update"

# Emergency vehicle class names (adjust based on your YOLO model)
EMERGENCY_CLASSES = ['ambulance', 'fire truck', 'police car', 'emergency']

# Find all weight files
weight_files = []
for root, dirs, files in os.walk(r"runs/detect/runs/detect"):
    for file in files:
        if file.endswith(".pt"):
            full_path = os.path.join(root, file)
            weight_files.append(full_path)

weight_files = sorted(weight_files, reverse=True)

if not weight_files:
    print("No weight files found!")
    exit(1)

print(f"Found {len(weight_files)} weight files:")
for i, wf in enumerate(weight_files):
    print(f"  {i}: {wf}")

# Start with the first (best) model
current_model_idx = 0
model_path = weight_files[current_model_idx]
print(f"\nLoading model: {model_path}")
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit(1)

print("\nWebcam opened!")
print(f"Sending data to ESP32 at: {ESP32_URL}")
print("Controls:")
print("  'q' - Quit")
print("  'n' - Next weight file")
print("  'p' - Previous weight file")

# Detection smoothing
history_length = 5
detection_history = []
emergency_history = []
frame_count = 0

def detect_emergency_in_quadrant(results, mid_x, mid_y):
    """Detect emergency vehicles and determine which quadrant they're in"""
    if not results[0].boxes:
        return None
    
    for box in results[0].boxes:
        # Get class name
        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower()
        
        # Check if it's an emergency vehicle
        if any(emergency_class in class_name for emergency_class in EMERGENCY_CLASSES):
            # Find which quadrant
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            
            if x_center < mid_x and y_center < mid_y:
                return "A"
            elif x_center >= mid_x and y_center < mid_y:
                return "B"
            elif x_center < mid_x and y_center >= mid_y:
                return "C"
            elif x_center >= mid_x and y_center >= mid_y:
                return "D"
    
    return None

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    mid_x = width // 2
    mid_y = height // 2
    
    # Run inference on full frame
    results = model(frame, conf=0.5, device='cpu')
    annotated_frame = results[0].plot()
    
    # Draw grid lines
    cv2.line(annotated_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)
    cv2.line(annotated_frame, (0, mid_y), (width, mid_y), (0, 255, 0), 2)
    
    # Count detections in each quadrant
    count_a = 0  # Top-left
    count_b = 0  # Top-right
    count_c = 0  # Bottom-left
    count_d = 0  # Bottom-right
    
    if results[0].boxes:
        for box in results[0].boxes:
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            
            if x_center < mid_x and y_center < mid_y:
                count_a += 1
            elif x_center >= mid_x and y_center < mid_y:
                count_b += 1
            elif x_center < mid_x and y_center >= mid_y:
                count_c += 1
            elif x_center >= mid_x and y_center >= mid_y:
                count_d += 1
    
    # Detect emergency vehicles
    emergency_road = detect_emergency_in_quadrant(results, mid_x, mid_y)
    
    # Add detection to history
    detection_history.append((count_a, count_b, count_c, count_d))
    if len(detection_history) > history_length:
        detection_history.pop(0)
    
    # Add emergency to history
    emergency_history.append(emergency_road)
    if len(emergency_history) > history_length:
        emergency_history.pop(0)
    
    # Calculate smoothed counts
    if len(detection_history) > 0:
        smoothed_a = sorted([d[0] for d in detection_history])[len(detection_history)//4]
        smoothed_b = sorted([d[1] for d in detection_history])[len(detection_history)//4]
        smoothed_c = sorted([d[2] for d in detection_history])[len(detection_history)//4]
        smoothed_d = sorted([d[3] for d in detection_history])[len(detection_history)//4]
    else:
        smoothed_a, smoothed_b, smoothed_c, smoothed_d = count_a, count_b, count_c, count_d
    
    # Determine smoothed emergency (most common in history)
    smoothed_emergency = None
    if emergency_history:
        non_none = [e for e in emergency_history if e is not None]
        if len(non_none) > len(emergency_history) // 2:
            # Majority vote
            from collections import Counter
            smoothed_emergency = Counter(non_none).most_common(1)[0][0]
    
    # Add labels and counts
    color_a = (0, 0, 255) if smoothed_emergency == "A" else (255, 0, 0)
    color_b = (0, 0, 255) if smoothed_emergency == "B" else (255, 0, 0)
    color_c = (0, 0, 255) if smoothed_emergency == "C" else (255, 0, 0)
    color_d = (0, 0, 255) if smoothed_emergency == "D" else (255, 0, 0)
    
    cv2.putText(annotated_frame, f"A: {smoothed_a}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_a, 2)
    cv2.putText(annotated_frame, f"B: {smoothed_b}", (mid_x + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_b, 2)
    cv2.putText(annotated_frame, f"C: {smoothed_c}", (10, mid_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_c, 2)
    cv2.putText(annotated_frame, f"D: {smoothed_d}", (mid_x + 10, mid_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_d, 2)
    
    if smoothed_emergency:
        cv2.putText(annotated_frame, f"EMERGENCY: Road {smoothed_emergency}", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Send HTTP POST every 5 frames
    frame_count += 1
    if frame_count % 5 == 0:
        payload = {
            "A": int(smoothed_a),
            "B": int(smoothed_b),
            "C": int(smoothed_c),
            "D": int(smoothed_d),
            "total": int(smoothed_a + smoothed_b + smoothed_c + smoothed_d)
        }
        
        # Add emergency field if detected
        if smoothed_emergency:
            payload["emergency"] = smoothed_emergency
        
        try:
            response = requests.post(ESP32_URL, json=payload, timeout=0.5)
            if response.status_code == 200:
                emergency_str = f" | EMERGENCY: {smoothed_emergency}" if smoothed_emergency else ""
                print(f"Sent: {payload['A']}, {payload['B']}, {payload['C']}, {payload['D']}{emergency_str}")
            else:
                print(f"ESP32 responded with status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send data: {e}")
    
    # Create display
    quad_tl_ann = annotated_frame[0:mid_y, 0:mid_x]
    quad_tr_ann = annotated_frame[0:mid_y, mid_x:width]
    quad_bl_ann = annotated_frame[mid_y:height, 0:mid_x]
    quad_br_ann = annotated_frame[mid_y:height, mid_x:width]
    
    top_half = cv2.hconcat([quad_tl_ann, quad_tr_ann])
    bottom_half = cv2.hconcat([quad_bl_ann, quad_br_ann])
    combined_frame = cv2.vconcat([top_half, bottom_half])
    
    model_name = weight_files[current_model_idx].split("\\")[-3]
    weight_name = weight_files[current_model_idx].split("\\")[-1]
    cv2.putText(combined_frame, f"Model: {model_name} ({weight_name})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined_frame, f"Model {current_model_idx+1}/{len(weight_files)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8 Live Detection - Smart Traffic", combined_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_model_idx = (current_model_idx + 1) % len(weight_files)
        model_path = weight_files[current_model_idx]
        print(f"\nLoading model: {model_path}")
        model = YOLO(model_path)
    elif key == ord('p'):
        current_model_idx = (current_model_idx - 1) % len(weight_files)
        model_path = weight_files[current_model_idx]
        print(f"\nLoading model: {model_path}")
        model = YOLO(model_path)

cap.release()
cv2.destroyAllWindows()