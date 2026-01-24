import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import requests
import json
import time
from collections import Counter

# ESP32 Configuration
ESP32_IP = "192.168.137.81"
ESP32_PORT = 80
ESP32_URL = f"http://{ESP32_IP}:{ESP32_PORT}/update"

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

# Detection history - using 10 frames for 30% threshold check
HISTORY_LENGTH = 10
THRESHOLD = 0.30  # 30% threshold

detection_history_a = []
detection_history_b = []
detection_history_c = []
detection_history_d = []

last_send_time = 0
SEND_INTERVAL = 1.0  # Send every 1 second

def get_smart_count(history):
    """
    Returns the highest value if it appears ≥30% of the time,
    otherwise returns the most frequent value (mode)
    """
    if len(history) == 0:
        return 0
    
    counter = Counter(history)
    
    # Get highest value and its frequency
    highest_value = max(history)
    highest_freq = counter[highest_value]
    highest_percentage = highest_freq / len(history)
    
    # If highest appears ≥30%, use it
    if highest_percentage >= THRESHOLD:
        return highest_value
    
    # Otherwise, use the most frequent value (mode)
    mode_value = counter.most_common(1)[0][0]
    return mode_value

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    mid_x = width // 2
    mid_y = height // 2
    
    # Run inference with higher confidence for better detections
    results = model(frame, conf=0.6, device='cpu', iou=0.5)
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
    
    # Add current counts to history
    detection_history_a.append(count_a)
    detection_history_b.append(count_b)
    detection_history_c.append(count_c)
    detection_history_d.append(count_d)
    
    # Keep only last HISTORY_LENGTH frames
    if len(detection_history_a) > HISTORY_LENGTH:
        detection_history_a.pop(0)
    if len(detection_history_b) > HISTORY_LENGTH:
        detection_history_b.pop(0)
    if len(detection_history_c) > HISTORY_LENGTH:
        detection_history_c.pop(0)
    if len(detection_history_d) > HISTORY_LENGTH:
        detection_history_d.pop(0)
    
    # Get smart counts using 30% highest or mode logic
    smart_count_a = get_smart_count(detection_history_a)
    smart_count_b = get_smart_count(detection_history_b)
    smart_count_c = get_smart_count(detection_history_c)
    smart_count_d = get_smart_count(detection_history_d)
    
    # Add labels and counts
    cv2.putText(annotated_frame, f"A: {smart_count_a}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"B: {smart_count_b}", (mid_x + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"C: {smart_count_c}", (10, mid_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"D: {smart_count_d}", (mid_x + 10, mid_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Send HTTP POST every 1 second
    current_time = time.time()
    if current_time - last_send_time >= SEND_INTERVAL:
        payload = {
            "A": int(smart_count_a),
            "B": int(smart_count_b),
            "C": int(smart_count_c),
            "D": int(smart_count_d),
            "total": int(smart_count_a + smart_count_b + smart_count_c + smart_count_d)
        }
        
        try:
            response = requests.post(ESP32_URL, json=payload, timeout=0.5)
            if response.status_code == 200:
                print(f"Sent: A={payload['A']}, B={payload['B']}, C={payload['C']}, D={payload['D']}")
            else:
                print(f"ESP32 responded with status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send data: {e}")
        
        last_send_time = current_time
    
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
    
    # Show algorithm details
    cv2.putText(combined_frame, f"Algorithm: 30% Highest or Mode", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
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