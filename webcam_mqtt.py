import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import paho.mqtt.client as mqtt
import json

# MQTT Configuration
MQTT_BROKER = "192.168.1.100"  # Change to your ESP32's IP or MQTT broker IP
MQTT_PORT = 1883
MQTT_TOPIC = "traffic/detection"
MQTT_CLIENT_ID = "traffic-detector"

# Initialize MQTT client
client = mqtt.Client(MQTT_CLIENT_ID)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker at {MQTT_BROKER}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")

client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Connect to MQTT broker
try:
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
except Exception as e:
    print(f"Error connecting to MQTT broker: {e}")
    print("MQTT disabled. Run without wireless connection.")

# Find all weight files
weight_files = []
for root, dirs, files in os.walk(r"runs/detect/runs/detect"):
    for file in files:
        if file.endswith(".pt"):
            full_path = os.path.join(root, file)
            weight_files.append(full_path)

weight_files = sorted(weight_files, reverse=True)  # Most recent first

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
print("Controls:")
print("  'q' - Quit")
print("  'n' - Next weight file")
print("  'p' - Previous weight file")

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Detection smoothing
history_length = 5
detection_history = []
frame_count = 0

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
    count_a = 0
    count_b = 0
    count_c = 0
    count_d = 0
    
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
    
    # Add detection to history
    detection_history.append((count_a, count_b, count_c, count_d))
    if len(detection_history) > history_length:
        detection_history.pop(0)
    
    # Calculate smoothed counts
    if len(detection_history) > 0:
        smoothed_a = sorted([d[0] for d in detection_history])[len(detection_history)//2]
        smoothed_b = sorted([d[1] for d in detection_history])[len(detection_history)//2]
        smoothed_c = sorted([d[2] for d in detection_history])[len(detection_history)//2]
        smoothed_d = sorted([d[3] for d in detection_history])[len(detection_history)//2]
    else:
        smoothed_a, smoothed_b, smoothed_c, smoothed_d = count_a, count_b, count_c, count_d
    
    # Add labels on frame
    cv2.putText(annotated_frame, f"A: {smoothed_a}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"B: {smoothed_b}", (mid_x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"C: {smoothed_c}", (10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"D: {smoothed_d}", (mid_x + 10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Add model info
    model_name = weight_files[current_model_idx].split("\\")[-3]
    cv2.putText(annotated_frame, f"Model: {model_name}", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Split and display
    quad_tl = annotated_frame[0:mid_y, 0:mid_x]
    quad_tr = annotated_frame[0:mid_y, mid_x:width]
    quad_bl = annotated_frame[mid_y:height, 0:mid_x]
    quad_br = annotated_frame[mid_y:height, mid_x:width]
    
    top_half = cv2.hconcat([quad_tl, quad_tr])
    bottom_half = cv2.hconcat([quad_bl, quad_br])
    display_frame = cv2.vconcat([top_half, bottom_half])
    
    cv2.imshow("YOLOv8 Detection + MQTT", display_frame)
    
    # Send data via MQTT every 5 frames
    frame_count += 1
    if frame_count % 5 == 0:
        payload = {
            "A": int(smoothed_a),
            "B": int(smoothed_b),
            "C": int(smoothed_c),
            "D": int(smoothed_d),
            "total": int(smoothed_a + smoothed_b + smoothed_c + smoothed_d)
        }
        try:
            client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
            print(f"MQTT Sent: {payload}")
        except Exception as e:
            print(f"Error sending MQTT: {e}")
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_model_idx = (current_model_idx + 1) % len(weight_files)
        model_path = weight_files[current_model_idx]
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
    elif key == ord('p'):
        current_model_idx = (current_model_idx - 1) % len(weight_files)
        model_path = weight_files[current_model_idx]
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

# Cleanup
cap.release()
cv2.destroyAllWindows()
client.loop_stop()
client.disconnect()
print("Disconnected from MQTT broker and closed webcam.")
