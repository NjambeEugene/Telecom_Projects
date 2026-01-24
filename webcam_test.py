import cv2
from ultralytics import YOLO

# Load the trained model
model_path = r"runs/detect/runs/detect/train/weights/best.pt"
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit(1)

print("Webcam opened. Press 'q' to quit")
print("Running real-time detection...")

# Set webcam resolution (optional, for faster processing)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        break
    
    # Run inference
    results = model(frame, conf=0.5, device='cpu')
    
    # Draw results on frame
    annotated_frame = results[0].plot()
    
    # Display frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam test completed!")
