import torch
import cv2

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the allowed classes for bigger vehicles (e.g., truck and bus)
ALLOWED_CLASSES = ['truck', 'bus']

def detect_trucks(image, size_threshold=0.1):  # Increased size threshold to 10% of the image
    # Perform object detection on the image
    results = model(image)

    # Get the dimensions of the image for bounding box size filtering
    image_height, image_width, _ = image.shape

    # Filter results for trucks and buses only, and optionally filter by bounding box size
    truck_detections = []
    for detection in results.xyxy[0]:  # Detections are in the format [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, conf, class_id = detection[:6]
        label = int(class_id)

        # Check if the detected object is in the allowed classes (truck, bus)
        if results.names[label] in ALLOWED_CLASSES:
            # Calculate the bounding box size (width * height)
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = (box_width * box_height) / (image_width * image_height)  # Normalize by image size

            # Keep detections that are above the size threshold (to exclude small vehicles)
            if box_area > size_threshold:
                truck_detections.append(detection)

    return truck_detections

def draw_boxes(image, detections):
    for detection in detections:
        x1, y1, x2, y2, conf = map(int, detection[:5])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Truck/Bus: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image
