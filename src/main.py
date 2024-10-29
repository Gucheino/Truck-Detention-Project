import time
from extracted_frames import extract_frames
from yolo_detection import detect_trucks, draw_boxes
import cv2
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

# Initialize time tracking
start_time = time.time()

# Define the video path and output folder for detected frames
video_path = './video/first_30_minutes.avi'
detected_output_folder = './detected_frames'

# Initialize truck detection counter
truck_count = 0

# Extract frames into memory
frames = extract_frames(video_path, interval=1.0)

# Create the output folder for detected truck frames if it doesn't exist
if not os.path.exists(detected_output_folder):
    os.makedirs(detected_output_folder)

# Process each frame in memory for truck detection
for i, frame in enumerate(frames):
    # Perform truck detection
    truck_detections = detect_trucks(frame, size_threshold=0.1)

    # If trucks are detected, draw bounding boxes and save the frame
    if truck_detections:
        frame_with_boxes = draw_boxes(frame, truck_detections)

        # Save the frame with detected trucks to the new folder
        detected_frame_path = os.path.join(detected_output_folder, f"frame_{i:05d}.jpg")
        cv2.imwrite(detected_frame_path, frame_with_boxes)
        print(f"Saved detected truck frame: {detected_frame_path}")
        
        # Increment the truck detection counter
        truck_count += 1

# Calculate the total time taken
end_time = time.time()
elapsed_time = end_time - start_time

# Convert elapsed time to minutes and seconds
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

# Display total number of detected truck frames and time taken
print(f"Truck detection completed. Total trucks detected in frames: {truck_count}")
print(f"Total time taken: {minutes} minutes and {seconds} seconds")
print(f"App was run on device: {device}")