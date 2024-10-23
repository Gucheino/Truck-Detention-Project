import time
from extracted_frames import extract_frames
from yolo_detection import detect_trucks, draw_boxes
import cv2
import os

# Initialize time tracking
start_time = time.time()

# Define the video path and output folder for extracted frames
video_path = './video/first_10_minutes.avi'
output_folder = './extracted_frames'
detected_output_folder = './detected_frames'

# Initialize truck detection counter
truck_count = 0

# Extract frames at 1-second intervals
extract_frames(video_path, output_folder, interval=1.0)

# Create the output folder for detected truck frames if it doesn't exist
if not os.path.exists(detected_output_folder):
    os.makedirs(detected_output_folder)

# Process each extracted frame for truck detection
for frame_filename in os.listdir(output_folder):
    frame_path = os.path.join(output_folder, frame_filename)
    frame = cv2.imread(frame_path)

    # Perform truck detection
    truck_detections = detect_trucks(frame, size_threshold=0.1)

    # If trucks are detected, draw bounding boxes
    if truck_detections:
        frame_with_boxes = draw_boxes(frame, truck_detections)

        # Save the frame with detected trucks to the new folder
        detected_frame_path = os.path.join(detected_output_folder, frame_filename)
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
print(f"Truck detection completed. Potential Truck Detections: {truck_count}")
print(f"Total time taken: {minutes} minutes and {seconds} seconds")

