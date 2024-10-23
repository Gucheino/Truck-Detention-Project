import cv2
import os

def extract_frames(video_path, output_folder, interval=1.0):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the frame rate (frames per second) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS could not be determined.")
        return
    
    frame_interval = int(fps * interval)  # Number of frames to skip to match the interval

    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only save frames at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Extracted {frame_filename}")
            extracted_count += 1

        frame_count += 1

    cap.release()

    print(f"Total frames extracted: {extracted_count}")
