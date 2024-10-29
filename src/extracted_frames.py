import cv2

def extract_frames(video_path, interval=1.0):
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Get the frame rate (frames per second) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)  # Number of frames to skip to match the interval

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only add frames at the specified interval
        if frame_count % frame_interval == 0:
            frames.append(frame)  # Store frame in memory
            print(f"Extracted frame {len(frames)}")

        frame_count += 1

    cap.release()
    return frames  # Return frames as a list in memory
