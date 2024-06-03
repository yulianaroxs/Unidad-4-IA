import cv2
import os

def extract_frames(video_path, output_dir, frames_per_second):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frames_per_second)
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path}.")

def process_all_videos(input_base_dir, output_base_dir, frames_per_second):
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            if file.endswith('.avi'):
                person_name = os.path.basename(root)
                video_path = os.path.join(root, file)
                output_dir = os.path.join(output_base_dir, person_name)
                extract_frames(video_path, output_dir, frames_per_second)
