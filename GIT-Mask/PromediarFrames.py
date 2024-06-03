import cv2
import numpy as np
import os

def average_frames(frames_dir, output_path):
    frames = [cv2.imread(os.path.join(frames_dir, f)) for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]
    if not frames:
        print(f"No frames found in {frames_dir}.")
        return
    
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    if cv2.imwrite(output_path, avg_frame):
        print(f"Saved averaged frame to {output_path}.")
    else:
        print(f"Failed to save averaged frame to {output_path}.")

def process_all_averaging(input_base_dir, output_base_dir):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for root, dirs, files in os.walk(input_base_dir):
        for dir in dirs:
            person_name = dir
            frames_dir = os.path.join(root, dir)
            output_path = os.path.join(output_base_dir, f"{person_name}.jpg")
            print(f"Processing {frames_dir} -> {output_path}")
            average_frames(frames_dir, output_path)