import cv2
import pandas as pd
import os
import json

def frame_to_csv(frame_path, csv_path, person_name):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Could not read frame from {frame_path}.")
        return
    
    # Convertir el frame a una lista de enteros en lugar de una cadena
    flat_frame = frame.flatten().tolist()
    data = {
        'person': person_name,
        'pixels': json.dumps(flat_frame)  # Serializar la lista de píxeles como una cadena JSON
    }
    
    df = pd.DataFrame([data])
    df.to_csv(csv_path, index=False)
    print(f"Saved frame data to {csv_path}.")

def process_all_to_csv(input_base_dir, output_base_dir):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    for file in os.listdir(input_base_dir):
        if file.endswith('.jpg'):
            person_name = file.split('.')[0]
            frame_path = os.path.join(input_base_dir, file)
            csv_path = os.path.join(output_base_dir, f"{person_name}.csv")
            print(f"Processing {frame_path} -> {csv_path}")
            frame_to_csv(frame_path, csv_path, person_name)
