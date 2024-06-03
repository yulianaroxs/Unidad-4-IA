import cv2
import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf

def predict_person(video_path, model_path, frames_per_second, img_height, img_width, channels, window_size):
    def extract_frames(video_path, frames_per_second):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / frames_per_second)
        
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                frame = cv2.resize(frame, (img_width, img_height))
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return np.array(frames)
    
    def average_frames(frames):
        return np.mean(frames, axis=0).astype(np.uint8)
    
    def prepare_data(frame, img_height, img_width, channels):
        flat_frame = frame.flatten().tolist()
        pixel_data = np.array(flat_frame).reshape((img_height, img_width, channels))
        return pixel_data

    # Extraer y procesar los frames del video
    frames = extract_frames(video_path, frames_per_second)
    avg_frame = average_frames(frames)
    processed_data = prepare_data(avg_frame, img_height, img_width, channels)
    
    # Crear ventanas de tiempo
    X_windows = create_time_windows(np.array([processed_data]), window_size)
    
    # Cargar el modelo
    model = tf.keras.models.load_model(model_path)
    
    # Hacer la predicción
    predictions = model.predict(X_windows)
    
    # Decodificar la predicción
    predicted_class = np.argmax(predictions, axis=1)
    
    return predicted_class