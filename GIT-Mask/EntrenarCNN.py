import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ast
import cv2

def load_data(csv_path, img_height, img_width, channels):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"El archivo CSV {csv_path} está vacío.")
            return None, None
        # Convertir la cadena de píxeles en un array numpy y redimensionar las imágenes
        pixels = []
        for pixel_str in df['pixels']:
            pixel_data = np.array(ast.literal_eval(pixel_str))
            if pixel_data.size != img_height * img_width * channels:
                print(f"Dimensiones incorrectas en {csv_path}. Esperado: {img_height * img_width * channels}, Encontrado: {pixel_data.size}")
                return None, None
            pixel_data = pixel_data.reshape((img_height, img_width, channels))
            pixels.append(cv2.resize(pixel_data, (img_width, img_height)))
        pixels = np.array(pixels)
        labels = df['person'].astype('category').cat.codes
        return pixels, labels
    except Exception as e:
        print(f"Error al cargar datos desde {csv_path}: {e}")
        return None, None

def create_time_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)

def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.LSTM(64, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(csv_base_dir):
    all_csv_files = [os.path.join(csv_base_dir, f) for f in os.listdir(csv_base_dir) if f.endswith('.csv')]
    
    if not all_csv_files:
        print("No se encontraron archivos CSV en el directorio.")
        return
    
    all_pixels = []
    all_labels = []
    
    img_height = 480  # Ajustar según la altura de las imágenes
    img_width = 848  # Ajustar según el ancho de las imágenes
    channels = 3  # Ajustar según los canales de las imágenes (por ejemplo, RGB)
    
    for csv_file in all_csv_files:
        if 'combined_frames.csv' in csv_file:
            continue
        pixels, labels = load_data(csv_file, img_height, img_width, channels)
        if pixels is not None and labels is not None:
            all_pixels.append(pixels)
            all_labels.append(labels)
    
    if not all_pixels or not all_labels:
        print("No se encontraron datos válidos para entrenar el modelo.")
        return

    X = np.vstack(all_pixels)
    y = np.concatenate(all_labels)
    
    window_size = 5  # Ajustar según sea necesario

    X_windows = create_time_windows(X, window_size)
    y_windows = y[window_size-1:]
    
    input_shape = (window_size, img_height, img_width, channels)
    model = build_model(input_shape)
    
    model.fit(X_windows, y_windows, epochs=50, batch_size=4, validation_split=0.2, verbose=1)  # Reducir batch_size
    
    model.save('trained_model.h5')  # Guardar el modelo
    print("Modelo entrenado y guardado con éxito.")