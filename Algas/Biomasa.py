import io
import sys
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ajuste para asegurarse de que la salida estándar maneje correctamente la codificación UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Rutas a la carpeta de imágenes y al archivo de Excel con los nombres de imágenes y biomasa
folder_path = 'Algas/IMAGENES'
file_path = 'Algas/Datos frames.xlsx'

# Verificar si la ruta del archivo Excel y la carpeta de imágenes existen
if not os.path.exists(folder_path):
    print(f"La carpeta no existe: {folder_path}")
    sys.exit(1)
if not os.path.exists(file_path):
    print(f"El archivo no existe: {file_path}")
    sys.exit(1)

# Cargar los datos desde el archivo Excel a un DataFrame de pandas
data = pd.read_excel(file_path, names=['nombre_imagen', 'biomasa'])

# Obtener los nombres de archivo de las imágenes en la carpeta especificada
image_files = os.listdir(folder_path)

# Filtrar el DataFrame para incluir solo aquellas entradas con imágenes reales
data = data[data['nombre_imagen'].isin([os.path.splitext(filename)[0] for filename in image_files])]

# Lista para almacenar imágenes procesadas
images = []
nuevo_ancho = 100
nuevo_alto = 100
new_size = (nuevo_ancho, nuevo_alto)

# Procesar cada imagen: leer, verificar, convertir a RGB, redimensionar y normalizar
for filename in image_files:
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        print(f"El archivo no existe: {filepath}")
        continue
    print(f"Procesando archivo: {filepath}")
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error al cargar la imagen: {filepath}")
        continue
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print(f"Error al convertir la imagen {filepath}: {e}")
        continue
    image_resized = cv2.resize(image_rgb, new_size)
    image_normalized = image_resized / 255.0
    images.append(image_normalized)

# Convertir la lista de imágenes a un array de numpy
X = np.array(images)
labels = data['biomasa'].values

# Identificar clases con más de una muestra
unique_labels, label_counts = np.unique(labels, return_counts=True)
valid_labels = unique_labels[label_counts > 1]

# Filtrar imágenes y etiquetas válidas
valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]
X_valid = X[valid_indices]
y_valid = labels[valid_indices]

# Dividir datos en entrenamiento, validación y prueba
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

# Definir y compilar modelo de TensorFlow Keras para regresión
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(nuevo_ancho, nuevo_alto, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_valid, y_valid))

# Evaluar modelo en el conjunto de prueba
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Pérdida en el conjunto de prueba:", test_loss)
print("Error absoluto medio en el conjunto de prueba:", test_mae)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test).flatten()

# Aplanar y_test si es necesario para comparar
if y_test.ndim > 1:
    y_test = y_test.flatten()

# Graficar resultados de predicciones versus reales
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Datos Reales', marker='o', linestyle='', alpha=0.5)
plt.plot(y_pred, label='Predicción del Modelo', marker='*', linestyle='dotted', alpha=0.5)
plt.title('Comparación de los Datos Reales con las Predicciones del Modelo')
plt.xlabel('Número de Muestra')
plt.ylabel('Valor de Biomasa')
plt.legend()
plt.grid(True)
plt.show()

# Visualizar una imagen antes y después del entrenamiento
index = 0
imagen_original = cv2.imread(os.path.join(folder_path, image_files[index]))
if imagen_original is None:
    print(f"Error al cargar la imagen: {os.path.join(folder_path, image_files[index])}")
else:
    imagen_original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen_original_rgb)
    plt.title(f"Imagen Original: {image_files[index]}")
    plt.axis('off')
    plt.show()

    # Mostrar la misma imagen procesada y su predicción de biomasa
    imagen_procesada = X_valid[index]
    prediccion = model.predict(np.expand_dims(imagen_procesada, axis=0)).flatten()[0]
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen_procesada)
    plt.title(f"Después del Entrenamiento\nPredicción de Biomasa: {prediccion:.2f}")
    plt.axis('off')
    plt.show()
