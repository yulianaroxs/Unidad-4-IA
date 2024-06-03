import pandas as pd
import numpy as np
import tensorflow as tf
from pyswarm import pso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import io
from sklearn.metrics import mean_squared_error

# Redirección de la salida estándar para manejar la codificación UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cargar los datos desde el archivo Excel
file_path = 'Entra-salidas/Demanda_de_Agua.xlsx'
data = pd.read_excel(file_path)

# Eliminar la columna no deseada y renombrar las demás
data = data.drop(data.columns[0], axis=1)
data.columns = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']

# Preprocesamiento de datos para eliminar valores atípicos
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_filtered = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

# División de datos en características (X) y etiquetas (y)
X = data_filtered[['K']].values.reshape(-1, 1)
y = data_filtered[['K']].values.reshape(-1, 1)  # Modificación aquí para que y sea la misma columna K

# Dividir los datos en conjuntos de entrenamiento y evaluación (70% - 30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# División del 30% restante en prueba y validación (dividido a la mitad, cada uno con el 15% del total)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Método para crear las ventanas de tiempo
def create_time_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)

# Definir el tamaño de la ventana de tiempo
window_size = 20 

# Crear ventanas de tiempo para los conjuntos de entrenamiento, validación y evaluación
trainning_input_windows = create_time_windows(X_train, window_size)
evaluation_input_windows = create_time_windows(X_test, window_size)
validation_input_windows = create_time_windows(X_val, window_size)

# Obtener las etiquetas correspondientes para las ventanas
y_train_windows = y_train[window_size-1:]
y_val_windows = y_val[window_size-1:]
y_test_windows = y_test[window_size-1:]

# Aplanar los datos si es necesario, dependiendo de la estructura del modelo
trainning_input_windows_flat = trainning_input_windows.reshape(trainning_input_windows.shape[0], -1)
evaluation_input_windows_flat = evaluation_input_windows.reshape(evaluation_input_windows.shape[0], -1)
validation_input_windows_flat = validation_input_windows.reshape(validation_input_windows.shape[0], -1)

# Función para construir y compilar un nuevo modelo basado en una solución específica
def build_model(solution):
    input_shape = trainning_input_windows_flat.shape[1]  # Asegurarse de que la forma de entrada sea correcta para las ventanas

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1)  # Asegúrese de que la capa de salida tenga la dimensión correcta
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Función para evaluar un modelo con datos de entrenamiento y validación
def evaluate_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_val, y_val), verbose=0)
    val_loss = history.history['val_loss'][-1]
    return val_loss

# Lista para almacenar todas las soluciones y sus pérdidas
all_solutions = []

# Función objetivo que evalúa la pérdida del modelo con columnas seleccionadas
def objective_function(x):
    if np.sum(x) == 0:  # Verificación para asegurar que hay al menos una característica seleccionada
        return np.inf
    selected_columns = trainning_input_windows_flat
    if selected_columns.size == 0:
        return np.inf  # Retornar infinito si no hay columnas seleccionadas
    # Evaluar el modelo con las columnas seleccionadas y almacenar la pérdida
    loss = model.evaluate(selected_columns, y_train_windows, verbose=0)
    all_solutions.append((x, loss))
    return loss  # La función retorna la pérdida como objetivo a minimizar

# Definir y compilar el modelo de red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)  # Asegúrese de que la capa de salida tenga la dimensión correcta
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(trainning_input_windows_flat, y_train_windows, epochs=50, batch_size=16, validation_data=(validation_input_windows_flat, y_val_windows), verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss = model.evaluate(evaluation_input_windows_flat, y_test_windows, verbose=1)
print("Pérdida en el conjunto de prueba:", test_loss)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(evaluation_input_windows_flat).flatten()

# Asegurarse de que y_test y y_pred tienen la misma forma
if y_test_windows.ndim > 1:
    y_test_windows = y_test_windows.flatten()

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test_windows, label='Datos Reales', marker='o', linestyle='', alpha=0.5)
plt.plot(y_pred, label='Predicción del Modelo', marker='*', linestyle='dotted', alpha=0.5)
plt.title('Comparación de los Datos Reales con las Predicciones del Modelo')
plt.xlabel('Número de Muestra')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

# Calcular el MSE para el conjunto de validación
mse = mean_squared_error(y_val_windows, model.predict(validation_input_windows_flat))
print("MSE del conjunto de validación:", mse)

# Ejecución de PSO, se utilizan 30 partículas y 100 iteraciones
dimensionality = trainning_input_windows_flat.shape[1]
lower_bound = np.zeros(dimensionality)
upper_bound = np.ones(dimensionality)
best_solution, best_value = pso(objective_function, lower_bound, upper_bound, swarmsize=30, maxiter=100)

# Ordenar y mostrar soluciones únicas
sorted_solutions = sorted(all_solutions, key=lambda x: x[1])
unique_solutions = []
for solution, _ in sorted_solutions:
    rounded_solution = np.round(solution).astype(int)
    if rounded_solution.tolist() not in unique_solutions:
        unique_solutions.append(rounded_solution.tolist())
        print(f"Solución {len(unique_solutions)}: {rounded_solution}")
        if len(unique_solutions) == 3:
            break
print("Todas las soluciones únicas:", unique_solutions)

# Encontrar y almacenar el mejor modelo basado en la pérdida de validación
best_val_loss = float('inf')
best_model = None
best_solution_vector = None

for solution in unique_solutions:
    model = build_model(solution)
    val_loss = evaluate_model(model, trainning_input_windows_flat, y_train_windows, validation_input_windows_flat, y_val_windows)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_solution_vector = solution

print("La mejor pérdida de validación encontrada:", best_val_loss)
print("El vector de la mejor solución encontrada:", best_solution_vector)
