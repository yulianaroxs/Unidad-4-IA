import pandas as pd
import numpy as np
import tensorflow as tf
from pyswarm import pso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cargar los datos CSV
data = pd.read_csv('Entra-salidas/entrada y salida normalizado.csv', encoding='utf-8')

# Cálculo del IQR para cada característica
# Calcular el primer y tercer cuartil de cada columna en los datos
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
# Calcular el rango intercuartílico (IQR) para identificar los outliers
IQR = Q3 - Q1

# Definir los límites para considerar un dato como válido
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrar los datos para eliminar outliers
data_filtered = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]


# Dividir los datos en características (X) y salida esperada (y)
X = data.iloc[:, :5].values
y = data.iloc[:, 9].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def select_columns(arr, columns):
    selected_columns = arr[:, np.where(columns)[0]]
    if selected_columns.shape[1] < 5:
        # Si la cantidad de columnas seleccionadas es menor que 5, rellenar con ceros
        selected_columns = np.pad(selected_columns, ((0, 0), (0, 5 - selected_columns.shape[1])), mode='constant')
    return selected_columns

# Lista para almacenar todas las soluciones y sus pérdidas
all_solutions = []

# Función objetivo que será optimizada por PSO
def objective_function(x):
    selected_columns = select_columns(X_test, x)
    # Evaluar el modelo con las columnas seleccionadas y almacenar la pérdida
    loss = model.evaluate(selected_columns, y_test, verbose=0)
    all_solutions.append((x, loss))
    return loss  # La función retorna la pérdida como objetivo a minimizar


# Definición del rango para cada dimensión en PSO (en este caso, 5 dimensiones)
lower_bound = [0, 0, 0, 0, 0]  # límite inferior para cada dimensión
upper_bound = [1, 1, 1, 1, 1]  # límite superior para cada dimensión

# Función para convertir valores a binario (0 o 1)
def clip_to_binary(x):
    return np.round(x).astype(int)

# Función que aplica la conversión binaria y evalúa la función objetivo
def constrained_objective_function(x):
    clipped_x = clip_to_binary(x)
    return objective_function(clipped_x)

# Definir la forma de entrada fija
input_shape = X_train.shape[1]

# Definir el modelo de red neuronal con capa de entrada fija
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),# se define la capa de entrada con la forma de entrada fija
    tf.keras.layers.Dense(8, activation='relu'), # se utilizan 8 neuronas en la capa oculta
    tf.keras.layers.Dense(4, activation='relu'),# se utilizan 4 neuronas en la capa oculta
    tf.keras.layers.Dense(1) # se utiliza una neurona en la capa de salida
])

# Compilar el modelo con el optimizador Adam y la pérdida del error cuadrático medio
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test).flatten()

# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Datos Originales de pH', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones de pH', alpha=0.5)
plt.title('Comparación de los valores originales y predichos de pH')
plt.xlabel('Índice de muestra')
plt.ylabel('pH')
plt.legend()
plt.show()

# Ejecución de PSO, se utilizan 30 partículas y 100 iteraciones
best_solution, best_value = pso(constrained_objective_function, lower_bound, upper_bound, maxiter=100, swarmsize=30)


# Ordenar las soluciones encontradas basándose en la pérdida
sorted_solutions = sorted(all_solutions, key=lambda x: x[1])

printed_solutions = set()  # Conjunto para almacenar soluciones ya impresas
printed_count = 0  # Contador para llevar el registro de las soluciones impresas

unique_solutions = []  # Lista para almacenar las tres soluciones únicas

for solution, _ in sorted_solutions:
    rounded_solution = np.round(solution).astype(int)
    if rounded_solution.tolist() not in unique_solutions:
        unique_solutions.append(rounded_solution.tolist())
        print(f"Solución {len(unique_solutions)}: {rounded_solution}")
        if len(unique_solutions) == 3:
            break

print("Todas las soluciones únicas:", unique_solutions)

# Función para construir y compilar un nuevo modelo basado en una solución específica
def build_model(solution):
    selected_columns = select_columns(X_train, solution)
    input_shape = selected_columns.shape[1]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Función para evaluar un modelo con datos de entrenamiento y validación
def evaluate_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_val, y_val), verbose=0)
    val_loss = history.history['val_loss'][-1]
    return val_loss

# Encontrar y almacenar el mejor modelo basado en la pérdida de validación
best_val_loss = float('inf')
best_model = None
best_solution_vector = None

for solution in unique_solutions:
    model = build_model(solution)
    val_loss = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_solution_vector = solution

print("La mejor pérdida de validación encontrada:", best_val_loss)
print("El vector de la mejor solución encontrada:", best_solution_vector)