import pandas as pd
import numpy as np
import tensorflow as tf
from pyswarm import pso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import io

# Corrigiendo la codificación estándar para evitar errores de Unicode en algunas plataformas
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize_data(df):
    # Elimina espacios en blanco en los nombres de las columnas y normaliza los datos numéricos
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df

# Cargar los datos CSV
data = pd.read_csv('Entra-salidas/entrada y salida2.csv', encoding='utf-8')

# Normalizar y limpiar los datos
data = normalize_data(data)

# Calcular el IQR para identificar y filtrar outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_filtered = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

# Asegurar de seleccionar las columnas correctas para X y y
X = data_filtered.iloc[:, :5].values
y = data_filtered.iloc[:, 9].values  # Asegúrate de que esta es la columna correcta para 'y'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def select_columns(arr, columns):
    selected_columns = arr[:, np.where(columns)[0]]
    if selected_columns.shape[1] < 5:
        selected_columns = np.pad(selected_columns, ((0, 0), (0, 5 - selected_columns.shape[1])), mode='constant')
    return selected_columns

all_solutions = []

def objective_function(x):
    selected_columns = select_columns(X_test, x)
    loss = model.evaluate(selected_columns, y_test, verbose=0)
    all_solutions.append((x, loss))
    return loss

lower_bound = [0, 0, 0, 0, 0]
upper_bound = [1, 1, 1, 1, 1]

def clip_to_binary(x):
    return np.round(x).astype(int)

def constrained_objective_function(x):
    clipped_x = clip_to_binary(x)
    return objective_function(clipped_x)

input_shape = X_train.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2)

y_pred = model.predict(X_test).flatten()

# Asumiendo que 'y_test' contiene los valores reales y 'y_pred' contiene los valores predichos
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Datos Originales de pH', alpha=0.7, marker='o')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones de pH', alpha=0.7, marker='x')
plt.title('Comparación de los valores originales y predichos de pH')
plt.xlabel('Índice de muestra')
plt.ylabel('pH')
plt.legend()
plt.show()

best_solution, best_value = pso(constrained_objective_function, lower_bound, upper_bound, maxiter=100, swarmsize=30)

sorted_solutions = sorted(all_solutions, key=lambda x: x[1])

printed_solutions = set()
printed_count = 0

unique_solutions = []

for solution, _ in sorted_solutions:
    rounded_solution = np.round(solution).astype(int)
    if rounded_solution.tolist() not in unique_solutions:
        unique_solutions.append(rounded_solution.tolist())
        print(f"Solución {len(unique_solutions)}: {rounded_solution}")
        if len(unique_solutions) == 3:
            break

print("Todas las soluciones únicas:", unique_solutions)

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

def evaluate_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_val, y_val), verbose=0)
    val_loss = history.history['val_loss'][-1]
    return val_loss

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
