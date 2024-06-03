"""
Cuantificación de imágenes con k-means
Cuantificación es una técnica de compresión con pérdida que consiste en agrupar 
todo un rango de valores en uno solo. Si cuantificamos el color de una imagen, 
reducimos el número de colores necesarios para representarla y el tamaño del fichero 
de la misma disminuye. Esto es importante, por ejemplo, para representar una imagen en 
dispositivos que sólo dan soporte a un número limitado de colores.

Vamos a cuantificar el color de la imagen siguiente utilizando k-means.

Cargamos la imagen
"""
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Abrir la imagen
I = Image.open("K-Means/tienda.jpg")



# Convertir la imagen en una matriz numpy y normalizarla
I1 = np.asarray(I, dtype=np.float32) / 255

# Extraer los canales de color por separado
R = I1[:, :, 0]
G = I1[:, :, 1]
B = I1[:, :, 2]

# Convertir los canales en matrices de una dimensión y construir la matriz de tres columnas
XR = R.reshape((-1, 1))
XG = G.reshape((-1, 1))
XB = B.reshape((-1, 1))
X = np.concatenate((XR, XG, XB), axis=1)

# Aplicar K-means para agrupar los colores en 60 grupos, 
# o nuevos colores, que se corresponderán con los centroides obtenidos con el k-means
n = 60
k_means = KMeans(n_clusters=n)
k_means.fit(X)

#Los centroides finales son los nuevos colores y cada pixel tiene ahora una 
# etiqueta que dice a qué grupo o cluster pertenece
centroides = k_means.cluster_centers_
etiquetas = k_means.labels_

#A partir de las etiquetas y los colores (intensidades de rojo, verde y azul) 
# de los centroides reconstruimos la matriz de la imagen utilizando únicamente 
# los colores de los centroides.
m = XR.shape
for i in range(m[0]):
    XR[i] = centroides[etiquetas[i]][0]
    XG[i] = centroides[etiquetas[i]][1]
    XB[i] = centroides[etiquetas[i]][2]

XR.shape = R.shape
XG.shape = G.shape
XB.shape = B.shape
XR = XR[:, :, np.newaxis]
XG = XG[:, :, np.newaxis]
XB = XB[:, :, np.newaxis]
Y = np.concatenate((XR, XG, XB), axis=2)

# Representar la imagen comprimida con 60 colores
imagen_comprimida = Image.fromarray(np.uint8(Y * 255))
imagen_comprimida.show()

# Guardar la imagen comprimida
imagen_comprimida.save("K-Means/tienda_comprimida.jpg")
#751 KB  KB el fichero inicial y 114 KB KB el fichero final.