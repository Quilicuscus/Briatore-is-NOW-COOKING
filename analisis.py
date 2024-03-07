import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = [1, 2, 3, 4, 5]
y = [121.5,121.65,121.78,121.89,122.000]

# Gráfico de líneas
plt.plot(x, y)

# Añade etiquetas y título
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Gráfico de líneas de dos variables')
plt.show()
y_array = np.array(y)

# Media, mediana y desviación estándar
media = np.mean(y_array)
mediana = np.median(y_array)
desviacion_estandar = np.std(y_array)

# Histograma de los datos
plt.hist(y_array, bins=10, alpha=0.5, color='blue', edgecolor='black')

# Líneas verticales para mostrar media y mediana
plt.axvline(media, color='red', linestyle='dashed', linewidth=1, label='Media')
plt.axvline(mediana, color='green', linestyle='dashed', linewidth=1, label='Mediana')

# Añade etiquetas y título al gráfico
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de los datos y medidas estadísticas')
plt.legend()

# Muestra el gráfico
plt.show()




