import numpy as np
from scipy.optimize import curve_fit

# Tiempos por vuelta originales (en segundos)
tiempos_por_vuelta = [25.001, 25.528, 25.813, 25.705, 26.397, 26.623, 26.703, 26.693]
tiempos_por_vuelta2 = [34.941, 34.215, 34.439, 34.318, 34.165, 34.654, 35.018, 34.191, 34.65, 34.516, 34.881]
tiempos_por_vuelta3 = [103.952, 104.861, 106.298, 104.928, 106.106]
tiempos_por_vuelta4 = [103.837, 103.745, 103.843, 105.745, 104.039, 105.592, 105.786, 104.979]
# Suponemos que el desgaste de los neumáticos se mide en segundos por vuelta
desgaste_por_vuelta = np.arange(len(tiempos_por_vuelta4))

# Función que representa la relación entre los tiempos por vuelta y el desgaste de los neumáticos, con intercepto
def relacion_desgaste(tiempo, inicial, incremento):
    return inicial + incremento * tiempo

# Ajustar la curva para encontrar el tiempo inicial del neumático y el incremento medio por vuelta
inicial_opt, incremento_opt = curve_fit(relacion_desgaste, desgaste_por_vuelta, tiempos_por_vuelta4)[0]

print("Tiempo inicial del neumático:", inicial_opt)
print("Incremento medio por vuelta:", incremento_opt)



