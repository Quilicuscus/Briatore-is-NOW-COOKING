import csv

from carrera import Carrera


condiciones_carrera = []

# Leer input.csv, ignorar los nombres de columnas y tomar las condiciones de carrera
with open("../input.csv") as input:
    archivo = csv.reader(input, delimiter = ",")
    numero_fila = 0
    for fila in archivo:
        if numero_fila:
            condiciones_carrera = fila
        numero_fila += 1

# Crear la carrera
# carrera = Carrera(*condiciones_carrera)
