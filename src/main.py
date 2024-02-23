import csv
from pathlib import Path

from carrera import Carrera


condiciones_carrera = []

# Localizar input.csv (en la carpeta raíz del repositorio)
INPUT = Path(__file__).parents[1] / "input.csv"


# Leer input.csv, ignorar los nombres de columnas y tomar las condiciones de carrera
with open(INPUT) as input:
    archivo = csv.reader(input, delimiter = ",")
    numero_fila = 0
    for fila in archivo:
        if numero_fila:
            condiciones_carrera = fila
        numero_fila += 1

# Crear la carrera
# carrera = Carrera(*condiciones_carrera)
