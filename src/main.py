from pathlib import Path

from carrera import Carrera



# Localizar input.csv en la carpeta raíz del repositorio
INPUT = str(Path(__file__).parents[1] / "input.csv")


# Crear la clase carrera e imprimir sus vueltas
carrera = Carrera(1, 1, 1, INPUT)
for vuelta in carrera.vueltas:
    print(vuelta)
    print()
