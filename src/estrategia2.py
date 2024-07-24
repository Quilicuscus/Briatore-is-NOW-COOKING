from ajuste1 import polinomio_grado2_prueba_con_original
from ajuste1 import polinomio_grado3_estimacion
from ajuste1 import exponencial_estimado
from ajuste1 import nuevologaritmo

def posicicion_elemento_mas_pequeno(lista):
    if not lista:
        return None
    min_value = min(lista)
    min_index = lista.index(min_value)
    return min_index

"""Primero tenemos que tener el array con las vueltas a analizar"""

vueltas_perez_fp2_hungria = [82.964, 83.859, 82.709, 83.183, 83.643]
vueltas_ver_fp2_COTA = [101.285, 101.452, 101.611, 101.710, 101.310, 103.686]

def analisis_pre_carrera(vueltas, neumatico:str):
    """Analisis precarrera para ver cual es el mejor ajuste para las vueltas"""
    p_valores = []

    resultado_vueltas_polinomio_grado2_original,resultado_pvalor_polinomio_grado2_original = polinomio_grado2_prueba_con_original(vueltas, 5)
    resultado_pvalor_polinomio_grado3_original,resultado_vueltas_polinomio_grado3_original = polinomio_grado3_estimacion(vueltas)
    resultado_pvalor_exponencial_original, resultado_vueltas_exponencial_original = exponencial_estimado(vueltas)
    resultado_pvalor_logaritmo_original, resultado_vueltas_logaritmo_original = nuevologaritmo(vueltas)

    p_valores = []
    p_valores.append(resultado_pvalor_polinomio_grado2_original)
    p_valores.append(resultado_pvalor_polinomio_grado3_original)
    p_valores.append(resultado_pvalor_exponencial_original)
    p_valores.append(resultado_pvalor_logaritmo_original)

    print(p_valores)

    posicion = posicicion_elemento_mas_pequeno(p_valores)
    if posicion == 0:
        print("polinomio de grado 2")
        return resultado_vueltas_polinomio_grado2_original
    elif posicion == 1:
        print("polinomio de grado 3")
        return resultado_vueltas_polinomio_grado3_original
    elif posicion == 2:
        print("exponencial")
        return resultado_vueltas_exponencial_original
    elif posicion == 3:
        print("logaritmo")
        return resultado_vueltas_logaritmo_original


