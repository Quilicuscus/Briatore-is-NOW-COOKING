from ajuste1 import polinomio_grado2_prueba_con_original
from ajuste1 import polinomio_grado3_estimacion
from ajuste1 import exponencial_estimado

"""Primero tenemos que tener el array con las vueltas a analizar"""
p_valores = []
vueltas_perez_fp2_hungria = [82.964, 83.859, 82.709, 83.183, 83.643]
vueltas_ver_fp2_COTA = [101.285, 101.452, 101.611, 101.710, 101.310, 103.686]

resultado_vueltas_polinomio_grado2_original,resultado_pvalor_polinomio_grado2_original = polinomio_grado2_prueba_con_original(vueltas_ver_fp2_COTA, 5)
resultado_pvalor_polinomio_grado3_original,resultado_vueltas_polinomio_grado3_original = polinomio_grado3_estimacion(vueltas_ver_fp2_COTA)
resultado_pvalor_exponencial_original, resultado_vueltas_exponencial_original = exponencial_estimado(vueltas_ver_fp2_COTA)


p_valores = []
p_valores.append(resultado_pvalor_polinomio_grado2_original)
p_valores.append(resultado_pvalor_exponencial_original)
print(resultado_pvalor_polinomio_grado3_original)
print(resultado_vueltas_polinomio_grado3_original)

p_valores.append(resultado_pvalor_polinomio_grado3_original)
print(p_valores)


