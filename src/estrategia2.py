from ajuste1 import polinomio_grado2_prueba_con_original

"""Primero tenemos que tener el array con las vueltas a analizar"""
p_valores = []
vueltas_perez_fp2_hungria = [82.964, 83.859, 82.709, 83.183, 83.643]
vueltas_ver_fp2_COTA = [101.285, 101.452, 101.611, 101.710, 101.310, 103.686]

resultado_vueltas_polinomio_grado2_original,resultado_pvalor_polinomio_grado2_original = polinomio_grado2_prueba_con_original(vueltas_ver_fp2_COTA, 5)


p_valores = []
p_valores.append(resultado_pvalor_polinomio_grado2_original)



