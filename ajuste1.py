import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f

def polinomio_grado2():
    # Función para ajustar un polinomio de grado 2 a los datos de tiempos por vuelta
    def ajustar_polinomio(tiempos_por_vuelta):
        x = np.arange(len(tiempos_por_vuelta))
        coeficientes, cov_matrix = curve_fit(lambda x, *c: np.polyval(c, x), x, tiempos_por_vuelta, p0=[0, 0, 0])
        return coeficientes, cov_matrix

    # Función para calcular el desgaste por vuelta a partir de los coeficientes del polinomio
    def calcular_desgaste(coeficientes):
        # Derivada de un polinomio de grado 2 es una función lineal
        desgaste_por_vuelta = 2 * coeficientes[0] * np.arange(len(coeficientes) - 1, 0, -1)
        return desgaste_por_vuelta

    # Función para calcular un único p-valor para la distribución general de los coeficientes
    def obtener_p_valor_general(tiempos_por_vuelta, coeficientes, cov_matrix, num_params):
        # Calcular el residuo cuadrático
        residuos = tiempos_por_vuelta - np.polyval(coeficientes, np.arange(len(tiempos_por_vuelta)))
        residuo_cuadratico = np.sum(residuos ** 2)
        # Calcular el residuo cuadrático reducido
        residuo_cuadratico_reducido = residuo_cuadratico / (len(tiempos_por_vuelta) - num_params)
        # Calcular la suma de cuadrados del modelo
        suma_cuadrados_modelo = np.sum((np.polyval(coeficientes, np.arange(len(tiempos_por_vuelta))) - np.mean(tiempos_por_vuelta)) ** 2)
        # Calcular la estadística F
        f_stat = (suma_cuadrados_modelo / num_params) / residuo_cuadratico_reducido
        # Calcular el p-valor
        p_valor = 1 - f.cdf(f_stat, num_params, len(tiempos_por_vuelta) - num_params)
        return p_valor

    # Datos de ejemplo: tiempos por vuelta
    tiempos_por_vuelta = np.array([103.25, 103.84, 104.69, 104.45, 105])  # Nuevos tiempos por vuelta en segundos

    # Ajuste de un polinomio de grado 2 a los datos de tiempos por vuelta
    coeficientes, cov_matrix = ajustar_polinomio(tiempos_por_vuelta)

    # Número de parámetros en el modelo (grado del polinomio + 1)
    num_params = len(coeficientes)

    # Calcular el p-valor general para la distribución de los coeficientes
    p_valor_general = obtener_p_valor_general(tiempos_por_vuelta, coeficientes, cov_matrix, num_params)

    # Imprimir el resultado
    print(f"P-valor general del ajuste: {p_valor_general}")

    # Gráfico del ajuste polinomial
    x = np.arange(len(tiempos_por_vuelta))
    y_pred = np.polyval(coeficientes, x)

    plt.plot(x, tiempos_por_vuelta, 'o', label='Tiempos por vuelta')
    plt.plot(x, y_pred, label='Ajuste polinomial')
    plt.title('Ajuste polinomial de grado 2 a los tiempos por vuelta')
    plt.xlabel('Vuelta')
    plt.ylabel('Tiempo por vuelta (s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def polinomio_grado3():


    # Función polinómica de grado 3
    def funcion_polinomica(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # Función para ajustar una función polinómica de grado 3 a los datos de tiempos por vuelta
    def ajustar_polinomica_grado_3(tiempos_por_vuelta):
        x = np.arange(len(tiempos_por_vuelta))
        parametros_optimos, cov_matrix = curve_fit(funcion_polinomica, x, tiempos_por_vuelta)
        return parametros_optimos, cov_matrix

    # Función para calcular un único p-valor para la distribución general de los coeficientes
    def obtener_p_valor_general(tiempos_por_vuelta, parametros_optimos, cov_matrix, num_params):
        # Calcular el residuo cuadrático
        residuos = tiempos_por_vuelta - funcion_polinomica(np.arange(len(tiempos_por_vuelta)), *parametros_optimos)
        residuo_cuadratico = np.sum(residuos ** 2)
        # Calcular el residuo cuadrático reducido
        residuo_cuadratico_reducido = residuo_cuadratico / (len(tiempos_por_vuelta) - num_params)
        # Calcular la suma de cuadrados del modelo
        suma_cuadrados_modelo = np.sum((funcion_polinomica(np.arange(len(tiempos_por_vuelta)), *parametros_optimos) - np.mean(tiempos_por_vuelta)) ** 2)
        # Calcular la estadística F
        f_stat = (suma_cuadrados_modelo / num_params) / residuo_cuadratico_reducido
        # Calcular el p-valor
        p_valor = 1 - f.cdf(f_stat, num_params, len(tiempos_por_vuelta) - num_params)
        return p_valor

    # Datos de ejemplo: tiempos por vuelta
    tiempos_por_vuelta = np.array([103.25, 103.84, 104.69, 104.45, 105])  # Nuevos tiempos por vuelta en segundos

    # Ajuste de una función polinómica de grado 3 a los datos de tiempos por vuelta
    parametros_optimos, cov_matrix = ajustar_polinomica_grado_3(tiempos_por_vuelta)

    # Número de parámetros en el modelo
    num_params = len(parametros_optimos)

    # Calcular el p-valor general para la distribución de los coeficientes
    p_valor_general = obtener_p_valor_general(tiempos_por_vuelta, parametros_optimos, cov_matrix, num_params)

    # Imprimir el resultado
    print(f"P-valor general del ajuste polinómico de grado 3: {p_valor_general}")

    # Gráfico del ajuste polinómico
    x = np.arange(len(tiempos_por_vuelta))
    y_pred = funcion_polinomica(x, *parametros_optimos)

    plt.plot(x, tiempos_por_vuelta, 'o', label='Tiempos por vuelta')
    plt.plot(x, y_pred, label='Ajuste polinómico de grado 3')
    plt.title('Ajuste polinómico de grado 3 a los tiempos por vuelta')
    plt.xlabel('Vuelta')
    plt.ylabel('Tiempo por vuelta (s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def exponencial():
    
    # Función exponencial
    def funcion_exponencial(x, a, b):
        return a * np.exp(b * x)

    # Función para ajustar una función exponencial a los datos de tiempos por vuelta
    def ajustar_exponencial(tiempos_por_vuelta):
        x = np.arange(len(tiempos_por_vuelta))
        parametros_optimos, cov_matrix = curve_fit(funcion_exponencial, x, tiempos_por_vuelta, p0=[1, 0.1])
        return parametros_optimos, cov_matrix

    # Función para calcular un único p-valor para la distribución general de los coeficientes
    def obtener_p_valor_general(tiempos_por_vuelta, parametros_optimos, cov_matrix, num_params):
        # Calcular el residuo cuadrático
        residuos = tiempos_por_vuelta - funcion_exponencial(np.arange(len(tiempos_por_vuelta)), *parametros_optimos)
        residuo_cuadratico = np.sum(residuos ** 2)
        # Calcular el residuo cuadrático reducido
        residuo_cuadratico_reducido = residuo_cuadratico / (len(tiempos_por_vuelta) - num_params)
        # Calcular la suma de cuadrados del modelo
        suma_cuadrados_modelo = np.sum((funcion_exponencial(np.arange(len(tiempos_por_vuelta)), *parametros_optimos) - np.mean(tiempos_por_vuelta)) ** 2)
        # Calcular la estadística F
        f_stat = (suma_cuadrados_modelo / num_params) / residuo_cuadratico_reducido
        # Calcular el p-valor
        p_valor = 1 - f.cdf(f_stat, num_params, len(tiempos_por_vuelta) - num_params)
        return p_valor

    # Datos de ejemplo: tiempos por vuelta
    tiempos_por_vuelta = np.array([103.25, 103.84, 104.69, 104.45, 105])  # Nuevos tiempos por vuelta en segundos

    # Ajuste de una función exponencial a los datos de tiempos por vuelta
    parametros_optimos, cov_matrix = ajustar_exponencial(tiempos_por_vuelta)

    # Número de parámetros en el modelo
    num_params = len(parametros_optimos)

    # Calcular el p-valor general para la distribución de los coeficientes
    p_valor_general = obtener_p_valor_general(tiempos_por_vuelta, parametros_optimos, cov_matrix, num_params)

    # Imprimir el resultado
    print(f"P-valor general del ajuste exponencial: {p_valor_general}")

    # Gráfico del ajuste exponencial
    x = np.arange(len(tiempos_por_vuelta))
    y_pred = funcion_exponencial(x, *parametros_optimos)

    plt.plot(x, tiempos_por_vuelta, 'o', label='Tiempos por vuelta')
    plt.plot(x, y_pred, label='Ajuste exponencial')
    plt.title('Ajuste exponencial a los tiempos por vuelta')
    plt.xlabel('Vuelta')
    plt.ylabel('Tiempo por vuelta (s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def logaritmo():


    # Definir la función logarítmica
    def funcion_logaritmica(x, a, b):
        return a * np.log(b * x)

    # Función para normalizar los datos
    def normalizar_datos(data):
        return (data - np.mean(data)) / np.std(data)

    # Función para desnormalizar los parámetros
    def desnormalizar_parametros(params, data):
        a_norm, b_norm = params
        mean_data = np.mean(data)
        std_data = np.std(data)
        a = a_norm * std_data
        b = b_norm / mean_data
        return a, b

    # Datos de tiempos por vuelta
    tiempos_por_vuelta = np.array([103.25, 103.84, 104.69, 104.45, 105])

    # Crear un arreglo para el eje x (vuelta)
    x = np.arange(1, len(tiempos_por_vuelta) + 1)

    # Normalizar los datos
    tiempos_norm = normalizar_datos(tiempos_por_vuelta)

    # Ajuste logarítmico con datos normalizados
    parametros_optimos_norm, _ = curve_fit(funcion_logaritmica, x, tiempos_norm)

    # Desnormalizar los parámetros
    parametros_optimos = desnormalizar_parametros(parametros_optimos_norm, tiempos_por_vuelta)

    # Generar los valores ajustados
    y_pred = funcion_logaritmica(x, *parametros_optimos)

    # Mostrar los parámetros del ajuste
    print("Parámetros del ajuste logarítmico:")
    print("a =", parametros_optimos[0])
    print("b =", parametros_optimos[1])

    # Graficar los resultados
    plt.scatter(x, tiempos_por_vuelta, label='Tiempos por vuelta reales')
    plt.plot(x, y_pred, color='red', label='Ajuste logarítmico')
    plt.title('Ajuste logarítmico a los tiempos por vuelta')
    plt.xlabel('Vuelta')
    plt.ylabel('Tiempo por vuelta (s)')
    plt.legend()
    plt.grid(True)
    plt.show()


    
logaritmo()