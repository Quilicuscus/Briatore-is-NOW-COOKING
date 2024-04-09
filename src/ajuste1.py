import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f

vueltas = [94.961, 95.413, 94.682, 95.091, 94.880, 95.200, 95.418, 95.679, 95.887, 96.224]

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
    tiempos_por_vuelta = np.array(vueltas)  # Nuevos tiempos por vuelta en segundos

    # Ajuste de un polinomio de grado 2 a los datos de tiempos por vuelta
    coeficientes, cov_matrix = ajustar_polinomio(tiempos_por_vuelta)

    # Número de parámetros en el modelo (grado del polinomio + 1)
    num_params = len(coeficientes)

    # Calcular el p-valor general para la distribución de los coeficientes
    p_valor_general = obtener_p_valor_general(tiempos_por_vuelta, coeficientes, cov_matrix, num_params)

    # Imprimir el resultado
    print(f"P-valor general del ajuste polinomico de grado 2: {p_valor_general}")

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

def polinomio_grado2_prueba():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import f

    # Datos de ejemplo: tiempos por vuelta
    vueltas = [94.961, 95.413, 94.682, 95.091, 94.880, 95.200, 95.418, 95.679, 95.887, 96.224]

    def polinomio_grado2(num_vueltas_futuras):
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

        # Datos de tiempos por vuelta
        tiempos_por_vuelta = np.array(vueltas)  # Nuevos tiempos por vuelta en segundos

        # Ajuste de un polinomio de grado 2 a los datos de tiempos por vuelta
        coeficientes, cov_matrix = ajustar_polinomio(tiempos_por_vuelta)

        # Número de parámetros en el modelo (grado del polinomio + 1)
        num_params = len(coeficientes)

        # Calcular el p-valor general para la distribución de los coeficientes
        p_valor_general = obtener_p_valor_general(tiempos_por_vuelta, coeficientes, cov_matrix, num_params)

        # Calcular futuros tiempos por vuelta en función del ajuste obtenido
        x_futuro = np.arange(len(tiempos_por_vuelta), len(tiempos_por_vuelta) + num_vueltas_futuras)
        tiempos_futuros = np.polyval(coeficientes, x_futuro)

        # Imprimir el array con las vueltas existentes y las vueltas futuras
        vueltas_completas = np.concatenate((tiempos_por_vuelta, tiempos_futuros))
        print("Vueltas completas (existentes + futuras):")
        print(vueltas_completas)

        # Imprimir el p-valor del ajuste
        print(f"P-valor general del ajuste polinomico de grado 2: {p_valor_general}")

        # Gráfico del ajuste polinomial y futuros tiempos por vuelta
        x = np.arange(len(tiempos_por_vuelta))
        x_futuro = np.arange(len(tiempos_por_vuelta), len(tiempos_por_vuelta) + num_vueltas_futuras)
        y_pred = np.polyval(coeficientes, x)
        tiempos_futuros = np.polyval(coeficientes, x_futuro)

        plt.plot(np.concatenate((x, x_futuro)), np.concatenate((tiempos_por_vuelta, tiempos_futuros)), 'o', label='Tiempos por vuelta y futuros')
        plt.plot(np.concatenate((x, x_futuro)), np.concatenate((y_pred, tiempos_futuros)), label='Ajuste polinomial y futuros tiempos')
        plt.title('Ajuste polinomial de grado 2 y futuros tiempos por vuelta')
        plt.xlabel('Vuelta')
        plt.ylabel('Tiempo por vuelta (s)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Llamada a la función para calcular y mostrar futuros tiempos por vuelta para 3 vueltas adicionales
    polinomio_grado2(num_vueltas_futuras=3)


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
    tiempos_por_vuelta = np.array(vueltas)  # Nuevos tiempos por vuelta en segundos

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
    tiempos_por_vuelta = np.array(vueltas)

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

def logaritmo1():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import f

    # Función logarítmica
    def funcion_logaritmica(x, a, b):
        return a * np.log(b * x)

    # Función para ajustar una función logarítmica a los datos
    def ajustar_logaritmica(tiempos_por_vuelta):
        # Crear un arreglo para el eje x (vuelta)
        x = np.arange(1, len(tiempos_por_vuelta) + 1)

        # Transformar los datos logarítmicamente
        tiempos_log = np.log(tiempos_por_vuelta)

        # Estimaciones iniciales para los parámetros
        a_init = max(tiempos_log) - min(tiempos_log)  # Estimación inicial para 'a'
        b_init = 1.0  # Estimación inicial para 'b'

        # Ajustar una función logarítmica a los datos
        parametros_optimos, _ = curve_fit(funcion_logaritmica, x, tiempos_log, p0=[a_init, b_init], maxfev=10000)

        # Obtener los parámetros finales de la función logarítmica
        a = parametros_optimos[0]
        b = parametros_optimos[1]

        # Generar los valores ajustados
        y_pred = a * np.log(b * x)

        # Calcular el residuo cuadrático
        residuos = tiempos_log - funcion_logaritmica(x, *parametros_optimos)
        residuo_cuadratico = np.sum(residuos ** 2)

        # Calcular el residuo cuadrático reducido
        residuo_cuadratico_reducido = residuo_cuadratico / (len(tiempos_por_vuelta) - 2)

        # Calcular la suma de cuadrados del modelo
        suma_cuadrados_modelo = np.sum((funcion_logaritmica(x, *parametros_optimos) - np.mean(tiempos_log)) ** 2)

        # Calcular la estadística F
        f_stat = (suma_cuadrados_modelo / 2) / residuo_cuadratico_reducido

        # Calcular el p-valor
        p_valor = 1 - f.cdf(f_stat, 2, len(tiempos_por_vuelta) - 2)

        # Devolver los parámetros del ajuste y el p-valor
        return a, b, p_valor

    # Datos de tiempos por vuelta
    tiempos_por_vuelta = np.array(vueltas)  # Ejemplo con 5 tiempos por vuelta

    # Realizar el ajuste logarítmico
    a, b, p_valor = ajustar_logaritmica(tiempos_por_vuelta)

    # Mostrar los resultados
    print("Parámetros del ajuste logarítmico:")
    print("a =", a)
    print("b =", b)
    print("P-valor del ajuste logarítmico:", p_valor)

    # Graficar los resultados
    x = np.arange(1, len(tiempos_por_vuelta) + 1)
    y_pred = a * np.log(b * x)

    plt.scatter(x, tiempos_por_vuelta, label='Tiempos por vuelta reales')
    plt.plot(x, np.exp(y_pred), color='red', label='Ajuste logarítmico')
    plt.title('Ajuste logarítmico a los tiempos por vuelta')
    plt.xlabel('Vuelta')
    plt.ylabel('Tiempo por vuelta (s)')
    plt.legend()
    plt.grid(True)
    plt.show()


def nuevologaritmo():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import f

    # Función logarítmica
    def funcion_logaritmica(x, a, b):
        return a * np.log(b * x)

    # Función para ajustar una función logarítmica a los datos
    def ajustar_logaritmica(tiempos_por_vuelta):
        # Crear un arreglo para el eje x (vuelta)
        x = np.arange(1, len(tiempos_por_vuelta) + 1)

        # Transformar los datos logarítmicamente
        tiempos_log = np.log(tiempos_por_vuelta)

        # Estimaciones iniciales para los parámetros
        a_init = max(tiempos_log) - min(tiempos_log)  # Estimación inicial para 'a'
        b_init = 1.0  # Estimación inicial para 'b'

        # Ajustar una función logarítmica a los datos
        parametros_optimos, _ = curve_fit(funcion_logaritmica, x, tiempos_log, p0=[a_init, b_init], maxfev=10000)

        # Obtener los parámetros finales de la función logarítmica
        a = parametros_optimos[0]
        b = parametros_optimos[1]

        # Generar los valores ajustados
        y_pred = a * np.log(b * x)

        # Calcular el residuo cuadrático
        residuos = tiempos_log - funcion_logaritmica(x, *parametros_optimos)
        residuo_cuadratico = np.sum(residuos ** 2)

        # Calcular el residuo cuadrático reducido
        residuo_cuadratico_reducido = residuo_cuadratico / (len(tiempos_por_vuelta) - 2)

        # Calcular la suma de cuadrados del modelo
        suma_cuadrados_modelo = np.sum((funcion_logaritmica(x, *parametros_optimos) - np.mean(tiempos_log)) ** 2)

        # Calcular la estadística F
        f_stat = (suma_cuadrados_modelo / 2) / residuo_cuadratico_reducido

        # Calcular el p-valor
        p_valor = 1 - f.cdf(f_stat, 2, len(tiempos_por_vuelta) - 2)

        # Devolver los parámetros del ajuste y el p-valor
        return a, b, p_valor

    # Función para estimar tiempos de vuelta para vueltas futuras
    def estimar_tiempos_futuros(a, b, vueltas_futuras):
        x_futuras = np.arange(len(tiempos_por_vuelta) + 1, len(tiempos_por_vuelta) + 1 + vueltas_futuras)
        y_pred_futuras = a * np.log(b * x_futuras)
        tiempos_futuros = np.exp(y_pred_futuras)
        return tiempos_futuros

    # Datos de tiempos por vuelta
    tiempos_por_vuelta = np.array(vueltas)  # Ejemplo con 8 tiempos por vuelta
    vueltas_futuras = 5  # Número de vueltas a predecir

    # Realizar el ajuste logarítmico
    a, b, p_valor = ajustar_logaritmica(tiempos_por_vuelta)

    # Estimar los tiempos para las vueltas futuras
    tiempos_futuros = estimar_tiempos_futuros(a, b, vueltas_futuras)

    # Crear lista con tiempos originales y estimados para vueltas futuras
    tiempos_originales_y_estimados = list(tiempos_por_vuelta) + list(tiempos_futuros)

    # Mostrar los resultados
    print("Parámetros del ajuste logarítmico:")
    print("a =", a)
    print("b =", b)
    print("P-valor del ajuste logarítmico:", p_valor)
    print("Tiempos originales y estimados para las próximas", vueltas_futuras, "vueltas:")
    print(tiempos_originales_y_estimados)

    # Graficar los resultados
    x = np.arange(1, len(tiempos_por_vuelta) + 1)
    y_pred = a * np.log(b * x)

    plt.scatter(x, tiempos_por_vuelta, label='Tiempos por vuelta reales')
    plt.plot(x, np.exp(y_pred), color='red', label='Ajuste logarítmico')
    plt.plot(np.arange(len(tiempos_por_vuelta) + 1, len(tiempos_por_vuelta) + 1 + vueltas_futuras), tiempos_futuros, 'gx', label='Tiempos estimados para vueltas futuras')
    plt.title('Ajuste logarítmico y estimación de tiempos para futuras vueltas')
    plt.xlabel('Vuelta')
    plt.ylabel('Tiempo por vuelta (s)')
    plt.legend()
    plt.grid(True)
    plt.show()


polinomio_grado2()
polinomio_grado2_prueba()


