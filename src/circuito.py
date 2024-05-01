class Circuito:
    """
    He should be black flagged for that, black flagged!

    Constantes:
    TIEMPO_BASE:float
        Tiempo en dar una vuelta sin considerar los estados de coche y pista
    GANANCIA_GOMA:float
        Tiempo ahorrado cada vuelta por coche en pista y vuelta recorrida

    Atributos:
    coches:int
        Número de coches en pista
    engomado:float
        Ganancia de tiempo por goma acumulada

    Métodos:
    vuelta(abandonos:int) -> bool
        Simula una vuelta de carrera
    """
    def __init__(self, tiempo_base:float, ganancia_goma:float, coches:int=1, engomado:float=-1):
        """
        Asigna los atributos indicados al circuito

        Argumentos:
        tiempo_base:float
            Tiempo hipotético con depósito vacío, gomas nuevas y sin engomado
        ganancia_goma:float
            Ganancia de tiempo por coche en circuito y vuelta recorrida
        coches:int
            Número de coches en el circuito
        engomado:float
            Ganancia de tiempo por goma actual
        """
        self.TIEMPO_BASE = tiempo_base
        self.GANANCIA_GOMA = ganancia_goma
        self.__coches = coches
        self.__engomado = 0 if engomado < 0 else engomado


    def vuelta(self, abandonos:int=0) -> bool:
        """
        Simula una vuelta de carrera
        Aumenta el engomado y decrementa el número de coches en pista

        Argumentos:
        abandonos:int
            Coches que han abandonado en la última vuelta

        Devuelve:
            Si quedan coches en la carrera
        """
        self.__engomado += self.GANANCIA_GOMA * self.__coches
        self.__coches = 0 if abandonos >= self.__coches else self.__coches - abandonos
        return bool(self.__coches)
