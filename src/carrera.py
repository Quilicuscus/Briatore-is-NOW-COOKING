from coche import Coche
from circuito import Circuito



class Carrera:
    """
    you need to reinstate the lap before, that's not right
    Esto es como el tablero del proyecto final de progra

    Atributos:
    coche: el coche para el que se diseñara la estrategia
    circuito: el circuito donde se realiza la carrera

    Métodos:
    tiempo_vuelta(): devuelve el tiempo de una vuelta al circuito en base a las condiciones presentes
    """

    def __init__(self, perdida_gas:float, perdida_ruedas:float, max_gas:int, tiempo_base:float,
                 ganancia_goma:float):
        self.__coche = Coche(perdida_gas, perdida_ruedas, max_gas)
        self.__circuito = Circuito(tiempo_base, ganancia_goma)


    def tiempo_vuelta(self) -> float:
        """Calcula el tiempo de una vuelta en base al tiempo de vuelta base, la goma en el circuito,
        el combustible en el tanque y la degradación de las ruedas."""
        return self.__circuito.TIEMPO_BASE - self.__circuito.engomado + self.__coche.peso_gas() + self.__coche.degradacion()
