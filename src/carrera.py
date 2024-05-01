from coche import Coche
from circuito import Circuito



class Carrera:
    """
    you need to reinstate the lap before, that's not right

    Atributos:
    coche:
        Coche para el que se diseñará la estrategia
    circuito:
        Circuito donde se realiza la carrera

    Métodos:
    vuelta() -> bool
        Simula una vuelta de carrera
    """
    def __init__(self, max_gas:float, tiempo_base:float, ganancia_goma:float):
        """
        Construye la clase con los argumentos dados

        Argumentos:
        max_gas:float
            Capacidad del tanque de combustible del coche
        tiempo_base:float
            Tiempo sin tener en cuenta las condiciones de carrera del circuito
        ganancia_goma:float
            Tiempo ahorrado cada vuelta por coche y vuelta recorrida del circuito
        """
        self.__coche = Coche(max_gas)
        self.__circuito = Circuito(tiempo_base, ganancia_goma)


    def vuelta(self) -> bool:
        """
        Simula una vuelta de carrera

        Devuelve:
            Si quedan coches en la carrera
        """
        self.__coche.vuelta()
        return self.__circuito.vuelta()
