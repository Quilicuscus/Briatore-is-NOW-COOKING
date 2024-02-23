from coche import Coche
from circuito import Circuito



class Carrera:
    """
    you need to reinstate the lap before, that's not right
    Esto es como el tablero del proyecto final de progra

    Atributos:
    coche: el coche para el que se diseñara la estrategia
    circuito: el circuito donde se realiza la carrera
    """

    def __init__(self, perdida_gas:float, perdida_ruedas:float, max_gas:int, tiempo_base:float,
                 ganancia_goma:float):
        self.coche = Coche(perdida_gas, perdida_ruedas, max_gas)
        self.circuito = Circuito(tiempo_base, ganancia_goma)
