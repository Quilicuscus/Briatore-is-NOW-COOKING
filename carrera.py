from coche import Coche
from circuito import Circuito



class Carrera:
    """Esto es como el tablero del proyecto final de progra"""

    def __init__(self, perdida_gas:float, perdida_ruedas:float, max_gas:int, tiempo_base:float,
                 ganancia_goma:float):
        self.coche = Coche(perdida_gas, perdida_ruedas, max_gas)
        self.circuito = Circuito(tiempo_base, ganancia_goma)
