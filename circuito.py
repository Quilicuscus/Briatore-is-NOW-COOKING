class Circuito:
    """
    Constantes:
    TIEMPO_BASE: tiempo en dar una vuelta sin considerar degradación y peso del combustible
    GANANCIA_GOMA: tiempo ahorrado por vuelta por cada unidad de engomado

    Atributos:
    engomado: la cantidad de goma en el circuito
    vueltas: el número de vueltas dadas en el circuito
    coches: los coches en el circuito. Es float ya que a las categorías se les dan pesos distintos
    """

    def __init__(self, tiempo_base:float, ganancia_goma:float, vueltas:int=0, coches:float=1, engomado:float=-1):
        self.TIEMPO_BASE = tiempo_base
        self.GANANCIA_GOMA = ganancia_goma

        self.engomado = 0 if engomado < 0 else engomado #vueltas*coches?
        self.__vueltas = vueltas
        self.__coches = coches
