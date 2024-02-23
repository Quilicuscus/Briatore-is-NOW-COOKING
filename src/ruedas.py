class Ruedas:
    """
    box now box for hard STAY OUT STAY OUT

    Constantes:
    DEGRADACION: pérdida de tiempo por compuesto y vuelta recorrida

    Atributos:
    compuesto: tipo de compuesto de los neumáticos
    vida: número de vueltas recorridas por las ruedas
    """

    def __init__(self, degradacion:list[float], compuesto:int, vida:int=0):
        self.DEGRADACION = degradacion
        self.__compuesto = compuesto
        self.__vida = vida
