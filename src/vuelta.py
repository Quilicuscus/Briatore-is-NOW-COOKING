class Vuelta():
    """
    you need to reinstate the lap before, that's not right

    Atributos:
    tiempo:float
        Tiempo de la vuelta
    litros_gas:float
        Litros en el depósito al inicio de la vuelta
    vida_ruedas:int
        Vida de las ruedas al inicio de la vuelta
    engomado:int
        Goma en el circuito (coches * vueltas) al inicio de la vuelta
    """
    def __init__(self, tiempo:float, litros_gas:float, vida_ruedas:int, engomado:int):
        """
        Construye el objeto vuelta con los atributos correspondientes

        Argumentos:
        tiempo:float
            Tiempo de la vuelta
        litros_gas:float
            Litros en el depósito al inicio de la vuelta
        vida_ruedas:int
            Vida de las ruedas al inicio de la vuelta
        engomado:int
            Goma en el circuito (coches * vueltas) al inicio de la vuelta
        """
        self.__tiempo = tiempo
        self.__litros_gas = litros_gas
        self.__vida_ruedas = vida_ruedas
        self.__engomado = engomado


    def __str__(self) -> str:
        """
        Convierte a la vuelta en string

        Devuelve:
            Vuelta en formato string
        """
        return "Tiempo: " + str(self.__tiempo) + "s\n" + \
               "Depósito: " + str(self.__litros_gas) + "l\n" + \
               "Ruedas: " + str(self.__vida_ruedas) + "vts\n" + \
               "Engomado: " + str(self.__engomado) + " uds"
