class Coche:
    """
    FIAAAAAUUUUUUN

    Constantes:
    MAX_GAS:float
        Capacidad del tanque de combustible

    Atributos:
    litros_gas:float
        Litros en el tanque de combustible
    vida_ruedas:int
        Vueltas recorridas por las ruedas

    Métodos:
    vuelta() -> None
        Recorre una vuelta al circuito
    parar(gas:float, ruedas:bool) -> None
        Lleva el coche a boxes
    """
    def __init__(self, max_gas:float, litros_gas:float=-1, vida_ruedas:int=0):
        """
        Asigna los atributos indicados al coche

        Argumentos:
        max_gas:float
            Capacidad del tanque de combustible
        litros_gas:float
            Litros de combustible a la salida. Lleno por defecto
        vida_ruedas:int
            Vida de las ruedas a la salida. Nuevas por defecto
        """
        self.MAX_GAS = max_gas
        self.__litros_gas = max_gas if litros_gas < 0 else litros_gas
        self.__vida_ruedas = vida_ruedas


    def vuelta(self) -> None:
        """
        Recorre una vuelta al circuito
        Actualiza el combustible en depósito y la vida de las ruedas
        """
        self.__litros_gas -= 0 # TODO: safety car y tal
        self.__vida_ruedas += 1


    def parar(self, gas:float, ruedas:bool=True) -> None:
        """
        Lleva el coche a boxes
        Resetea la degradación de las ruedas y añade el combustible al coche

        Argumentos:
        gas:float
            Gas a repostar
        ruedas:bool
            Si se cambian ruedas en la parada
        """
        # Añadir combustible al tanque
        self.__litros_gas = \
            self.MAX_GAS if gas >= self.MAX_GAS - self.__litros_gas else self.__litros_gas + gas

        # Poner neumáticos nuevos
        self.__vida_ruedas = 0 if ruedas else self.__vida_ruedas
