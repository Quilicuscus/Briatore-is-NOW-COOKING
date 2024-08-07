class Coche:
    """
    FIAAAAAUUUUUUN

    Constantes:
    PERDIDA_GAS: tiempo extra en recorrer cada vuelta por cada litro de combustible en el depósito
    PERDIDA_RUEDAS: tiempo extra en recorrer cada vuelta por cada vuelta recorrida con las ruedas
    MAX_GAS: capacidad del tanque de combustible

    Atributos:
    litros_gas: litros en el tanque de combustible en un momento dado
    vida_ruedas: vueltas recorridas por las ruedas en un momento dado - Sustituir por clase Ruedas
    - Consumo gas??
    Métodos:
    peso_gas(): devuelve la pérdida de tiempo en una vuelta a causa del peso del combustible actual
    degradacion(): devuelve la pérdida de tiempo en una vuelta a causa de la degradación de las gomas
    parar(): lleva el coche a boxes
    """

    def __init__(self, perdida_gas:float, perdida_ruedas:float, max_gas:int, litros_gas:float=-1,
                 vida_ruedas:int=0):
        self.PERDIDA_GAS = perdida_gas
        self.PERDIDA_RUEDAS = perdida_ruedas
        self.MAX_GAS = max_gas

        self.__litros_gas = max_gas if litros_gas < 0 else litros_gas
        self.__vida_ruedas = vida_ruedas
        #Añadir ruedas


    #def vuelta(self,....)
    #def update(meter nuevos atributos con csv - para cambiar en mitad de carrera?)

    def peso_gas(self) -> float:
        """Devuelve la pérdida de tiempo esperada durante una vuelta causada por el peso del
        combustible en dicha vuelta"""
        return self.PERDIDA_GAS * self.__litros_gas

    def degradacion(self) -> float:
        """Devuelve la pérdida de tiempo esperada en una vuelta a causa de la degradación de las
        gomas del coche en dicha vuelta"""
        return self.PERDIDA_RUEDAS * self.__vida_ruedas #Clase rueda??

    def parar(self, tiempo_pit:float, gas:float, ruedas:bool=True, repairs:float=0) -> float:
        """Este método simula una parada en boxes. Calcula el tiempo de la parada en base al tiempo de
        recorrer el pitlane, el combustible a repostar, si se cambian las ruedas y las posibles
        reparaciones. También resetea la degradación de las ruedas y añade el combustible al coche"""
        ##Fast repair? - A lo mejor hay que hacer algo con el tiempo de la vuelta
        # Añadir combustible al tanque
        if self.__litros_gas + gas >= self.MAX_GAS:
            self.__litros_gas = self.MAX_GAS
        else:
            self.__litros_gas += gas

        # Poner neumáticos nuevos
        self.__vida_ruedas = 0

        #Habria que hacer return del tiempo de la parada - tiempo/litro repostado o algo??
