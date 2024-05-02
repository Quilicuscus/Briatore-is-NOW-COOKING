import csv

from strategy_gate import StrategyGate
from coche import Coche
from circuito import Circuito
from vuelta import Vuelta



class Carrera:
    """
    It's a motor race Toto, they went car racing

    Atributos:
    coche:Coche
        Coche para el que se diseñará la estrategia
    circuito:Circuito
        Circuito donde se realiza la carrera
    vueltas:list[Vuelta]
        Lista con los datos de las vueltas previas

    Métodos:
    leer_vueltas(archivo:str) -> list[Vuelta]
        Lee un CSV con datos de vueltas y devuelve una lista con objetos vuelta correspondientes
    vuelta() -> bool
        Simula una vuelta de carrera
    """
    def __init__(self, max_gas:float, tiempo_base:float, ganancia_goma:float, archivo_vueltas:str):
        """
        Construye la clase con los argumentos dados

        Argumentos:
        max_gas:float
            Capacidad del tanque de combustible del coche
        tiempo_base:float
            Tiempo sin tener en cuenta las condiciones de carrera del circuito
        ganancia_goma:float
            Tiempo ahorrado cada vuelta por coche y vuelta recorrida del circuito
        archivo_vueltas:str
            Archivo con los datos de vueltas previas
        """
        self.__coche = Coche(max_gas)
        self.__circuito = Circuito(tiempo_base, ganancia_goma)
        self.vueltas = self.__leer_vueltas(archivo_vueltas)


    def __leer_vueltas(self, archivo:str) -> list[Vuelta]:
        """
        Lee un archivo CSV con datos de vueltas y las guarda en una lista de objetos vuelta

        Argumentos:
        archivo:str
            Almacén con los datos de las vueltas

        Devuelve:
            Lista con objetos vuelta correspondientes
        """
        vueltas = []
        try:
            with open(archivo) as entrada:
                reader = csv.reader(entrada, delimiter = ",")
                next(reader)
                for fila in reader:
                    vueltas.append(Vuelta(float(fila[0]), float(fila[1]),
                                          int(fila[2]), int(fila[3])))
        except Exception as error:
            raise StrategyGate("Archivo CSV no válido") from error
        return vueltas


    def vuelta(self) -> bool:
        """
        Simula una vuelta de carrera

        Devuelve:
            Si quedan coches en la carrera
        """
        self.__coche.vuelta()
        return self.__circuito.vuelta()
