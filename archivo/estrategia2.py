import itertools
from itertools import permutations

# El tiempo por vuelta sin desgaste de neumáticos es 100 segundos por vuelta
tiempo_base1 = 100
tiempo_base2 = 101
tiempo_base3 = 102

# El desgaste de neumaticos etc hace que el tiempo por vuelta aumente un segundo por vuelta
extra_1 = 0.9
extra_2 = 0.6
extra_3 = 0.3

# Numero de vueltas de la carrera
vueltas = 20

# Número de paradas que pretendemos hacer durante la carrera
numero_paradas = 2

vueltas_desde_la_parada = 1
lista_de_tiempos_de_estretegias = []



# Combinatoria para encontrar todas las combinaciones de las vueltas de
# parada en boxes
# Generar los números del 1 al 10
numeros = list(range(1, vueltas+1))
# Generar todas las combinaciones de dos números para así poder todas las posibles dos paradas
combinaciones = list(itertools.combinations(numeros, numero_paradas))
# Imprimir todas las combinaciones
# for combinacion in combinaciones:
    # for i in combinacion:
        # print(i)
    # print(combinacion)

# Combinatoria para hallar todas las  permutaciones de los neumáticos
numeros_neumaticos = [1, 2, 3]
permutaciones = permutations(numeros_neumaticos, numero_paradas)
# for permutacion in permutaciones:
    # print(permutacion)


for combinacion in combinaciones:
    for permutacion in permutaciones:
        neumatico = 0
        vuelta_actual = 1
        tiempo_total = 0
        tiempo_vuelta_actual = 0
        vueltas_desde_la_parada = 1
        ruedas = 2 #Compuesto de neumaticos que lleva el coche

        for i in range(1,vueltas+1):
            # print("Nos encontramos en la vuelta %i" % (vuelta_actual))

            if i in combinacion:

                # print("Esta vuelta paramos en boxes")
                if neumatico == 0:
                    # print ("Esta es la primera parada")
                    a = permutacion[neumatico]
                    if a == 1:
                        # print("Ponemos el neumatico blando")

                        if ruedas == 1:
                            # print("Hasta ahora estábamos con el neumatico blando")
                            tiempo_vuelta_actual = tiempo_base1 + extra_1*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 1, el blando")
                            ruedas = 1
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                        if ruedas == 2:
                            # print("Hasta ahora estabamos con el neumatico medio")
                            tiempo_vuelta_actual = tiempo_base2 + extra_2*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 1, el blando")
                            ruedas = 1
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1


                        if ruedas == 3:
                            # print("Hasta ahora estabamos con el nuematico duro")
                            tiempo_vuelta_actual = tiempo_base3 + extra_3*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 1, el blando")
                            ruedas = 1
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                    elif a == 2:
                        # print("Ponemos el neumatico medio")
                        if ruedas == 1:
                            # print("Hasta ahora estábamos con el neumatico blando")
                            tiempo_vuelta_actual = tiempo_base1 + extra_1*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 2, el medio")
                            ruedas = 2
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                        if ruedas == 2:
                            # print("Hasta ahora estabamos con el neumatico medio")
                            tiempo_vuelta_actual = tiempo_base2 + extra_2*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 2, el medio")
                            ruedas = 2
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                        if ruedas == 3:
                            # print("Hasta ahora estabamos con el nuematico duro")
                            tiempo_vuelta_actual = tiempo_base3 + extra_3*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 1, el medio")
                            ruedas = 2
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                    elif a == 3:
                        # print("Ponemos el neumático duro")
                        if ruedas == 1:
                            # print("Hasta ahora estábamos con el neumatico blando")
                            tiempo_vuelta_actual = tiempo_base1 + extra_1*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 3, el duro")
                            ruedas = 3
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                        if ruedas == 2:
                            # print("Hasta ahora estabamos con el neumatico medio")
                            tiempo_vuelta_actual = tiempo_base2 + extra_2*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 3, el duro")
                            ruedas = 1
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

                        if ruedas == 3:
                            # print("Hasta ahora estabamos con el nuematico duro")
                            tiempo_vuelta_actual = tiempo_base3 + extra_3*(vueltas_desde_la_parada-1)
                            tiempo_vuelta_actual+=20
                            # print("El tiempo de esta vuelta ha sido: %i" %(tiempo_vuelta_actual))
                            # print("Vamos a montar el neumatico 3, el duro")
                            ruedas = 1
                            tiempo_total += tiempo_vuelta_actual
                            # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                            vueltas_desde_la_parada = 1
                            vuelta_actual += 1
                            neumatico+=1

            else:

                # print("Esta vuelta no paramos en boxes")

                if ruedas == 1:
                    # print("Esta vuelta la hemos hecho con neumaticos blandos")
                    tiempo_vuelta_actual = tiempo_base1 + extra_1*(vueltas_desde_la_parada-1)
                    # print("El tiempo de esta vuelta ha sido %i" % (tiempo_vuelta_actual))
                    tiempo_total+=tiempo_vuelta_actual
                    # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                    vueltas_desde_la_parada+=1
                    vuelta_actual+=1

                elif ruedas == 2:
                    # print("Esta vuelta la hemos hecho con neumaticos medios")
                    tiempo_vuelta_actual = tiempo_base2 + extra_2*(vueltas_desde_la_parada-1)
                    # print("El tiempo de esta vuelta ha sido %i" % (tiempo_vuelta_actual))
                    tiempo_total+=tiempo_vuelta_actual
                    # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                    vueltas_desde_la_parada+=1
                    vuelta_actual+=1

                elif ruedas == 3:
                    # print("Esta vuelta la hemos hecho con el neumatico duro")
                    tiempo_vuelta_actual = tiempo_base3 + extra_3*(vueltas_desde_la_parada-1)
                    # print("El tiempo de esta vuelta ha sido %i" % (tiempo_vuelta_actual))
                    tiempo_total+=tiempo_vuelta_actual
                    # print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
                    vueltas_desde_la_parada+=1
                    vuelta_actual+=1

        # print("El tiempo total de carrera ha sido %i" % (tiempo_total))
        lista_de_tiempos_de_estretegias.append(tiempo_total)

print(lista_de_tiempos_de_estretegias)
estrategias = []

# Obtenemos el tiempo de carreras mas corto
numero_mas_pequeno = min(lista_de_tiempos_de_estretegias)

# Obtenemos la estrategia que genera ese tiempo de carrera mas corto
numero_de_estrategia = lista_de_tiempos_de_estretegias.index(numero_mas_pequeno)

# Queremos obtener todas las estrategias que obtengan el mismo tiempo mínimo, aunque sea parando en vueltas distintas:
estrategias_finales=[]

for i in range(len(lista_de_tiempos_de_estretegias)):
    if lista_de_tiempos_de_estretegias[i] == numero_mas_pequeno:
        estrategias_finales.append(combinaciones[i])

print("El tiempo de carrera mas corto posibles es: %i" % (numero_mas_pequeno))
print("Esto se consigue con la estrategia numero %i que es la siguiente: " % (numero_de_estrategia))
print(combinaciones[numero_de_estrategia])
print("Estas son otras estrategias que devuelven el mismo tiempo final de carrera: ")
print(estrategias_finales)
