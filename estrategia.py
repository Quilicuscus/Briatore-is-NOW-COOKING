import itertools

# El tiempo por vuelta sin desgaste de neumáticos es 100 segundos por vuelta
tiempo_base = 100

# El desgaste de neumaticos etc hace que el tiempo por vuelta aumente un segundo por vuelta
extra = 1

# Numero de vueltas de la carrera
vueltas = 20

# Número de paradas que pretendemos hacer durante la carrera
numero_paradas = 2

vueltas_desde_la_parada = 1
lista_de_tiempos_de_estretegias = []


# Generar los números del 1 al 10
numeros = list(range(1, vueltas+1))

# Generar todas las combinaciones de dos números para así poder todas las posibles dos paradas
combinaciones = list(itertools.combinations(numeros, numero_paradas))

# Imprimir todas las combinaciones
for combinacion in combinaciones:
    for i in combinacion:
        print(i)
    print(combinacion)

for combinacion in combinaciones:
    vuelta_actual = 1
    tiempo_total = 0
    tiempo_vuelta_actual = 0
    vueltas_desde_la_parada = 1

    for i in range(1,vueltas+1):
        print("Nos encontramos en la vuelta %i" % (vuelta_actual))

        if i in combinacion:

            print("Esta vuelta paramos en boxes")
            tiempo_vuelta_actual = tiempo_base + extra*(vueltas_desde_la_parada-1)
            # Añadimos 20 segundos de la parada
            tiempo_vuelta_actual += 20
            print("El tiempo de esta vuelta ha sido: %i" % (tiempo_vuelta_actual))
            tiempo_total += tiempo_vuelta_actual
            print("Por ahora el tiempo de carrera es %i" % (tiempo_total))

            vueltas_desde_la_parada = 1
            vuelta_actual += 1

        else:
            print("Esta vuelta no paramos en boxes")
            tiempo_vuelta_actual = tiempo_base + extra*(vueltas_desde_la_parada-1)
            print("El tiempo de esta vuelta ha sido %i" % (tiempo_vuelta_actual))
            tiempo_total+=tiempo_vuelta_actual
            print("Por ahora el tiempo de carrera es %i" % (tiempo_total))
            vueltas_desde_la_parada+=1
            vuelta_actual+=1

    print("El tiempo total de carrera ha sido %i" % (tiempo_total))
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










