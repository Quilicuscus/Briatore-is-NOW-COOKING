import itertools
from itertools import permutations

# El tiempo por vuelta sin desgaste de neumáticos es 100 segundos por vuelta
lista_tiempos_base = [100.0, 101.0, 102.0]
# tiempo_base1 = 100
# tiempo_base2 = 101
# tiempo_base3 = 102

# El desgaste de neumaticos etc hace que el tiempo por vuelta aumente entre 
# 0,9 y 0,3 por vuelta en funcion del neumático
lista_extras = [0.9,0.6,0.3]
# extra_1 = 0.9
# extra_2 = 0.6
# extra_3 = 0.3

# Numero de vueltas de la carrera
vueltas = 30



# Número de paradas que pretendemos hacer durante la carrera
numero_paradas = 1

vueltas_desde_la_parada = 0
lista_de_tiempos_de_las_estretegias = []


# Combinatoria para encontrar todas las combinaciones de las vueltas de parada
# en boxes
# Generar las combinaciones del numero del 1 al 10
numeros = list(range(1, vueltas))

# lista_paradas contiene todas las tuplas con las vueltas en las que hay que 
# parar
lista_paradas = list(itertools.combinations(numeros, numero_paradas))

# Combinatoria para hallar todas las permutaciones de los neumáticos
numeros_neumaticos = [0, 1, 2] # 1 es blando, 2 es medio y 3 es duro
compuesto_neumaticos = ["blando", "medio", "duro"]


# neumaticos a poner contiene en que orden ponemos los neumaticos que 
# tenemos que poner en el coche

neumaticos_a_poner = list(permutations(numeros_neumaticos, numero_paradas+1))
ruedas = neumaticos_a_poner[0]

print(lista_paradas)
print(neumaticos_a_poner)
print(len(lista_paradas))
print(len(neumaticos_a_poner))
# Queremos por cada combinacion de vueltas en las que tenemos que parar en 
# boxes, se prueben todas las opciones de los 3 compuestos de neumaticos
# Por ejemplo: en lista_paradas tenemos (1,2)(1,3)(1,4).... Para cada una de 
# ellas tenemos que probar todas las permutaciones de neumaticos: (1,2)(2,1)...

for i in lista_paradas: # Para cada combinacion de vueltas de parada
    for j in neumaticos_a_poner: # Para cada permutacion de neumaticos a poner
        # Comienza la simulación de la carrera, así que ponemos las 
        # condiciones iniciales:

        tiempo_total = 0.0 # tiempo total de carrera

        tiempo_vuelta_actual = 0.0 # tiempo de la vuelta en la que estamos
        vueltas_desde_la_parada = 0 # Vueltas desde la ultima parada
        numero_parada = 1 # Paradas realizadas
        ruedas = j[0]
        #print(ruedas)

        for k in range(1,vueltas+1): # k tambien es la vuelta en la que estamos
            print("Nos encontramos en la vuelta %i" % (k))
            print("Esta vuelta llevamos el neumatico %s" % (compuesto_neumaticos[ruedas]))

            if k in i: # Si k está en la combinacion de paradas, hay que parar
                print("Hay que parar")
                neumatico_que_ponemos = j[numero_parada]
                print("ponemos el %s " % (compuesto_neumaticos[neumatico_que_ponemos]))
                

                tiempo_vuelta_actual = lista_tiempos_base[ruedas] + lista_extras[ruedas]*vueltas_desde_la_parada
                print("El tiempo de esta vuelta ha sido %f" % (tiempo_vuelta_actual))

                tiempo_total+=tiempo_vuelta_actual + 20
                print("Por ahora el tiempo de carrera es %f" % (tiempo_total))

                vueltas_desde_la_parada = 0
                numero_parada+=1

                ruedas = neumatico_que_ponemos

            else:
                print("Esta vuelta no hay que parar")
                tiempo_vuelta_actual = lista_tiempos_base[ruedas] + lista_extras[ruedas]*vueltas_desde_la_parada
                print("El tiempo de esta vuelta ha sido %f" % (tiempo_vuelta_actual))
                tiempo_total+=tiempo_vuelta_actual
                print("Por ahora el tiempo de carrera es %f" % (tiempo_total))
                vueltas_desde_la_parada +=1

        print("El tiempo total de carrera es %f " % (tiempo_total))
        lista_de_tiempos_de_las_estretegias.append(tiempo_total)


numero_mas_pequeno = min(lista_de_tiempos_de_las_estretegias)
print(numero_mas_pequeno)
# print(lista_paradas)

posiciones = []
for i in range(len(lista_de_tiempos_de_las_estretegias)):
    if lista_de_tiempos_de_las_estretegias[i] == numero_mas_pequeno:
        posiciones.append(i)
estrategia_final1=[]

contador = 0
for i in lista_paradas: 
    for j in neumaticos_a_poner:
        if contador in posiciones:
            estrategia_final1.append(i)
            estrategia_final1.append(j)
        contador+=1

print(estrategia_final1)
print(estrategia_final1)        
print(contador)
print(posiciones)










