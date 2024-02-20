tiempo_vuelta_estandar = 100
tiempo_vuelta_actual = 0
tiempo_extra_por_vuelta = 1
vueltas = 10
tiempo_boxes = 20
tiempo_total = 0
tiempo_final = 100000000000
lista_de_tiempos = []
k = 1
numero_paradas = 0

for j in range(1, vueltas + 1):
    for i in range(1, vueltas + 1):
        if i == j:
            print("Nos encontramos en la vuelta %i " % (i))
            print("En esta vuelta paramos en boxes, por lo que sumamos 20 segundos")
            tiempo_vuelta_actual = (tiempo_vuelta_estandar + tiempo_extra_por_vuelta*(k-1)) + 20
            k = 1
            tiempo_total+=tiempo_vuelta_actual
            print("El tiempo de esta vuelta ha sido: %i " % (tiempo_vuelta_actual))
            print("El tiempo total por ahora de carrera es %i " % (tiempo_total))
        else:
            print("Esta vuelta no paramos en boxes")
            print("Nos encontramos en la vuelta %i " % (i))
            tiempo_vuelta_actual = tiempo_vuelta_estandar + tiempo_extra_por_vuelta*(k-1)
            print("El tiempo de esta vuelta ha sido: %i " % (tiempo_vuelta_actual))
            tiempo_total += tiempo_vuelta_actual
            print("El tiempo total por ahora de carrera es %i " % (tiempo_total))
            k+=1

    lista_de_tiempos.append(tiempo_total)
    tiempo_vuelta_estandar = 100
    tiempo_vuelta_actual = 0
    tiempo_total = 0
    k = 1


print(lista_de_tiempos)
tiempo_mas_pequeno = min(lista_de_tiempos)
numero_estrategia = lista_de_tiempos.index(tiempo_mas_pequeno)
print("Creemos que la mejor estrategia es la numero %i " %(numero_estrategia+1))

