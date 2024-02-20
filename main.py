tiempo_vuelta = 100
tiempo_vuelta_actual = 0
extra = 1
vueltas = 10
tiempo_boxes = 20
tiempo_total = 0
tiempo_final = 100000000000

numero_paradas = 0

for i in range (1, vueltas + 1):
    print("Nos encontramos en la vuelta %i " % (i))
    tiempo_vuelta_actual = tiempo_vuelta + extra*(i-1)
    print("El tiempo de esta vuelta ha sido: %i " % (tiempo_vuelta_actual))
    tiempo_total += tiempo_total + tiempo_vuelta_actual

if tiempo_total < tiempo_final:
    tiempo_final = tiempo_total

print(tiempo_final)


