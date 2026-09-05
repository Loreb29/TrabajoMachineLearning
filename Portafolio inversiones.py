import random

# Datos de los proyectos
costos = [20000, 35000, 15000, 40000, 25000,
          30000, 10000, 45000, 28000, 18000]

retornos = [32000, 50000, 24000, 58000, 45000,
            42000, 18000, 65000, 48000, 30000]

presupuesto = 150000
penalizacion = 10


def decodificar(individuo):
    proyectos = []

    for i, gen in enumerate(individuo):
        if gen == 1:
            proyectos.append(i + 1)

    return proyectos


def calcular_costo(individuo):
    return sum(
        individuo[i] * costos[i]
        for i in range(len(individuo))
    )


def calcular_retorno(individuo):
    return sum(
        individuo[i] * retornos[i]
        for i in range(len(individuo))
    )


def calcular_fitness(individuo):
    costo = calcular_costo(individuo)
    retorno = calcular_retorno(individuo)

    if costo <= presupuesto:
        return retorno
    else:
        exceso = costo - presupuesto
        return retorno - (penalizacion * exceso)


# Crear un individuo aleatorio
individuo = [random.randint(0, 1) for _ in range(10)]

proyectos = decodificar(individuo)
costo = calcular_costo(individuo)
retorno = calcular_retorno(individuo)
fitness = calcular_fitness(individuo)

print("Genotipo:", individuo)
print("Proyectos seleccionados:", proyectos)
print("Costo total:", costo)
print("Retorno total:", retorno)
print("Fitness:", fitness)