# Practica Algoritmos geneticos

## Ejercicio 1 - Portafolio de Inversiones
Se utilizó un algoritmo genético para seleccionar un portafolio de inversiones mediante un genotipo binario. Cada posición del cromosoma representa un proyecto, donde 1 indica que el proyecto es seleccionado y 0 que no lo es.
Su objetivo es maximizar el retorno esperado de las inversiones, teniendo como restricción que el costo total del protafolio no puede superar el presupuesto máximo establecido.
## Ejercicio 2 - Selección de Personal Estricta
En este ejercicio se realizo el uso de la estructura ya creada por el ejercicio de la mochila, sin necesidad de hacer cambios al código fuente del algoritmo genético, aunque con una función de aptitud propia, bajo el funcionamiento siempre se eligen 5 candidatos finales de una población base de 12, la convergencia siempre suele ser de alrededor de 44 de aptitud.
## Ejercicio 3 - Operador de Cruzamiento de Dos Puntos
En este ejercicio se uso el ejemplo del problema de la mochila como base, y allí se creo la función ```python _crossover_two_point```, la cuál permite el cruzamiento de 2 puntos, esto lo hace mediante 2 numeros aleatorios, con un rango de 1 a la longitud del cromosoma-1, en el caso de que los números sean iguales, se itera hasta que se arregle esto, dependiendo de cual de los 2 puntos es mayor se usa una logica de mezcla levemente distinta (cambiando el orden), la mezcla realizada es una intercalada, donde el hijo toma la primera parte del padre, la segunda de la madre y la tercera del padre, ejemplo:

Puntos: 3, 6

Padre: 100|101|1

Madre: 011|001|0

Hijo1: 100|001|1

Hijo2: 011|101|0

```python
    def _crossover_two_point(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        #Se sigue usando probabilidad para calcular si hay cruzamiento o no
        if random.random() < self.pc:
            # Dos puntos de cruzamiento aleatorios
            point = random.randint(1, self.chromosome_length - 1)
            point2 = random.randint(1, self.chromosome_length - 1)
            #Si son el mismo numero, se recalculará hasta que sean distintos
            if point==point2:
                while point==point2:
                    point2 = random.randint(1, self.chromosome_length - 1)

            # Dependiendo de si uno es mayor que otro, cambia el orden de mezcla
            if point<point2:
                child1 = parent1[:point] + parent2[point:point2] + parent1[point2:]
                child2 = parent2[:point] + parent1[point:point2] + parent1[point2:]
            else:
                child1 = parent1[:point2] + parent2[point2:point] + parent1[point:]
                child2 = parent2[:point2] + parent1[point2:point] + parent1[point:]
            return child1, child2
        else:
            # Si no hay cruzamiento, los hijos son copias idénticas (deep copy) de los padres.
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
```
## Ejercicio 4 - Análisis de resultados
Para controlar las soluciones que incumplen esta restricción se implementó una función de penalización. Cuando el costo total supera el presupuesto, se resta una cantidad proporcional al exceso:

Fitness = Retorno − Factor de penalización × Exceso de presupuesto

### Penalización fuerte (factor = 10)

Acá el castigo es grande. En la simulación, el mejor individuo de cada generación siempre terminó siendo uno que sí cumplía el presupuesto (238000 de fitness, estable desde la generación 6 en adelante).

Sí seguían apareciendo individuos que se pasaban del presupuesto (por la mutación), entre 6 y 13 por generación. Pero como el castigo era tan fuerte, esos individuos quedaban con fitness muy bajo. Entonces casi no los elegía el torneo para reproducirse, y con el tiempo se iban quedando atrás, con una penalización alta, las soluciones malas mueren rápido y no logran pasar a las siguientes generaciones.

### Penalización suave (factor = 2)

Acá el castigo es pequeño. En la simulación, el mejor individuo de cada generación fue uno que sí se pasaba del presupuesto, y aun así tenía mejor fitness (245000) que cualquier solución que sí cumplía.

Esto pasó porque el castigo de 2 no alcanzaba a bajarle tanto el puntaje. Entonces le seguía convieniendo tener más retorno aunque se pasara un poco del presupuesto.

Por esta razón, en casi todas las generaciones había entre 15 y 23 individuos que no cumplían el presupuesto, y no bajaba con el tiempo como en el otro caso. Osea que esos individuos "malos" lograban sobrevivir varias generaciones sin problema.
