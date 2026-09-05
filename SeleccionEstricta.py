import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

# Establecer una semilla para reproducibilidad (opcional)
random.seed(42)
np.random.seed(42)
class AlgoritmoGenetico:
    """
    Implementación genérica de un Algoritmo Genético Canónico.

    Atributos:
        population_size (int): Número de individuos en la población.
        chromosome_length (int): Longitud del cromosoma (número de bits/genes).
        pc (float): Probabilidad de cruzamiento (crossover).
        pm (float): Probabilidad de mutación.
        elitism (bool): Si se aplica elitismo (preservar al mejor individuo).
        fitness_func (callable): Función que calcula la aptitud de un fenotipo.
        decode_func (callable): Función que decodifica un genotipo a un fenotipo.
        selection_method (str): Método de selección ('roulette' o 'tournament').
        tournament_size (int, opcional): Tamaño del torneo para la selección por torneo.

    Principios Bioinspirados en el código:
    *   **Población**: Lista de `individuos`.
    *   **Individuo / Cromosoma**: Representado como una lista de enteros (0 o 1).
    *   **Gen / Alelo**: Cada entero (0 o 1) en la lista del cromosoma.
    *   **Locus**: El índice de cada gen dentro del cromosoma.
    """

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        pc: float,
        pm: float,
        fitness_func: callable,
        decode_func: callable,
        selection_method: str = 'roulette',
        elitism: bool = True,
        tournament_size: int = 3,
    ):
        # Validación de parámetros para asegurar que son coherentes y evitan errores lógicos.
        if not (0 <= pc <= 1 and 0 <= pm <= 1):
            raise ValueError("pc y pm deben estar entre 0 y 1.")
        if population_size <= 0 or chromosome_length <= 0:
            raise ValueError("population_size y chromosome_length deben ser positivos.")
        if selection_method not in ['roulette', 'tournament']:
            raise ValueError("selection_method debe ser 'roulette' o 'tournament'.")

        # Asignación de parámetros a atributos de la instancia.
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.pc = pc
        self.pm = pm
        self.elitism = elitism
        self.fitness_func = fitness_func
        self.decode_func = decode_func
        self.selection_method = selection_method
        self.tournament_size = tournament_size

        # Atributos para almacenar el estado del algoritmo genético.
        self.population: list[list[int]] = [] # La población actual de individuos (genotipos binarios).
        self.max_fitness_history: list[float] = [] # Historial del fitness máximo por generación.
        self.avg_fitness_history: list[float] = [] # Historial del fitness promedio por generación.
        self.best_individual_genotype: list[int] = [] # El mejor genotipo encontrado hasta ahora.
        self.best_individual_fitness: float = -float('inf') # El fitness del mejor genotipo encontrado hasta ahora.

    def _initialize_population(self) -> None:
        """
        Inicializa la población con cromosomas binarios generados aleatoriamente.
        Corresponde a la fase inicial de generación de individuos aleatorios en una población.
        Cada individuo es una lista de bits (0 o 1) de longitud `chromosome_length`.
        """
        self.population = [
            [random.randint(0, 1) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

    def _calculate_all_fitness(self, population: list[list[int]]) -> tuple[list[float], list]:
        """
        Calcula la aptitud (fitness) para cada individuo en la población.
        Para cada cromosoma (genotipo), primero lo decodifica a su representación real (fenotipo),
        y luego evalúa la aptitud de ese fenotipo utilizando `fitness_func`.

        Args:
            population (list[list[int]]): La población actual de genotipos.

        Returns:
            tuple[list[float], list]: Una tupla que contiene la lista de valores de aptitud
                                     y la lista de fenotipos decodificados.
        Principios Bioinspirados en el código:
        *   **Función de Aptitud (Fitness)**: Cuantifica la 'calidad' del individuo.
        *   **Fenotipo**: Se obtiene del genotipo para evaluar la aptitud.
        """
        fitness_values = []
        phenotypes = []
        for chromosome in population:
            phenotype = self.decode_func(chromosome) # Decodificación e_tilde: S -> X (genotipo a fenotipo)
            fitness = self.fitness_func(phenotype) # Evaluación del fenotipo
            fitness_values.append(fitness)
            phenotypes.append(phenotype)
        return fitness_values, phenotypes

    def _select_proportional(self, population: list[list[int]], fitness_values: list[float]) -> [list[int]]:
        """
        Realiza la selección de individuos utilizando el método de la Ruleta de Holland.
        Individuos con mayor aptitud tienen una mayor probabilidad de ser seleccionados para la siguiente generación.
        Maneja valores de aptitud no negativos mediante un desplazamiento si es necesario para evitar probabilidades negativas.

        Args:
            population (list[list[int]]): La población actual.
            fitness_values (list[float]): Los valores de aptitud correspondientes a cada individuo.

        Returns:
            list[list[int]]: La nueva población (pool de apareamiento) después de la selección, con el mismo tamaño que la población original.

        Principios Bioinspirados en el código:
        *   **Selección Natural**: Individuos con mayor aptitud tienen mayor probabilidad de ser seleccionados.
        *   **Ruleta de Holland**: La probabilidad de selección P(b_i) = f(b_i) / sum(f_k).
        """
        # Manejo de aptitudes no negativas: desplazar si el mínimo es negativo.
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            # Si hay aptitudes negativas, las desplazamos para que sean todas >= 0.
            # Se añade un pequeño valor (1e-6) para evitar divisiones por cero si todas las aptitudes ajustadas fueran 0.
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        else:
            # Si todas son no negativas, se usan directamente.
            adjusted_fitness = fitness_values

        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            # Si todas las aptitudes son cero (después del ajuste), seleccionar aleatoriamente para evitar errores.
            return random.choices(population, k=self.population_size)

        # Calcular las probabilidades de selección proporcionales a la aptitud ajustada.
        selection_probabilities = [f / total_fitness for f in adjusted_fitness]

        # La selección crea el 'pool de apareamiento' para la próxima generación.
        new_population = random.choices(population, weights=selection_probabilities, k=self.population_size)
        return new_population

    def _select_tournament(self, population: list[list[int]], fitness_values: list[float]) -> list[list[int]]:
        """
        Realiza la selección de individuos utilizando el método por Torneo.
        En cada paso, se seleccionan aleatoriamente `tournament_size` individuos y el de mayor aptitud entre ellos es elegido.
        Este proceso se repite `population_size` veces para formar la nueva población.

        Args:
            population (list[list[int]]): La población actual.
            fitness_values (list[float]): Los valores de aptitud correspondientes.

        Returns:
            list[list[int]]: La nueva población (pool de apareamiento) después de la selección.

        Principios Bioinspirados en el código:
        *   **Selección Natural**: Individuos compiten; los 'más fuertes' (mayor aptitud) prevalecen.
        """
        new_population = []
        for _ in range(self.population_size):
            # Seleccionar 'tournament_size' individuos aleatoriamente para el torneo.
            tournament_contestants_indices = random.sample(range(self.population_size), self.tournament_size)

            # Encontrar el mejor individuo (el de mayor fitness) del torneo.
            best_contestant_index = tournament_contestants_indices[0]
            for i in tournament_contestants_indices:
                if fitness_values[i] > fitness_values[best_contestant_index]:
                    best_contestant_index = i
            # El ganador del torneo se añade a la nueva población (se realiza una copia profunda para evitar referencias).
            new_population.append(copy.deepcopy(population[best_contestant_index]))
        return new_population

    def _crossover_one_point(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        Realiza el cruzamiento de un punto entre dos padres para producir dos hijos.
        Con una probabilidad `pc`, se elige un punto de corte aleatorio y los segmentos de los padres se intercambian.
        Si no se produce cruzamiento, los hijos son copias de los padres.

        Args:
            parent1 (list[int]): Genotipo del primer padre.
            parent2 (list[int]): Genotipo del segundo padre.

        Returns:
            tuple[list[int], list[int]]: Una tupla con los genotipos de los dos hijos resultantes.

        Principios Bioinspirados en el código:
        *   **Recombinación / Cruzamiento**: Intercambio de material genético entre padres.
        *   **Cromosomas homólogos**: Los cromosomas de los padres se alinean para el intercambio.
        """
        if random.random() < self.pc:
            # Punto de cruzamiento aleatorio (excluyendo los extremos 0 y chromosome_length).
            point = random.randint(1, self.chromosome_length - 1)
            # Se combinan los segmentos de los padres para formar los hijos.
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            # Si no hay cruzamiento, los hijos son copias idénticas (deep copy) de los padres.
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate_flip_bit(self, chromosome: list[int]) -> list[int]:
        """
        Realiza la mutación bit a bit en un cromosoma.
        Por cada gen en el cromosoma, con una probabilidad `pm`, su valor se invierte (0 a 1, o 1 a 0).

        Args:
            chromosome (list[int]): El genotipo a mutar.

        Returns:
            list[int]: El genotipo mutado.

        Principios Bioinspirados en el código:
        *   **Mutación**: Cambios aleatorios en el material genético (genotipo).
        *   **Alelo**: El valor de un gen cambia (0 a 1 o 1 a 0).
        """
        mutated_chromosome = copy.deepcopy(chromosome)
        for i in range(self.chromosome_length):
            if random.random() < self.pm:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]  # Invertir el bit (0->1, 1->0).
        return mutated_chromosome

    def _apply_elitism(self, new_population: list[list[int]]) -> list[list[int]]:
        """
        Aplica el mecanismo de elitismo, preservando al mejor individuo de la generación anterior
        (almacenado como `self.best_individual_genotype`) en la nueva población.
        El individuo con menor aptitud de la nueva población es reemplazado por el élite.

        Args:
            new_population (list[list[int]]): La población recién generada (después de crossover y mutación).

        Returns:
            list[list[int]]: La población con el individuo élite insertado.

        Principios Bioinspirados en el código:
        *   **Elitismo**: El individuo con mayor aptitud de la generación se garantiza que sobreviva a la siguiente.
        *   **Convergencia Monotónica**: El fitness máximo nunca decrece de una generación a otra.
        """
        if not self.best_individual_genotype: # Si aún no se ha encontrado un mejor individuo (primera generación).
            return new_population # No hay élite para aplicar.

        # Encontrar el peor individuo en la nueva población para ser reemplazado.
        new_fitness_values, _ = self._calculate_all_fitness(new_population)
        worst_individual_index = np.argmin(new_fitness_values)

        # Reemplazar al peor individuo de la nueva población con una copia del élite de la generación anterior.
        new_population[worst_individual_index] = copy.deepcopy(self.best_individual_genotype)
        return new_population

    def run(self, num_generations: int) -> tuple[list[int], float]:
        """
        Ejecuta el Algoritmo Genético para un número dado de generaciones.
        Este es el bucle principal del AG, que coordina la inicialización, evaluación, selección, cruzamiento y mutación
        a lo largo de múltiples generaciones.

        Args:
            num_generations (int): El número de generaciones a ejecutar.

        Returns:
            tuple[list[int], float]: El genotipo del mejor individuo encontrado en toda la ejecución y su aptitud.
        """
        self._initialize_population() # Inicializa la población con individuos aleatorios.

        for generation in range(num_generations):
            # 1. Evaluar la aptitud de la población actual.
            fitness_values, _ = self._calculate_all_fitness(self.population)

            # Actualizar el mejor individuo global (élite) si se encuentra uno mejor en la generación actual.
            current_best_index = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_index]
            current_best_genotype = copy.deepcopy(self.population[current_best_index])

            if current_best_fitness > self.best_individual_fitness:
                self.best_individual_fitness = current_best_fitness
                self.best_individual_genotype = current_best_genotype

            # Registrar el historial de fitness máximo y promedio para la visualización.
            self.max_fitness_history.append(self.best_individual_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))

            # 2. Selección: Crear el 'pool de apareamiento' (next_generation_candidates).
            if self.selection_method == 'roulette':
                mating_pool = self._select_proportional(self.population, fitness_values)
            else:  # tournament
                mating_pool = self._select_tournament(self.population, fitness_values)

            # 3. Cruzamiento y Mutación para crear la nueva generación.
            new_population = []
            # Asegurarse de que el 'mating_pool' tenga un número par de individuos para el cruzamiento por pares.
            if len(mating_pool) % 2 != 0:
                mating_pool.append(random.choice(mating_pool))

            random.shuffle(mating_pool) # Mezclar para formar pares aleatorios para el cruzamiento.

            # Realizar cruzamiento y mutación en pares de padres para generar hijos.
            for i in range(0, self.population_size, 2):
                parent1 = mating_pool[i]
                parent2 = mating_pool[i + 1]

                child1, child2 = self._crossover_one_point(parent1, parent2) # Aplicar cruzamiento.

                child1 = self._mutate_flip_bit(child1) # Aplicar mutación al primer hijo.
                child2 = self._mutate_flip_bit(child2) # Aplicar mutación al segundo hijo.

                new_population.extend([child1, child2]) # Añadir los hijos a la nueva población.

            # Asegurarse de que la nueva población tenga el tamaño correcto (puede ser ligeramente mayor por el ajuste de pares).
            self.population = new_population[:self.population_size]

            # 4. Elitismo (si está habilitado): Asegura que el mejor individuo no se pierda.
            if self.elitism:
                self.population = self._apply_elitism(self.population)

        # Retornar el mejor individuo encontrado y su aptitud final.
        return self.best_individual_genotype, self.best_individual_fitness

# --- Definición de funciones específicas para el Problema de la Mochila ---

# Definición de los ítems disponibles, cada uno con peso y valor.
ITEMS = [
    {'name': 'Libro', 'weight': 2, 'value': 10},
    {'name': 'Laptop', 'weight': 5, 'value': 100},
    {'name': 'Agua', 'weight': 1, 'value': 5},
    {'name': 'Snacks', 'weight': 3, 'value': 20},
    {'name': 'Linterna', 'weight': 1, 'value': 15},
    {'name': 'Cámara', 'weight': 4, 'value': 80},
    {'name': 'Tienda', 'weight': 10, 'value': 60},
    {'name': 'Botiquín', 'weight': 2, 'value': 25},
]
MAX_CAPACITY = 15 # Capacidad máxima de peso que la mochila puede soportar.
PENALTY_FACTOR = 5 # Factor para penalizar el fitness cuando se excede la capacidad.

def decode_knapsack(genotype: list[int]) -> dict:
    """
    Decodifica un genotipo (selección de ítems) en peso total y valor total de la mochila.
    Corresponde a la función e_tilde: S -> X.
    El genotipo es una cadena binaria donde cada bit indica si un ítem (según su índice) está incluido (1) o no (0).
    """
    total_weight = 0
    total_value = 0
    selected_items_indices = [] # Almacena los índices de los ítems seleccionados.
    for i, bit in enumerate(genotype):
        if bit == 1:
            # Si el bit es 1, el ítem se incluye en la mochila.
            total_weight += ITEMS[i]['weight']
            total_value += ITEMS[i]['value']
            selected_items_indices.append(i)
    return {
        'total_weight': total_weight,
        'total_value': total_value,
        'selected_items_indices': selected_items_indices
    }

def fitness_knapsack(phenotype: dict) -> float:
    """
    Calcula la aptitud para el problema de la mochila.
    El fitness es el valor total de los ítems, pero se aplica una penalización significativa
    si el peso total excede la capacidad máxima de la mochila.
    """
    total_weight = phenotype['total_weight']
    total_value = phenotype['total_value']

    if total_weight > MAX_CAPACITY:
        # Penalización por exceso de capacidad máxima.
        # La aptitud disminuye proporcionalmente al exceso de peso (exceso * PENALTY_FACTOR).
        return total_value - PENALTY_FACTOR * (total_weight - MAX_CAPACITY)
    else:
        # Si no se excede la capacidad, la aptitud es simplemente el valor total.
        return float(total_value)

# --- Parámetros del AG para el Problema de la Mochila ---
POP_SIZE_KNAPSACK = 50 # Tamaño de la población: 50 individuos.
CHROM_LEN_KNAPSACK = len(ITEMS) # Longitud del cromosoma: un bit por cada ítem disponible.
PC_KNAPSACK = 0.8 # Probabilidad de cruzamiento del 80%.
PM_KNAPSACK = 0.01 # Probabilidad de mutación del 1% por bit.
NUM_GENERATIONS_KNAPSACK = 100 # Número total de generaciones a ejecutar.

# Inicialización de la instancia del Algoritmo Genético con los parámetros definidos.
knapsack_ga = AlgoritmoGenetico(
    population_size=POP_SIZE_KNAPSACK,
    chromosome_length=CHROM_LEN_KNAPSACK,
    pc=PC_KNAPSACK,
    pm=PM_KNAPSACK,
    fitness_func=fitness_knapsack,
    decode_func=decode_knapsack,
    selection_method='tournament', # Usamos selección por torneo para este caso.
    tournament_size=5, # Tamaño del torneo para la selección.
    elitism=True, # Elitismo activado.
)

# Ejecutar el Algoritmo Genético para el número especificado de generaciones.
final_best_genotype_knapsack, final_best_fitness_knapsack = knapsack_ga.run(NUM_GENERATIONS_KNAPSACK)
# Decodificar el mejor genotipo encontrado para obtener el fenotipo (peso, valor e ítems seleccionados).
final_best_phenotype_knapsack = decode_knapsack(final_best_genotype_knapsack)

# Obtener los nombres de los ítems seleccionados para una salida más legible.
selected_items = [ITEMS[i]['name'] for i in final_best_phenotype_knapsack['selected_items_indices']]


# --- Ejercicio 2: Selección de Personal Estricta ---
# Se tienen 12 candidatos, cada uno con una puntuación de "Habilidad Técnica".
# Restricción: el equipo debe estar conformado por EXACTAMENTE 5 personas.
# La aptitud debe penalizar cualquier cromosoma que no sume exactamente 5 bits encendidos.

# Definición de los candidatos disponibles, cada uno con su puntuación de habilidad técnica.
CANDIDATOS = [
    {'name': 'Candidato_1', 'habilidad': 8},
    {'name': 'Candidato_2', 'habilidad': 5},
    {'name': 'Candidato_3', 'habilidad': 9},
    {'name': 'Candidato_4', 'habilidad': 3},
    {'name': 'Candidato_5', 'habilidad': 7},
    {'name': 'Candidato_6', 'habilidad': 6},
    {'name': 'Candidato_7', 'habilidad': 10},
    {'name': 'Candidato_8', 'habilidad': 4},
    {'name': 'Candidato_9', 'habilidad': 9},
    {'name': 'Candidato_10', 'habilidad': 2},
    {'name': 'Candidato_11', 'habilidad': 6},
    {'name': 'Candidato_12', 'habilidad': 8},
]
TEAM_SIZE = 5 # Restricción estricta: el equipo debe tener exactamente 5 integrantes.
PENALTY_FACTOR_PERSONNEL = 10 # Factor de penalización por cada persona de más o de menos respecto al TEAM_SIZE.

def decode_personnel(genotype: list[int]) -> dict:
    """
    Decodifica un genotipo (selección de candidatos) en la habilidad técnica total del equipo
    y la cantidad de candidatos seleccionados.
    Corresponde a la función e_tilde: S -> X.
    El genotipo es una cadena binaria donde cada bit indica si un candidato (según su índice)
    fue seleccionado (1) o no (0) para el equipo.
    """
    total_habilidad = 0
    num_seleccionados = 0
    selected_candidates_indices = [] # Almacena los índices de los candidatos seleccionados.
    for i, bit in enumerate(genotype):
        if bit == 1:
            # Si el bit es 1, el candidato se incluye en el equipo.
            total_habilidad += CANDIDATOS[i]['habilidad']
            num_seleccionados += 1
            selected_candidates_indices.append(i)
    return {
        'total_habilidad': total_habilidad,
        'num_seleccionados': num_seleccionados,
        'selected_candidates_indices': selected_candidates_indices
    }

def fitness_personnel(phenotype: dict) -> float:
    """
    Calcula la aptitud para el problema de Selección de Personal Estricta.
    El fitness es la suma de habilidad técnica del equipo, pero se aplica una penalización
    fuerte si el número de integrantes seleccionados no es EXACTAMENTE igual a TEAM_SIZE (5),
    ya sea por exceso o por defecto.
    """
    total_habilidad = phenotype['total_habilidad']
    num_seleccionados = phenotype['num_seleccionados']

    diferencia = abs(num_seleccionados - TEAM_SIZE)

    if diferencia > 0:
        # Penalización por no cumplir con exactamente TEAM_SIZE integrantes.
        # La aptitud disminuye proporcionalmente a la diferencia respecto al tamaño requerido.
        return total_habilidad - PENALTY_FACTOR_PERSONNEL * diferencia
    else:
        # Si el equipo tiene exactamente TEAM_SIZE integrantes, la aptitud es la habilidad total.
        return float(total_habilidad)

# --- Parámetros del AG para el Problema de Selección de Personal ---
POP_SIZE_PERSONNEL = 12 # Tamaño de la población: 50 individuos.
CHROM_LEN_PERSONNEL = len(CANDIDATOS) # Longitud del cromosoma: un bit por cada candidato disponible.
PC_PERSONNEL = 0.8 # Probabilidad de cruzamiento del 80%.
PM_PERSONNEL = 0.01 # Probabilidad de mutación del 1% por bit.
NUM_GENERATIONS_PERSONNEL = 100 # Número total de generaciones a ejecutar.

print("\n--- Ejecutando AG para el Problema de Selección de Personal Estricta ---")
# Inicialización de la instancia del Algoritmo Genético con los parámetros definidos.
personnel_ga = AlgoritmoGenetico(
    population_size=POP_SIZE_PERSONNEL,
    chromosome_length=CHROM_LEN_PERSONNEL,
    pc=PC_PERSONNEL,
    pm=PM_PERSONNEL,
    fitness_func=fitness_personnel,
    decode_func=decode_personnel,
    selection_method='tournament', # Usamos selección por torneo para este caso.
    tournament_size=5, # Tamaño del torneo para la selección.
    elitism=True, # Elitismo activado.
)

# Ejecutar el Algoritmo Genético para el número especificado de generaciones.
final_best_genotype_personnel, final_best_fitness_personnel = personnel_ga.run(NUM_GENERATIONS_PERSONNEL)
# Decodificar el mejor genotipo encontrado para obtener el fenotipo (habilidad total y candidatos seleccionados).
final_best_phenotype_personnel = decode_personnel(final_best_genotype_personnel)

print(f"\n--- Resultados Finales Selección de Personal Estricta ---")
print(f"Mejor Genotipo: {''.join(map(str, final_best_genotype_personnel))}")
print(f"Mejor Fitness (Habilidad Técnica Total): {final_best_fitness_personnel:.2f}")
print(f"Número de Integrantes Seleccionados: {final_best_phenotype_personnel['num_seleccionados']}")
print(f"Tamaño de Equipo Requerido: {TEAM_SIZE}")

# Obtener los nombres de los candidatos seleccionados para una salida más legible.
selected_candidates = [CANDIDATOS[i]['name'] for i in final_best_phenotype_personnel['selected_candidates_indices']]
print(f"Candidatos Seleccionados: {', '.join(selected_candidates)}")
