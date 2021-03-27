import math
import numpy as np
from numpy import random as rd
from random import randrange
import ga

# Population size.
num_individuals = 100

# Generations count.
num_generations = 10000

# Number of the weights to optimize. (x, y)
num_weights = 2

# Number of reproducing pairs.
num_parents_mating = int(num_individuals / 2)

# Percentage of mutation odds.
mutation_chance = 0

# Generates first population
pop_size = (num_individuals, num_weights)
start_population = np.random.uniform(low=-10.0, high=10.0, size=pop_size)

# Get fitness_score property from the chromossome.
def get_fitness_score(elem):
    return elem[1]

# Sorts a generation, ordering by fitness value (ascending).
def sort_generation(population):
    population.sort(key=get_fitness_score)

# Function that remarks the fitness of the subject.
def bird_function(x,y):
    return math.sin(x) * math.exp((1 - math.cos(y))**2) + math.cos(y) * math.exp((1 - math.sin(x))**2) + (x - y)**2

# Grants a fitness_score for each of the population individuals.
def fitness_function(population):
    evaluated_pop = []

    for individual in start_population:
        x = individual[0]
        y = individual[1]
        fitness_score = bird_function(x, y)
        evaluated_pop.append(([x,y], fitness_score))

    sort_generation(evaluated_pop)
    return evaluated_pop

# Gets a list of the fitest individuals of a population.
def define_parents(population):
    mating_parents = []
    for parent_index in range(num_parents_mating):
        mating_parents.append(population[parent_index])
    return mating_parents

# Makes the gene crossover to generate the new offspring.
def population_mating(population):
    offspring = []
    for index, chromossome in enumerate(population):
        chromossome_index = 0
        while True:
            chromossome_index = rd.randint(len(population))
            if (chromossome_index != index):
                break

        first_spawn = ([chromossome[0][0],population[chromossome_index][0][1]])
        offspring.append(first_spawn)
        second_spawn = ([population[chromossome_index][0][0],chromossome[0][1]])
        offspring.append(second_spawn)

    return offspring

# Determines a random gene to be randonly mutated.
def mutate(chromossome):
    target_chromossome  = rd.randint(1)
    chromossome[target_chromossome] = np.random.uniform(low=-10.0, high=10.0)
    return chromossome

# Determines when a mutation should occur.
def mutation_odds(chance):
    odd = randrange(1,100)
    return chance >= odd

# Controls the mutation proccess.
def define_offspring_mutation(offspring):
    if(mutation_odds(mutation_chance)):
        mutation_target = rd.randint(len(offspring))
        mutate(offspring[mutation_target])

# Prints each chromossome of the generation.
def print_generation(gen):
    print("[{0}]".format('\n'.join(map(str, gen))))

# Iterate over generations
actual_generation = start_population
evaluated_population = fitness_function(actual_generation)
for generation in range(num_generations):
    parents = define_parents(evaluated_population)
    offspring = population_mating(parents)
    define_offspring_mutation(offspring)
    actual_generation = fitness_function(offspring.copy())
    actual_generation.extend(parents)
    evaluated_population = actual_generation.copy()

best_match = min(evaluated_population, key=lambda chromossome: chromossome[1])
print_generation(evaluated_population)
print("\n\nBest solution fitness : ", best_match)
print("\n\nReference GLOBAL MINIMUM : ", bird_function(4.70104,3.15294))



# GLOBAL MINS
# (4.70104, 3.15294)
# (−1.58214, −3.13024)




