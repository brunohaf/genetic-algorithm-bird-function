import math
import numpy as np
from numpy import random as rd
from random import randrange
import ga

# Population size.
num_individuals = 8

# Generations count.
num_generations = 5

# Number of the weights to optimize.
num_weights = 2

# Number of reproducing pairs.
num_parents_mating = int(num_individuals / 2)

# Mutation odds
mutation_chance = 100

# Generate first population
pop_size = (num_individuals, num_weights)
new_population = np.random.uniform(low=-10.0, high=10.0, size=pop_size)

# Sort self.indiv_fits in descending order by self.fitness value
def get_fitness_score(elem):
    return elem[1]

def sort_generation(population):
    population.sort(key=get_fitness_score)

def bird_function(x,y):
    return math.sin(x) * math.exp((1 - math.cos(y))**2) + math.cos(y) * math.exp((1 - math.sin(x))**2) + (x - y)**2

def fitness_function(population):
    evaluated_pop = []

    for individual in new_population:
        x = individual[0]
        y = individual[1]
        fitness_score = bird_function(x, y)
        evaluated_pop.append(([x,y], fitness_score))

    sort_generation(evaluated_pop)
    return evaluated_pop

def define_parents(population):
    mating_parents = []
    for parent_index in range(num_parents_mating):
        mating_parents.append(population[parent_index])
    return mating_parents


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

def mutate(chromossome):
    target_chromossome  = rd.randint(1)
    chromossome[target_chromossome] = np.random.uniform(low=-10.0, high=10.0)
    return chromossome

def mutation_odds(chance):
    odd = randrange(1,100)
    return chance >= odd

def offspring_mutation(offspring):
    if(mutation_odds(mutation_chance)):
        mutation_target = rd.randint(len(offspring))
        mutate(offspring[mutation_target])

# WIP
pop = fitness_function(new_population)
parents = define_parents(pop)
new_generation = population_mating(parents)
offspring_mutation(new_generation)

# GLOBAL MINS
# (4.70104, 3.15294)
# (−1.58214, −3.13024)