import numpy as np
from scipy.optimize import minimize, differential_evolution
from deap import base, creator, tools, algorithms

# Definición de la Giunta Function
def giunta_function(x):
    """
    Compute the value of the Giunta function at point x.
    
    Parameters:
    x (array): Input array where len(x) = 2.
    
    Returns:
    float: Value of the Giunta function.
    """
    return 0.6 + sum(
        np.sin(16/15 * xi - 1) + np.sin(16/15 * xi - 1)**2 + 1/50 * np.sin(4 * (16/15 * xi - 1))
        for xi in x
    )

# Definición de límites
bounds = [(-1, 1), (-1, 1)]

# Algoritmo de optimización por minimización
def optimize_minimize():
    """
    Perform optimization using the 'minimize' function from scipy.optimize.
    
    Returns:
    dict: Optimization result.
    """
    result = minimize(giunta_function, x0=[0, 0], bounds=bounds)
    return result

# Algoritmo de optimización por evolución diferencial
def optimize_differential_evolution():
    """
    Perform optimization using the 'differential_evolution' function from scipy.optimize.
    
    Returns:
    dict: Optimization result.
    """
    result = differential_evolution(giunta_function, bounds)
    return result

# Algoritmo de optimización mediante un algoritmo genético
def optimize_genetic_algorithm():
    """
    Perform optimization using a genetic algorithm implemented with DEAP.
    
    Returns:
    dict: Best individual and its fitness.
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Ajuste: Devolver un valor como una tupla.
    toolbox.register("evaluate", lambda ind: (giunta_function(ind),))

    population = toolbox.population(n=300)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, giunta_function(best_individual)

# Resultados de los diferentes métodos
minimize_result = optimize_minimize()
de_result = optimize_differential_evolution()
ga_result = optimize_genetic_algorithm()

print("Minimize Result:", minimize_result.x, minimize_result.fun)
print("Differential Evolution Result:", de_result.x, de_result.fun)
print("Genetic Algorithm Result:", ga_result[0], ga_result[1])
