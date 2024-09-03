import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import newton, fmin_cg, fmin_bfgs, fmin_ncg
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
x_min_global = np.array([0.45834282, 0.45834282])
f_min_global = 0.060447

# Algoritmo de optimización por minimización
def optimize_minimize():
    result = minimize(giunta_function, x0=[0, 0], bounds=bounds)
    return result

# Algoritmo de optimización por evolución diferencial
def optimize_differential_evolution():
    result = differential_evolution(giunta_function, bounds)
    return result

# Algoritmo de optimización mediante un algoritmo genético
def optimize_genetic_algorithm():
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

# Método de Región de Confianza
def optimize_trust_region():
    result = minimize(giunta_function, x0=[0, 0], method='trust-constr', bounds=bounds)
    return result

# Método de Quasi-Newton (BFGS)
def optimize_quasi_newton():
    result = minimize(giunta_function, x0=[0, 0], method='BFGS')
    return result

# Método de Máximo Descenso
def optimize_maximum_descent():
    result = fmin_cg(giunta_function, x0=[0, 0])
    return result, giunta_function(result)

# Función de evaluación de los resultados
def evaluate_result(x_result, f_result):
    error_x = np.linalg.norm(x_result - x_min_global)
    error_f = abs(f_result - f_min_global)
    return error_x, error_f

# Resultados de los diferentes métodos
minimize_result = optimize_minimize()
de_result = optimize_differential_evolution()
ga_result = optimize_genetic_algorithm()
trust_region_result = optimize_trust_region()
quasi_newton_result = optimize_quasi_newton()
max_descent_result = optimize_maximum_descent()

# Evaluación de los resultados
errors = {
    "Minimize": evaluate_result(minimize_result.x, minimize_result.fun),
    "Differential Evolution": evaluate_result(de_result.x, de_result.fun),
    "Genetic Algorithm": evaluate_result(ga_result[0], ga_result[1]),
    "Trust Region": evaluate_result(trust_region_result.x, trust_region_result.fun),
    "Quasi-Newton": evaluate_result(quasi_newton_result.x, quasi_newton_result.fun),
    "Max Descent": evaluate_result(max_descent_result[0], max_descent_result[1]),
}

# Impresión de resultados
print("Minimize Result:", minimize_result.x, minimize_result.fun)
print("Differential Evolution Result:", de_result.x, de_result.fun)
print("Genetic Algorithm Result:", ga_result[0], ga_result[1])
print("Trust Region Result:", trust_region_result.x, trust_region_result.fun)
print("Quasi-Newton Result:", quasi_newton_result.x, quasi_newton_result.fun)
print("Maximum Descent Result:", max_descent_result[0], max_descent_result[1])

# Impresión de errores
print("\nErrores respecto al mínimo global:")
for method, (error_x, error_f) in errors.items():
    print(f"{method} - Error en x: {error_x}, Error en f: {error_f}")
