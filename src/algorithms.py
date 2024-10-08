import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
from scipy.optimize import fmin_cg
from pyswarm import pso
from deap import base, creator, tools, algorithms

# Definition of the Giunta Function
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

# Plotting the Giunta Function
def plot_giunta_function():
    """
    Plot the Giunta Function over a grid in the input space.
    """
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([giunta_function([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Giunta Function")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# Plot the Giunta Function
plot_giunta_function()

# Definition of bounds
bounds = [(-1, 1), (-1, 1)]
x_min_global = np.array([0.45834282, 0.45834282])
f_min_global = 0.060447

# Timer decorator
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper

# Optimization algorithm using minimize
@time_it
def optimize_minimize():
    """
    Optimize the Giunta function using the minimize method.

    Returns:
    result: Optimization result object.
    """
    result = minimize(giunta_function, x0=[0, 0], bounds=bounds)
    return result

# Optimization algorithm using differential evolution
@time_it
def optimize_differential_evolution():
    """
    Optimize the Giunta function using differential evolution.

    Returns:
    result: Optimization result object.
    """
    result = differential_evolution(giunta_function, bounds)
    return result

# Optimization algorithm using a genetic algorithm
@time_it
def optimize_genetic_algorithm():
    """
    Optimize the Giunta function using a genetic algorithm.

    Returns:
    best_individual (list): Best individual found.
    best_fitness (float): Fitness value of the best individual.
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

    # Adjustment: Return a value as a tuple.
    toolbox.register("evaluate", lambda ind: (giunta_function(ind),))

    population = toolbox.population(n=300)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, giunta_function(best_individual)

# Trust Region Method
@time_it
def optimize_trust_region():
    """
    Optimize the Giunta function using the trust region method.

    Returns:
    result: Optimization result object.
    """
    result = minimize(giunta_function, x0=[0, 0], method='trust-constr', bounds=bounds)
    return result

# Quasi-Newton Method (BFGS)
@time_it
def optimize_quasi_newton():
    """
    Optimize the Giunta function using the BFGS quasi-Newton method.

    Returns:
    result: Optimization result object.
    """
    result = minimize(giunta_function, x0=[0, 0], method='BFGS')
    return result

# Method of Steepest Descent
@time_it
def optimize_maximum_descent():
    """
    Optimize the Giunta function using the method of steepest descent.

    Returns:
    result (list): Optimal point found.
    optimal_value (float): Value of the Giunta function at the optimal point.
    """
    result = fmin_cg(giunta_function, x0=[0, 0])
    return result, giunta_function(result)

# Optimization algorithm using PSO
@time_it
def optimize_pso():
    """
    Optimize the Giunta function using Particle Swarm Optimization.

    Returns:
    best_position: The best position found.
    best_value: The value of the Giunta function at the best position.
    """
    best_position, best_value = pso(giunta_function, [b[0] for b in bounds], [b[1] for b in bounds])
    return best_position, best_value

# Optimization algorithm using Simulated Annealing
@time_it
def optimize_simulated_annealing():
    """
    Optimize the Giunta function using Simulated Annealing.

    Returns:
    result: Optimization result object.
    """
    result = dual_annealing(giunta_function, bounds=bounds)
    return result

# Basin Hopping Method
@time_it
def optimize_basin_hopping():
    """
    Optimize the Giunta function using the Basin Hopping method.

    Returns:
    result: Optimization result object.
    """
    minimizer_kwargs = {"method": "BFGS"}
    result = basinhopping(giunta_function, x0=[0, 0], minimizer_kwargs=minimizer_kwargs)
    return result

# Function to evaluate the results
def evaluate_result(x_result, f_result):
    """
    Evaluate the optimization result against the global minimum.

    Parameters:
    x_result (array): The result point to evaluate.
    f_result (float): The function value at the result point.

    Returns:
    error_x (float): The error in the position.
    error_f (float): The error in the function value.
    """
    error_x = np.linalg.norm(x_result - x_min_global)
    error_f = abs(f_result - f_min_global)
    return error_x, error_f

# Results from different methods
minimize_result, minimize_time = optimize_minimize()
de_result, de_time = optimize_differential_evolution()
ga_result, ga_time = optimize_genetic_algorithm()
trust_region_result, trust_region_time = optimize_trust_region()
quasi_newton_result, quasi_newton_time = optimize_quasi_newton()
max_descent_result, max_descent_time = optimize_maximum_descent()
pso_result, pso_time = optimize_pso()
sa_result, sa_time = optimize_simulated_annealing()
bh_result, bh_time = optimize_basin_hopping()

# Evaluation of the results
errors = {
    "Minimize": evaluate_result(minimize_result.x, minimize_result.fun),
    "Differential Evolution": evaluate_result(de_result.x, de_result.fun),
    "Genetic Algorithm": evaluate_result(ga_result[0], ga_result[1]),
    "Trust Region": evaluate_result(trust_region_result.x, trust_region_result.fun),
    "Quasi-Newton": evaluate_result(quasi_newton_result.x, quasi_newton_result.fun),
    "Max Descent": evaluate_result(max_descent_result[0], max_descent_result[1]),
    "PSO": evaluate_result(np.array(pso_result[0]), pso_result[1]),
    "Simulated Annealing": evaluate_result(sa_result.x, sa_result.fun),
    "Basin Hopping": evaluate_result(bh_result.x, bh_result.fun),
}

# Printing results
print("Minimize Result:", minimize_result.x, minimize_result.fun, "Time:", minimize_time)
print("Differential Evolution Result:", de_result.x, de_result.fun, "Time:", de_time)
print("Genetic Algorithm Result:", ga_result[0], ga_result[1], "Time:", ga_time)
print("Trust Region Result:", trust_region_result.x, trust_region_result.fun, "Time:", trust_region_time)
print("Quasi-Newton Result:", quasi_newton_result.x, quasi_newton_result.fun, "Time:", quasi_newton_time)
print("Maximum Descent Result:", max_descent_result[0], max_descent_result[1], "Time:", max_descent_time)
print("PSO Result:", pso_result[0], pso_result[1], "Time:", pso_time)
print("Simulated Annealing Result:", sa_result.x, sa_result.fun, "Time:", sa_time)
print("Basin Hopping Result:", bh_result.x, bh_result.fun, "Time:", bh_time)

# Creating a DataFrame for the results
results_data = {
    "Method": ["Minimize", "Differential Evolution", "Genetic Algorithm", "Trust Region",
               "Quasi-Newton", "Max Descent", "PSO", "Simulated Annealing", "Basin Hopping"],
    "x": [minimize_result.x, de_result.x, ga_result[0], trust_region_result.x, quasi_newton_result.x, 
          max_descent_result[0], pso_result[0], sa_result.x, bh_result.x],
    "f(x)": [minimize_result.fun, de_result.fun, ga_result[1], trust_region_result.fun, 
             quasi_newton_result.fun, max_descent_result[1], pso_result[1], sa_result.fun, bh_result.fun],
    "Time (s)": [minimize_time, de_time, ga_time, trust_region_time, quasi_newton_time, 
                 max_descent_time, pso_time, sa_time, bh_time]
}

results_df = pd.DataFrame(results_data)

# Printing errors
print("\nErrors relative to the global minimum:")
for method, (error_x, error_f) in errors.items():
    print(f"{method} - Error in x: {error_x}, Error in f: {error_f}")

# Plotting results
methods = ["Minimize", "Differential Evolution", "Genetic Algorithm", "Trust Region",
           "Quasi-Newton", "Max Descent", "PSO", "Simulated Annealing", "Basin Hopping"]

# Creating a DataFrame for the errors
errors_data = {
    "Method": ["Minimize", "Differential Evolution", "Genetic Algorithm", "Trust Region",
               "Quasi-Newton", "Max Descent", "PSO", "Simulated Annealing", "Basin Hopping"],
    "Error in x": [errors[m][0] for m in methods],
    "Error in f": [errors[m][1] for m in methods]
}

errors_df = pd.DataFrame(errors_data)

errors_x = [errors[m][0] for m in methods]
errors_f = [errors[m][1] for m in methods]
times = [minimize_time, de_time, ga_time, trust_region_time, quasi_newton_time,
         max_descent_time, pso_time, sa_time, bh_time]

results_df = results_df.round(8)
errors_df = errors_df.round(8)

# Plotting the error in position
plt.figure(figsize=(8, 4))  
plt.bar(methods, errors_x)
plt.ylabel('Error in x', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.title('Error in Position', fontsize=16)
plt.tight_layout()  
plt.show()

# Plotting the time taken
plt.figure(figsize=(8, 4))  
plt.bar(methods, times)
plt.ylabel('Time (s)', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.title('Time Taken', fontsize=16)
plt.tight_layout()  
plt.show()

results_df['x'] = results_df['x'].apply(lambda x: np.round(x, 8))

# Plotting the results as a table
fig, ax = plt.subplots(figsize=(10, 4))  
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)  
table.scale(1.2, 1.2)  

plt.title('Optimization Results', fontsize=16)  
plt.tight_layout()  
plt.show()

# Plotting the errors as a table
fig, ax = plt.subplots(figsize=(10, 4))  
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=errors_df.values, colLabels=errors_df.columns, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)  
table.scale(1.2, 1.2)  

plt.title('Errors Relative to Global Minimum', fontsize=16)  
plt.tight_layout()  
plt.show()
