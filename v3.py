import numpy as np
import pandas as pd
import time
import random
from scipy.optimize import linprog


# ---------------------------------------------------
# Helper: Generate a random LP problem that is feasible
# ---------------------------------------------------
def generate_random_lp_problem(n_vars=3, n_constraints=4, lower_bound=0, upper_bound=10):
    """
    Generates a random LP of the form:
         minimize     c^T x
         subject to   A_ub x <= b_ub
                      x in [lower_bound, upper_bound]^n_vars

    We first generate a random feasible point and then choose b_ub
    so that each constraint is satisfied (with some added slack).
    """
    x_feasible = np.random.uniform(lower_bound, upper_bound, size=n_vars)
    A_ub = np.random.uniform(0, 10, size=(n_constraints, n_vars))
    slack = np.random.uniform(1, 10, size=n_constraints)  # extra room for feasibility
    b_ub = A_ub.dot(x_feasible) + slack
    c = np.random.uniform(-10, 10, size=n_vars)
    bounds = [(lower_bound, upper_bound) for _ in range(n_vars)]
    return {"c": c, "A_ub": A_ub, "b_ub": b_ub, "bounds": bounds, "x_feasible": x_feasible}


# ---------------------------------------------------
# Method 1: Solve using the classical simplex (HiGHS) method
# ---------------------------------------------------
def solve_lp_simplex(lp):
    c, A_ub, b_ub, bounds = lp["c"], lp["A_ub"], lp["b_ub"], lp["bounds"]
    start_time = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    solve_time = time.time() - start_time
    if res.success:
        solution = res.x
        objective = res.fun
    else:
        solution = None
        objective = None
    return solution, objective, solve_time


# ---------------------------------------------------
# Helper: Fitness function with penalty for constraint violation.
# ---------------------------------------------------
def fitness_function(x, c, A_ub, b_ub, penalty_factor=1000):
    obj = np.dot(c, x)
    penalty = 0.0
    for i in range(len(b_ub)):
        violation = np.dot(A_ub[i], x) - b_ub[i]
        if violation > 0:
            penalty += penalty_factor * violation
    return obj + penalty


def solve_lp_ga(lp, population_size=100, generations=200, mutation_rate=0.1,
                penalty_factor=1000, initial_population=None, early_stop_generations=20, improvement_threshold=1e-4):
    c, A_ub, b_ub, bounds = lp["c"], lp["A_ub"], lp["b_ub"], lp["bounds"]
    n_vars = len(c)

    # Initialize population (using warm start if provided)
    population = []
    if initial_population is not None:
        for individual in initial_population:
            clipped = [max(bounds[i][0], min(individual[i], bounds[i][1])) for i in range(n_vars)]
            population.append(clipped)
        while len(population) < population_size:
            individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_vars)]
            population.append(individual)
        population = population[:population_size]
    else:
        for _ in range(population_size):
            individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_vars)]
            population.append(individual)

    def mutate(individual):
        for i in range(n_vars):
            if random.random() < mutation_rate:
                individual[i] += random.gauss(0, 1)
                individual[i] = max(bounds[i][0], min(individual[i], bounds[i][1]))
        return individual

    def crossover(parent1, parent2):
        point = random.randint(1, n_vars - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def tournament_selection(pop):
        tournament_size = 3
        candidates = random.sample(pop, tournament_size)
        candidates.sort(key=lambda ind: fitness_function(ind, c, A_ub, b_ub, penalty_factor))
        return candidates[0]

    best_individual = None
    best_fitness = float('inf')
    start_time = time.time()

    # Variables for early stopping
    generations_without_improvement = 0

    for gen in range(generations):
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = sorted(new_population, key=lambda ind: fitness_function(ind, c, A_ub, b_ub, penalty_factor))[
                     :population_size]
        current_best = population[0]
        current_fitness = fitness_function(current_best, c, A_ub, b_ub, penalty_factor)

        # Check for improvement
        if best_fitness - current_fitness < improvement_threshold:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = current_best

        # Early stopping condition
        if generations_without_improvement >= early_stop_generations:
            print(f"Early stopping at generation {gen}")
            break

    solve_time = time.time() - start_time
    is_feasible = all(np.dot(A_ub[i], best_individual) <= b_ub[i] + 1e-6 for i in range(len(b_ub)))
    if not is_feasible:
        objective = None
    else:
        objective = np.dot(c, best_individual)

    return best_individual, objective, solve_time, population


# ---------------------------------------------------
# Main function: Run multiple LP problems using warm-started GA
# ---------------------------------------------------
def main():
    num_problems = 10  # number of LP problems to test
    results = []  # to store results from each method for each problem
    problems = []  # to store the problem definitions

    warm_start_population = None  # Will hold the final GA population from the previous problem

    for i in range(num_problems):
        # Generate an LP problem.
        lp = generate_random_lp_problem(n_vars=3, n_constraints=4, lower_bound=0, upper_bound=10)
        problems.append(lp)

        # Solve using the classical simplex method.
        sol_simplex, obj_simplex, time_simplex = solve_lp_simplex(lp)
        results.append({
            "Problem": i,
            "Method": "Simplex (HiGHS)",
            "Solution": str(sol_simplex),
            "Objective": obj_simplex,
            "SolveTime_sec": time_simplex,
            "Feasible": sol_simplex is not None
        })

        # Solve using the GA, using the previous population for warm starting.
        sol_ga, obj_ga, time_ga, final_population = solve_lp_ga(lp, initial_population=warm_start_population)
        results.append({
            "Problem": i,
            "Method": "Genetic Algorithm (Warm Start)",
            "Solution": str(sol_ga),
            "Objective": obj_ga,
            "SolveTime_sec": time_ga,
            "Feasible": obj_ga is not None
        })

        # Set the final GA population as the warm start for the next LP problem.
        warm_start_population = final_population

    # Create DataFrames for the results and problem definitions.
    df_results = pd.DataFrame(results)
    problem_details = []
    for idx, lp in enumerate(problems):
        problem_details.append({
            "Problem": idx,
            "c": str(lp["c"]),
            "A_ub": str(lp["A_ub"].tolist()),
            "b_ub": str(lp["b_ub"].tolist()),
            "Bounds": str(lp["bounds"])
        })
    df_problems = pd.DataFrame(problem_details)

    # Write the results to an Excel file.
    report_filename = "lp_warm_start_ga_report.xlsx"
    with pd.ExcelWriter(report_filename) as writer:
        df_results.to_excel(writer, sheet_name="Results", index=False)
        df_problems.to_excel(writer, sheet_name="Problems", index=False)

    print(f"Excel report generated: {report_filename}")


if __name__ == "__main__":
    main()
