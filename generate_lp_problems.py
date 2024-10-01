import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pulp
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import train_test_split

class IntegerLPProblem:
    def __init__(self, num_variables, num_constraints):
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.objective = None
        self.constraints = None
        self.bounds = None
        self.solution = None

    def generate_problem(self):
        self.objective = np.random.randint(-10, 11, size=self.num_variables)
        self.constraints = np.random.randint(-5, 6, size=(self.num_constraints, self.num_variables))
        rhs = np.random.randint(1, 51, size=self.num_constraints)
        self.constraints = np.column_stack((self.constraints, rhs))
        self.bounds = [(0, 10) for _ in range(self.num_variables)]

    def solve(self):
        prob = pulp.LpProblem("Integer LP Problem", pulp.LpMinimize)
        vars = [pulp.LpVariable(f'x{i}', lowBound=self.bounds[i][0], upBound=self.bounds[i][1], cat='Integer')
                for i in range(self.num_variables)]
        prob += pulp.lpSum(self.objective[i] * vars[i] for i in range(self.num_variables))
        for i in range(self.num_constraints):
            prob += pulp.lpSum(self.constraints[i, j] * vars[j] for j in range(self.num_variables)) <= self.constraints[i, -1]
        prob.solve()
        self.solution = {
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'variables': [var.varValue for var in prob.variables()]
        }

    def to_dict(self):
        return {
            'num_variables': self.num_variables,
            'num_constraints': self.num_constraints,
            'objective': self.objective.tolist(),
            'constraints': self.constraints.tolist(),
            'bounds': self.bounds,
            'solution': self.solution
        }

class LPDatasetGenerator:
    def __init__(self, num_problems, min_variables, max_variables, min_constraints, max_constraints):
        self.num_problems = num_problems
        self.min_variables = min_variables
        self.max_variables = max_variables
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.problems = []

    def generate_dataset(self):
        for _ in tqdm(range(self.num_problems), desc="Generating problems"):
            num_vars = np.random.randint(self.min_variables, self.max_variables + 1)
            num_cons = np.random.randint(self.min_constraints, self.max_constraints + 1)
            problem = IntegerLPProblem(num_vars, num_cons)
            problem.generate_problem()
            problem.solve()
            self.problems.append(problem.to_dict())

    def to_dataframe(self):
        return pd.DataFrame(self.problems)

def evaluate(individual, problem):
    objective = problem['objective']
    constraints = problem['constraints']

    obj_value = sum(x * c for x, c in zip(individual, objective))

    penalty = 0
    for constraint in constraints:
        left_side = sum(x * c for x, c in zip(individual, constraint[:-1]))
        if left_side > constraint[-1]:
            penalty += (left_side - constraint[-1]) * 100

    return obj_value + penalty,

def calculate_objective_value(solution, problem):
    return sum(x * c for x, c in zip(solution, problem['objective']))

def create_toolbox(num_variables, bounds):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, bounds[0], bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_variables)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=bounds[0], up=bounds[1], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def solve_with_ga(problem, pop_size=50, ngen=50, cxpb=0.7, mutpb=0.2):
    num_variables = problem['num_variables']
    bounds = problem['bounds'][0]  # Assuming all variables have the same bounds

    toolbox = create_toolbox(num_variables, bounds)
    toolbox.register("evaluate", evaluate, problem=problem)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                   stats=stats, halloffame=hof, verbose=False)

    return {
        'best_solution': hof[0],
        'best_fitness': hof[0].fitness.values[0],
        'predicted_objective': calculate_objective_value(hof[0], problem)
    }

def process_dataset(df, is_training=True):
    results = []
    for _, problem in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        ga_result = solve_with_ga(problem)
        results.append({
            'problem_id': _,
            'best_solution': ga_result['best_solution'],
            'best_fitness': ga_result['best_fitness'],
            'predicted_objective': ga_result['predicted_objective'],
            'optimal_solution': problem['solution']['variables'],
            'optimal_objective': problem['solution']['objective_value']
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # Generate a dataset of problems
    generator = LPDatasetGenerator(num_problems=1000, min_variables=2, max_variables=5,
                                   min_constraints=2, max_constraints=5)
    generator.generate_dataset()

    # Create a pandas DataFrame from the generated problems
    problems_df = generator.to_dataframe()

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(problems_df, test_size=0.2, random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")

    # Process training and testing datasets
    train_results = process_dataset(train_df, is_training=True)
    test_results = process_dataset(test_df, is_training=False)

    # Calculate and print the mean absolute error for training and testing
    train_mae = np.mean(np.abs(train_results['predicted_objective'] - train_results['optimal_objective']))
    test_mae = np.mean(np.abs(test_results['predicted_objective'] - test_results['optimal_objective']))

    print(f"\nTraining Mean Absolute Error: {train_mae}")
    print(f"Testing Mean Absolute Error: {test_mae}")

    # Combine results
    results_combined = pd.DataFrame({
        'problem_id': test_results['problem_id'],
        'predicted_objective_value': test_results['predicted_objective'],
        'actual_objective_value': test_results['optimal_objective'],
        'solution_fitness': test_results['best_fitness']
    })

    # Print combined results
    print("\nCombined results of predicted and actual optimal solutions:")
    print(results_combined)


    # Optionally, save the results to a CSV file
    results_combined.to_csv('/Users/marybeth/Desktop/results.combined.csv', index=False)

    # Print rows where predicted != actual
    mismatches = results_combined[
        results_combined['predicted_objective_value'] != results_combined['actual_objective_value']]

    print("\nRows where predicted objective value doesn't match actual objective value:")
    if mismatches.empty:
        print("No mismatches found. The genetic algorithm found the optimal solution for all test problems.")
    else:
        print(mismatches)
        print(f"\nNumber of mismatches: {len(mismatches)} out of {len(results_combined)} test problems")

        # Calculate and print the average difference for mismatches
        avg_difference = np.mean(np.abs(mismatches['predicted_objective_value'] - mismatches['actual_objective_value']))
        print(f"Average absolute difference in mismatches: {avg_difference:.4f}")

    # Calculate and print overall statistics
    total_problems = len(results_combined)
    mismatch_percentage = (len(mismatches) / total_problems) * 100
    print(f"\nPercentage of problems where GA didn't find the optimal solution: {mismatch_percentage:.2f}%")

    overall_avg_difference = np.mean(
        np.abs(results_combined['predicted_objective_value'] - results_combined['actual_objective_value']))
    print(f"Overall average absolute difference: {overall_avg_difference:.4f}")