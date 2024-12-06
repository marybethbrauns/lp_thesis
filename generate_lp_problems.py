import numpy as np
import pandas as pd
import pulp
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os
from openpyxl.styles import Font, PatternFill

# Utility Functions
def initialize_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def generate_problem(num_vars, num_constraints):
    obj = np.random.randint(-10, 11, num_vars)
    constraints = np.hstack([
        np.random.randint(-5, 6, (num_constraints, num_vars)),
        np.random.randint(1, 51, (num_constraints, 1))
    ])
    return obj, constraints

def solve_pulp(obj, constraints):
    prob = pulp.LpProblem("ILP", pulp.LpMinimize)
    vars = [pulp.LpVariable(f'x{i}', cat='Integer') for i in range(len(obj))]
    prob += pulp.lpSum(obj[i] * vars[i] for i in range(len(obj)))
    for constraint in constraints:
        prob += pulp.lpSum(constraint[j] * vars[j] for j in range(len(obj))) <= constraint[-1]
    prob.solve()
    return {
        "status": pulp.LpStatus[prob.status],
        "objective_value": pulp.value(prob.objective),
        "variables": [v.varValue for v in vars],
        "solve_time": prob.solutionTime,
    }

def solve_genetic(obj, constraints, bounds=(0, 10), ngen=50, pop_size=50, cxpb=0.7, mutpb=0.2):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, bounds[0], bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(obj))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (sum(x * c for x, c in zip(ind, obj)) +
                                              sum(max(0, sum(x * c for x, c in zip(ind, con[:-1])) - con[-1]) * 100
                                                  for con in constraints),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=bounds[0], up=bounds[1], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
    best = list(hof[0])
    return {"objective_value": sum(x * c for x, c in zip(best, obj)), "variables": best}

def solve_problems(num_problems, num_vars, num_constraints):
    results = []
    for _ in tqdm(range(num_problems), desc="Solving problems"):
        obj, constraints = generate_problem(num_vars, num_constraints)
        math_solution = solve_pulp(obj, constraints)
        ga_solution = solve_genetic(obj, constraints)
        results.append({
            "objective": obj.tolist(),
            "constraints": constraints.tolist(),
            "math_solution": math_solution,
            "ga_solution": ga_solution,
            "objective_diff": abs(math_solution["objective_value"] - ga_solution["objective_value"]),
        })
    return results

def calculate_metrics(results):
    diffs = [r["objective_diff"] for r in results]
    return {"mean_diff": np.mean(diffs), "median_diff": np.median(diffs), "max_diff": np.max(diffs)}

def save_to_excel(results, metrics, file_path="results.xlsx"):
    """Save detailed results and metrics to an Excel file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    wb = Workbook()

    # Sheet 1: Problem Details
    ws_problems = wb.active
    ws_problems.title = "Problem Details"
    ws_problems.append([
        "Problem ID", "Objective", "Constraints", "Math Objective Value",
        "Math Variables", "GA Objective Value", "GA Variables", "Objective Difference"
    ])
    for idx, r in enumerate(results):
        ws_problems.append([
            idx, str(r["objective"]), str(r["constraints"]),
            r["math_solution"]["objective_value"], str(r["math_solution"]["variables"]),
            r["ga_solution"]["objective_value"], str(r["ga_solution"]["variables"]),
            r["objective_diff"]
        ])

    # Sheet 2: Metrics Overview
    ws_metrics = wb.create_sheet("Metrics Overview")
    ws_metrics.append(["Metric", "Value"])
    for k, v in metrics.items():
        ws_metrics.append([k, v])

    # Sheet 3: Detailed Solver Comparison
    ws_comparison = wb.create_sheet("Solver Comparison")
    ws_comparison.append([
        "Problem ID", "Math Objective", "GA Objective",
        "Math Solve Time", "Objective Difference"
    ])
    for idx, r in enumerate(results):
        ws_comparison.append([
            idx,
            r["math_solution"]["objective_value"],
            r["ga_solution"]["objective_value"],
            r["math_solution"]["solve_time"],
            r["objective_diff"]
        ])

    # Sheet 4: Constraints Breakdown
    ws_constraints = wb.create_sheet("Constraints Breakdown")
    ws_constraints.append(["Problem ID", "Constraint ID", "Constraint Details"])
    for idx, r in enumerate(results):
        for cid, constraint in enumerate(r["constraints"]):
            ws_constraints.append([idx, cid, str(constraint)])

    # Sheet 5: Variable Analysis
    ws_variables = wb.create_sheet("Variable Analysis")
    ws_variables.append([
        "Problem ID", "Variable Index", "Math Variable Value", "GA Variable Value"
    ])
    for idx, r in enumerate(results):
        for vid, (math_var, ga_var) in enumerate(zip(r["math_solution"]["variables"], r["ga_solution"]["variables"])):
            ws_variables.append([idx, vid, math_var, ga_var])

    # Style the Excel sheets for readability
    for ws in wb.worksheets:
        style_excel_sheet(ws)

    # Save workbook
    wb.save(file_path)
    print(f"Detailed results saved to {file_path}")


def style_excel_sheet(ws):
    """Apply consistent styling to an Excel sheet."""
    for cell in ws[1]:  # Style header row
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
    for column_cells in ws.columns:  # Auto-adjust column widths
        max_length = max(len(str(cell.value)) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = max_length



if __name__ == "__main__":
    initialize_random_seeds(seed=42)

    num_problems = 100  # Number of problems to solve
    num_vars = 4        # Number of variables per problem
    num_constraints = 5 # Number of constraints per problem

    print("Generating and solving problems...")
    results = solve_problems(num_problems, num_vars, num_constraints)

    #Calculate metrics to analyze solver performance
    metrics = calculate_metrics(results)
    print("\nMetrics Summary:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    #Save detailed results and metrics to an Excel file
    save_to_excel(results, metrics, "results.xlsx")
    print("\nExperiment completed. Results saved to 'results.xlsx'.")

