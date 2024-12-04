import numpy as np
import pandas as pd
import pulp
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
import time
from typing import List, Dict, Tuple
import os

# Utility Functions
def initialize_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def calculate_objective_value(variables, coefficients):
    """Calculate the objective value as the dot product of variables and coefficients."""
    return sum(x * c for x, c in zip(variables, coefficients))

def calculate_constraint_penalty(variables, constraints):
    """Calculate penalty for constraint violations."""
    penalty = 0
    for constraint in constraints:
        violation = max(0, sum(x * c for x, c in zip(variables, constraint[:-1])) - constraint[-1])
        penalty += violation * 100
    return penalty

def calculate_objective_and_speedup(math_obj, gen_obj, math_time, gen_time):
    """Calculate objective differences and speedup factors."""
    if math_obj is None:
        math_obj = 0
    gen_obj = int(gen_obj)
    return abs(math_obj - gen_obj), math_time / gen_time

def aggregate_metrics(series):
    """Calculate mean, median, min, and max for a pandas series."""
    return {
        'mean': series.mean(),
        'median': series.median(),
        'min': series.min(),
        'max': series.max()
    }

def append_rows_and_style(ws, data, header=True):
    """Append rows of data to a worksheet and apply consistent styling."""
    for row in data:
        ws.append(row)
    if header:
        style_excel_sheet(ws)

def style_excel_sheet(ws):
    """Apply consistent styling to an Excel sheet."""
    # Style the header
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")

    # Adjust column widths
    for column_cells in ws.columns:
        length = max(len(str(cell.value)) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length

    # Add a table
    table = Table(displayName=f"Table_{ws.title.replace(' ', '_')}", ref=ws.dimensions)
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False, showLastColumn=False,
                           showRowStripes=True, showColumnStripes=True)
    table.tableStyleInfo = style
    ws.add_table(table)

def append_dataframe_to_sheet(ws, dataframe):
    """Append a dataframe to an Excel worksheet."""
    # Convert non-scalar columns (e.g., lists, dictionaries) to strings
    dataframe = dataframe.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, (list, dict)) else x))

    # Append rows to the worksheet
    for row in dataframe_to_rows(dataframe, index=False, header=True):
        ws.append(row)



def style_excel_sheet(ws):
    """Apply consistent styling to an Excel sheet."""
    for cell in ws[1]:  # Header row styling
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
    for column_cells in ws.columns:  # Auto-adjust column widths
        length = max(len(str(cell.value)) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length
    table = Table(displayName=f"Table_{ws.title.replace(' ', '_')}", ref=ws.dimensions)
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                           showLastColumn=False, showRowStripes=True, showColumnStripes=True)
    table.tableStyleInfo = style
    ws.add_table(table)

# Core Classes
class IntegerLPProblem:
    def __init__(self, num_variables: int, num_constraints: int):
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.objective = np.random.randint(-10, 11, size=self.num_variables)
        self.constraints = np.column_stack(
            (np.random.randint(-5, 6, size=(self.num_constraints, self.num_variables)),
             np.random.randint(1, 51, size=self.num_constraints))
        )
        self.bounds = [(0, 10) for _ in range(self.num_variables)]
        self.mathematical_solution = {}
        self.genetic_solution = {}

    def solve_mathematical(self):
        """Solve using PuLP (mathematical method)."""
        prob = pulp.LpProblem("Integer LP Problem", pulp.LpMinimize)
        vars = [pulp.LpVariable(f'x{i}', lowBound=self.bounds[i][0], upBound=self.bounds[i][1], cat='Integer')
                for i in range(self.num_variables)]
        prob += pulp.lpSum(self.objective[i] * vars[i] for i in range(self.num_variables))
        for constraint in self.constraints:
            prob += pulp.lpSum(constraint[j] * vars[j] for j in range(self.num_variables)) <= constraint[-1]

        start_time = time.time()
        prob.solve()
        solve_time = time.time() - start_time

        self.mathematical_solution = {
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'variables': [var.varValue for var in prob.variables()],
            'solve_time': solve_time
        }

    def solve_genetic(self):
        """Solve using Genetic Algorithm."""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, self.bounds[0][0], self.bounds[0][1])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=self.num_variables)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_ga_solution)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=self.bounds[0][0], up=self.bounds[0][1], indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start_time = time.time()
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=False)
        solve_time = time.time() - start_time

        best_solution = list(hof[0])
        self.genetic_solution = {
            'status': 'Optimal',
            'objective_value': calculate_objective_value(best_solution, self.objective),
            'variables': best_solution,
            'solve_time': solve_time,
            'fitness': hof[0].fitness.values[0]
        }

    def evaluate_ga_solution(self, individual):
        """Evaluate a GA solution by combining objective and penalty."""
        obj_value = calculate_objective_value(individual, self.objective)
        penalty = calculate_constraint_penalty(individual, self.constraints)
        return obj_value + penalty,

    def to_dict(self):
        return {
            'num_variables': self.num_variables,
            'num_constraints': self.num_constraints,
            'objective': self.objective.tolist(),
            'constraints': self.constraints.tolist(),
            'bounds': self.bounds,
            'mathematical_solution': self.mathematical_solution,
            'genetic_solution': self.genetic_solution
        }

class LPDatasetGenerator:
    def __init__(self, num_problems: int, min_variables: int, max_variables: int,
                 min_constraints: int, max_constraints: int):
        self.num_problems = num_problems
        self.min_variables = min_variables
        self.max_variables = max_variables
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.problems = []

    def generate_dataset(self):
        for _ in tqdm(range(self.num_problems), desc="Generating and solving problems"):
            problem = IntegerLPProblem(
                np.random.randint(self.min_variables, self.max_variables + 1),
                np.random.randint(self.min_constraints, self.max_constraints + 1)
            )
            problem.solve_mathematical()
            problem.solve_genetic()
            self.problems.append(problem.to_dict())

    def to_dataframe(self):
        return pd.DataFrame(self.problems)

class Experiment:
    def __init__(self, num_problems: int, min_variables: int, max_variables: int,
                 min_constraints: int, max_constraints: int):
        self.dataset_generator = LPDatasetGenerator(
            num_problems, min_variables, max_variables, min_constraints, max_constraints)
        self.problems_df = None
        self.results_combined = None
        self.metrics = None

    def run_experiment(self):
        """Run the experiment: generate dataset, solve problems, calculate metrics."""
        self.dataset_generator.generate_dataset()
        self.problems_df = self.dataset_generator.to_dataframe()
        self.results_combined = self._prepare_combined_results()
        self.metrics = self._compute_aggregate_metrics()
        return self.metrics

    def _prepare_combined_results(self):
        """Prepare detailed results for each problem."""
        combined_results = []
        for idx, problem in self.problems_df.iterrows():
            math_obj = problem['mathematical_solution']['objective_value']
            gen_obj = problem['genetic_solution']['objective_value']
            math_time = problem['mathematical_solution']['solve_time']
            gen_time = problem['genetic_solution']['solve_time']
            obj_diff, speedup = calculate_objective_and_speedup(math_obj, gen_obj, math_time, gen_time)
            combined_results.append({
                'problem_id': idx,
                'mathematical_objective_value': math_obj,
                'genetic_objective_value': gen_obj,
                'objective_difference': obj_diff,
                'mathematical_solve_time': math_time,
                'genetic_solve_time': gen_time,
                'speedup_factor': speedup
            })
        return pd.DataFrame(combined_results)

    def _compute_aggregate_metrics(self):
        """Calculate overall metrics from the combined results."""
        objective_differences = self.results_combined['objective_difference']
        speedup_factors = self.results_combined['speedup_factor']
        return {
            'mean_objective_difference': objective_differences.mean(),
            'median_objective_difference': objective_differences.median(),
            'max_objective_difference': objective_differences.max(),
            'mean_speedup': speedup_factors.mean(),
            'median_speedup': speedup_factors.median(),
            'max_speedup': speedup_factors.max(),
            'min_speedup': speedup_factors.min()
        }

class ExperimentExcelReport:
    def __init__(self, problems_df, results_combined, metrics):
        self.problems_df = problems_df
        self.results_combined = results_combined
        self.metrics = metrics

    def generate_excel_report(self, file_path="experiment_results.xlsx"):
        """Generate an Excel report with problem data, results, and metrics."""
        wb = Workbook()

        # Explanation Sheet
        ws_explanation = wb.active
        ws_explanation.title = "Explanation"
        explanations = [
            ("Sheet", "Description"),
            ("Explanation", "Overview of workbook contents."),
            ("Master List of Problems", "Detailed list of problems and solutions."),
            ("Combined Results", "Detailed performance results for each problem."),
            ("Overall Metrics", "Aggregate metrics comparing GA and mathematical solver."),
        ]
        self._append_list_to_sheet(ws_explanation, explanations)

        # Master List of Problems
        ws_master = wb.create_sheet("Master List of Problems")
        append_dataframe_to_sheet(ws_master, self.problems_df)

        # Combined Results
        ws_combined = wb.create_sheet("Combined Results")
        append_dataframe_to_sheet(ws_combined, self.results_combined)

        # Metrics Sheet
        ws_metrics = wb.create_sheet("Overall Metrics")
        metrics_data = [(k, v) for k, v in self.metrics.items()]
        self._append_list_to_sheet(ws_metrics, metrics_data)

        # Apply styling
        for sheet in wb.worksheets:
            style_excel_sheet(sheet)

        # Ensure the directory exists if provided
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save the workbook
        wb.save(file_path)
        print(f"\nExcel report saved to {file_path}")

    @staticmethod
    def _append_list_to_sheet(ws, data):
        """Append rows of list data to an Excel sheet."""
        for row in data:
            ws.append(row)



if __name__ == "__main__":
    initialize_random_seeds()

    # Run the experiment
    experiment = Experiment(num_problems=100, min_variables=2, max_variables=5, min_constraints=2, max_constraints=5)
    metrics = experiment.run_experiment()

    # Generate and save the Excel report
    report = ExperimentExcelReport(
        problems_df=experiment.problems_df,
        results_combined=experiment.results_combined,
        metrics=metrics
    )
    report.generate_excel_report("lp_experiment_results.xlsx")