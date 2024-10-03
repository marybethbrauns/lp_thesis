import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pulp
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import train_test_split
import time
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment

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
        start_time = time.time()
        prob = pulp.LpProblem("Integer LP Problem", pulp.LpMinimize)
        vars = [pulp.LpVariable(f'x{i}', lowBound=self.bounds[i][0], upBound=self.bounds[i][1], cat='Integer')
                for i in range(self.num_variables)]
        prob += pulp.lpSum(self.objective[i] * vars[i] for i in range(self.num_variables))
        for i in range(self.num_constraints):
            prob += pulp.lpSum(self.constraints[i, j] * vars[j] for j in range(self.num_variables)) <= self.constraints[i, -1]
        prob.solve()
        end_time = time.time()
        self.solution = {
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'variables': [var.varValue for var in prob.variables()],
            'solve_time': end_time - start_time
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

class GeneticAlgorithmSolver:
    def __init__(self, pop_size=50, ngen=50, cxpb=0.7, mutpb=0.2):
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

    @staticmethod
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

    @staticmethod
    def calculate_objective_value(solution, problem):
        return sum(x * c for x, c in zip(solution, problem['objective']))

    def create_toolbox(self, num_variables, bounds):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, bounds[0], bounds[1])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_variables)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=bounds[0], up=bounds[1], indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def solve(self, problem):
        num_variables = problem['num_variables']
        bounds = problem['bounds'][0]  # Assuming all variables have the same bounds

        toolbox = self.create_toolbox(num_variables, bounds)
        toolbox.register("evaluate", self.evaluate, problem=problem)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start_time = time.time()
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                       ngen=self.ngen, stats=stats, halloffame=hof, verbose=False)
        end_time = time.time()

        return {
            'best_solution': hof[0],
            'best_fitness': hof[0].fitness.values[0],
            'predicted_objective': self.calculate_objective_value(hof[0], problem),
            'solve_time': end_time - start_time
        }

class Experiment:
    def __init__(self, num_problems, min_variables, max_variables, min_constraints, max_constraints):
        self.dataset_generator = LPDatasetGenerator(num_problems, min_variables, max_variables,
                                                    min_constraints, max_constraints)
        self.ga_solver = GeneticAlgorithmSolver()
        self.problems_df = None
        self.train_df = None
        self.test_df = None
        self.train_results = None
        self.test_results = None
        self.results_combined = None

    def generate_dataset(self):
        self.dataset_generator.generate_dataset()
        self.problems_df = self.dataset_generator.to_dataframe()

    def split_dataset(self, test_size=0.2, random_state=42):
        self.train_df, self.test_df = train_test_split(self.problems_df, test_size=test_size, random_state=random_state)

    def process_dataset(self, df, is_training=True):
        results = []
        for _, problem in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            ga_result = self.ga_solver.solve(problem)
            results.append({
                'problem_id': _,
                'best_solution': ga_result['best_solution'],
                'best_fitness': ga_result['best_fitness'],
                'predicted_objective': ga_result['predicted_objective'],
                'ga_solve_time': ga_result['solve_time'],
                'optimal_solution': problem['solution']['variables'],
                'optimal_objective': problem['solution']['objective_value'],
                'pulp_solve_time': problem['solution']['solve_time']
            })
        return pd.DataFrame(results)

    def run_experiment(self):
        self.generate_dataset()
        self.split_dataset()
        print(f"Training set size: {len(self.train_df)}")
        print(f"Testing set size: {len(self.test_df)}")
        self.train_results = self.process_dataset(self.train_df, is_training=True)
        self.test_results = self.process_dataset(self.test_df, is_training=False)

    def calculate_metrics(self):
        train_mae = np.mean(np.abs(self.train_results['predicted_objective'] - self.train_results['optimal_objective']))
        test_mae = np.mean(np.abs(self.test_results['predicted_objective'] - self.test_results['optimal_objective']))
        print(f"\nTraining Mean Absolute Error: {train_mae}")
        print(f"Testing Mean Absolute Error: {test_mae}")

        self.results_combined = pd.DataFrame({
            'problem_id': self.test_results['problem_id'],
            'predicted_objective_value': self.test_results['predicted_objective'],
            'actual_objective_value': self.test_results['optimal_objective'],
            'solution_fitness': self.test_results['best_fitness'],
            'ga_solve_time': self.test_results['ga_solve_time'],
            'pulp_solve_time': self.test_results['pulp_solve_time']
        })

        self.results_combined['time_difference'] = self.results_combined['pulp_solve_time'] - self.results_combined['ga_solve_time']
        self.results_combined['ga_faster'] = self.results_combined['time_difference'] > 0
        self.results_combined['speedup_factor'] = self.results_combined['pulp_solve_time'] / self.results_combined['ga_solve_time']

        mismatches = self.results_combined[self.results_combined['predicted_objective_value'] != self.results_combined['actual_objective_value']]
        total_problems = len(self.results_combined)
        mismatch_percentage = (len(mismatches) / total_problems) * 100
        overall_avg_difference = np.mean(np.abs(self.results_combined['predicted_objective_value'] - self.results_combined['actual_objective_value']))
        ga_faster_count = self.results_combined['ga_faster'].sum()
        avg_speedup = self.results_combined['speedup_factor'].mean()
        median_speedup = self.results_combined['speedup_factor'].median()

        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'mismatch_percentage': mismatch_percentage,
            'overall_avg_difference': overall_avg_difference,
            'ga_faster_count': ga_faster_count,
            'ga_faster_percentage': (ga_faster_count / total_problems) * 100,
            'avg_speedup': avg_speedup,
            'median_speedup': median_speedup,
            'mismatches': mismatches,
            'total_problems': total_problems
        }

    def generate_excel_report(self, metrics):
        wb = Workbook()
        ws = wb.active
        ws.title = "Overall Metrics"

        metrics_data = [
            ("Training Mean Absolute Error", metrics['train_mae']),
            ("Testing Mean Absolute Error", metrics['test_mae']),
            ("Percentage of problems where GA didn't find the optimal solution", metrics['mismatch_percentage']),
            ("Overall average absolute difference", metrics['overall_avg_difference']),
            ("Number of problems where GA was faster", metrics['ga_faster_count']),
            ("Percentage of problems where GA was faster", metrics['ga_faster_percentage']),
            ("Average speedup factor (PuLP time / GA time)", metrics['avg_speedup']),
            ("Median speedup factor", metrics['median_speedup'])
        ]

        for row, (metric, value) in enumerate(metrics_data, start=1):
            ws.cell(row=row, column=1, value=metric)
            ws.cell(row=row, column=2, value=value)
            ws.cell(row=row, column=2).number_format = '0.00'

        ws_combined = wb.create_sheet("Combined Results")
        for r in dataframe_to_rows(self.results_combined, index=False, header=True):
            ws_combined.append(r)

        ws_top_speedup = wb.create_sheet("Top 5 Speedup")
        top_speedup = self.results_combined.nlargest(5, 'speedup_factor')[['problem_id', 'speedup_factor', 'ga_solve_time', 'pulp_solve_time']]
        for r in dataframe_to_rows(top_speedup, index=False, header=True):
            ws_top_speedup.append(r)

        ws_bottom_speedup = wb.create_sheet("Bottom 5 Speedup")
        bottom_speedup = self.results_combined.nsmallest(5, 'speedup_factor')[['problem_id', 'speedup_factor', 'ga_solve_time', 'pulp_solve_time']]
        for r in dataframe_to_rows(bottom_speedup, index=False, header=True):
            ws_bottom_speedup.append(r)

        ws_mismatches = wb.create_sheet("Mismatches")
        if not metrics['mismatches'].empty:
            for r in dataframe_to_rows(metrics['mismatches'], index=False, header=True):
                ws_mismatches.append(r)
        else:
            ws_mismatches['A1'] = "No mismatches found. The genetic algorithm found the optimal solution for all test problems."

        for sheet in wb:
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for column in sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column_letter].width = adjusted_width

        excel_filename = '/Users/marybeth/Desktop/p_solver_results.xlsx'
        wb.save(excel_filename)
        print(f"\nKey metrics and results have been saved to {excel_filename}")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    experiment = Experiment(num_problems=1000, min_variables=2, max_variables=5,
                            min_constraints=2, max_constraints=5)
    experiment.run_experiment()
    metrics = experiment.calculate_metrics()
    experiment.generate_excel_report(metrics)

    print("\nExperiment completed. Check the Excel file for detailed results.")