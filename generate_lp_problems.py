import numpy as np
import pulp
import json
from tqdm import tqdm


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
            prob += pulp.lpSum(self.constraints[i, j] * vars[j] for j in range(self.num_variables)) <= self.constraints[
                i, -1]
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

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.problems, f)

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            self.problems = json.load(f)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    # Generate a dataset of 1000 problems
    generator = LPDatasetGenerator(num_problems=1000, min_variables=2, max_variables=5, min_constraints=2,
                                   max_constraints=5)
    generator.generate_dataset()

    # Save the dataset to a file
    generator.save_to_file('lp_dataset.json')

    print(f"Generated and saved {len(generator.problems)} problems to lp_dataset.json")

    # Example of loading the dataset
    new_generator = LPDatasetGenerator(0, 0, 0, 0, 0)  # Parameters don't matter for loading
    new_generator.load_from_file('lp_dataset.json')
    print(f"Loaded {len(new_generator.problems)} problems from lp_dataset.json")

    # Print the first problem as an example
    print("\nExample problem:")
    print(json.dumps(new_generator.problems[0], indent=2))