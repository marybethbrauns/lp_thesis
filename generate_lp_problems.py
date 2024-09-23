import numpy as np
from scipy.optimize import linprog
import pulp

class IntegerLPProblem:
    def __init__(self, num_variables, num_constraints):
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.objective = None
        self.constraints = None
        self.bounds = None
        self.solution = None

    def generate_problem(self):
        # Generate objective coefficients
        self.objective = np.random.randint(-10, 11, size=self.num_variables)

        # Generate constraint coefficients and right-hand side
        self.constraints = np.random.randint(-5, 6, size=(self.num_constraints, self.num_variables))
        rhs = np.random.randint(1, 51, size=self.num_constraints)

        # Combine coefficients and RHS
        self.constraints = np.column_stack((self.constraints, rhs))

        # Set bounds for variables (0 to 10 for this example)
        self.bounds = [(0, 10) for _ in range(self.num_variables)]

    def solve(self):
        # Create a PuLP problem
        prob = pulp.LpProblem("Integer LP Problem", pulp.LpMinimize)

        # Define variables
        vars = [pulp.LpVariable(f'x{i}', lowBound=self.bounds[i][0], upBound=self.bounds[i][1], cat='Integer')
                for i in range(self.num_variables)]

        # Set objective
        prob += pulp.lpSum(self.objective[i] * vars[i] for i in range(self.num_variables))

        # Add constraints
        for i in range(self.num_constraints):
            prob += pulp.lpSum(self.constraints[i, j] * vars[j] for j in range(self.num_variables)) <= self.constraints[i, -1]

        # Solve the problem
        prob.solve()

        # Store the solution
        self.solution = {
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'variables': [var.varValue for var in prob.variables()]
        }

    def __str__(self):
        problem_str = "Integer Linear Programming Problem:\n"
        problem_str += f"Minimize: {' + '.join([f'{c}x{i}' for i, c in enumerate(self.objective)])}\n"
        problem_str += "Subject to:\n"
        for i, constraint in enumerate(self.constraints):
            problem_str += f"  {' + '.join([f'{c}x{j}' for j, c in enumerate(constraint[:-1])])} <= {constraint[-1]}\n"
        problem_str += f"Bounds: {self.bounds}\n"
        if self.solution:
            problem_str += f"Solution: {self.solution}\n"
        return problem_str

# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    problem = IntegerLPProblem(num_variables=3, num_constraints=2)
    problem.generate_problem()
    print("Generated problem:")
    print(problem)
    problem.solve()
    print("\nSolved problem:")
    print(problem)