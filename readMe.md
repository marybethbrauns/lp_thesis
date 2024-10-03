Mary Elizabeth Brauns Senior Thesis with Dr. David Murphy
Hillsdale College Department of Mathematics

# Integer Linear Programming Solver using Genetic Algorithm

## Overview

This project implements an Integer Linear Programming (ILP) solver using a Genetic Algorithm (GA) and compares its performance with a traditional exact solver (PuLP). The main goal is to evaluate the efficiency and accuracy of the GA approach in solving ILP problems.

## Background

### Integer Linear Programming (ILP)

Integer Linear Programming is a mathematical optimization technique used to find the best solution to a problem with the following characteristics:

1. The objective function is linear (e.g., maximize profit or minimize cost).
2. The constraints are linear inequalities or equalities.
3. Some or all of the variables are restricted to integer values.

ILP is widely used in various fields, including operations research, economics, and computer science, to solve complex optimization problems such as resource allocation, scheduling, and network flow problems.

## Mathematical Foundations of PuLP

PuLP is based on the mathematical principles of linear programming (LP) and integer linear programming (ILP). Here's an explanation of the underlying mathematics:

### Linear Programming

A linear programming problem is formulated as follows:

Optimize (Minimize or Maximize): 
```
Z = c₁x₁ + c₂x₂ + ... + cₙxₙ
```

Subject to constraints:
```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ ≤ b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ ≤ bₘ
```

And non-negativity constraints:
```
x₁, x₂, ..., xₙ ≥ 0
```

Where:
- Z is the objective function to be optimized
- x₁, x₂, ..., xₙ are the decision variables
- c₁, c₂, ..., cₙ are the coefficients of the objective function
- a₁₁, a₁₂, ..., aₘₙ are the coefficients of the constraints
- b₁, b₂, ..., bₘ are the right-hand side values of the constraints

### Integer Linear Programming

Integer Linear Programming (ILP) is an extension of LP where some or all of the variables are constrained to be integers. In a pure ILP problem, all variables are integers:

```
x₁, x₂, ..., xₙ ∈ ℤ
```

In a mixed integer linear programming (MILP) problem, only some variables are constrained to be integers.

### Solution Methods

PuLP can use various algorithms to solve these problems, including:

1. **Simplex Algorithm**: For solving LP problems. It moves along the vertices of the feasible region defined by the constraints, improving the objective function value at each step until the optimal solution is found.

2. **Branch and Bound**: For ILP problems. This method:
   - Solves the LP relaxation (ignoring integer constraints)
   - If the solution has non-integer values for integer variables, it branches the problem into subproblems
   - Continues branching and bounding until an optimal integer solution is found

3. **Cutting Plane Methods**: These add additional constraints to the LP relaxation to make the solution closer to integer values.

4. **Branch and Cut**: This combines branch and bound with cutting plane methods.

### Example in PuLP

Here's how a simple ILP problem might be formulated in PuLP:

```python
import pulp

# Create the model
model = pulp.LpProblem("Maximize Profit", pulp.LpMaximize)

# Define variables
x = pulp.LpVariable("x", lowBound=0, cat='Integer')
y = pulp.LpVariable("y", lowBound=0, cat='Integer')

# Define the objective function
model += 3*x + 2*y, "Profit"

# Define constraints
model += 2*x + y <= 100, "Labor"
model += x + y <= 80, "Materials"

# Solve the problem
model.solve()

# Print the results
print(f"x = {x.varValue}")
print(f"y = {y.varValue}")
print(f"Profit = {pulp.value(model.objective)}")
```

This example demonstrates how PuLP translates the mathematical formulation into Python code, making it easier for users to define and solve complex optimization problems.

## Key Components

1. **IntegerLPProblem**: A class that represents an individual ILP problem. It can generate random problems and solve them using PuLP.

2. **LPDatasetGenerator**: A class that generates a dataset of ILP problems.

3. **Genetic Algorithm Solver**: Implemented using the DEAP library, this solver attempts to find optimal or near-optimal solutions to ILP problems.

4. **Performance Comparison**: The script compares the GA solutions with exact solutions from PuLP in terms of both accuracy and solve time.

## Main Functions

- `evaluate()`: Fitness function for the GA.
- `calculate_objective_value()`: Calculates the objective value of a solution.
- `create_toolbox()`: Sets up the DEAP toolbox for the GA.
- `solve_with_ga()`: Solves a single problem using the GA.
- `process_dataset()`: Processes a set of problems using the GA.

## Workflow

1. Generate a dataset of ILP problems.
2. Split the dataset into training and testing sets.
3. Solve problems using both GA and PuLP.
4. Compare results in terms of solution quality and solve time.
5. Output detailed statistics and analysis.

## How to Use

1. Ensure all required libraries are installed:
   ```
   pip install numpy pandas tqdm pulp deap scikit-learn
   ```

2. Run the script:
   ```
   python <script_name>.py
   ```

3. The script will generate problems, solve them, and output results to the console. It will also save a CSV file with detailed results.

## Output

The script provides comprehensive output, including:

- Mean Absolute Error for training and testing sets
- Detailed results for each problem
- Analysis of mismatches between GA and optimal solutions
- Efficiency comparison between GA and PuLP
- Statistics on solve times and speedup factors

## Customization

You can customize the following parameters in the `__main__` section:

- Number of problems to generate
- Range of variables and constraints
- GA parameters (population size, number of generations, etc.)

## Notes

- The GA may not always find the optimal solution but can be faster than exact methods for some problems.
- The efficiency of the GA vs. PuLP can vary depending on the problem characteristics.
- This implementation is for educational and comparative purposes and may not be optimized for large-scale industrial use.
- While PuLP guarantees finding the optimal solution (if one exists), it may become computationally expensive for large or complex problems. The GA approach trades off some accuracy for potentially faster solve times, especially on larger problems.

## Future Improvements

- Implement more sophisticated GA operators
- Add support for different types of ILP problems (e.g., mixed-integer programming)
- Optimize GA parameters automatically based on problem characteristics
- Parallelize GA operations for improved performance
- Incorporate other metaheuristic algorithms for comparison (e.g., Simulated Annealing, Particle Swarm Optimization)

## Contributing

Contributions to improve the code or extend its functionality are welcome. Please submit pull requests or open issues for any bugs or feature requests.
