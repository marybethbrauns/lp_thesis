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

### PuLP

PuLP is an open-source linear programming library for Python. It provides a user-friendly interface to define and solve linear programming and integer programming problems. Key features of PuLP include:

1. Support for both continuous and integer variables.
2. Ability to interface with various solvers (e.g., GLPK, CPLEX, Gurobi).
3. Simple and intuitive syntax for defining objective functions and constraints.
4. Capability to handle large-scale optimization problems.

In this project, we use PuLP as the exact solver to find optimal solutions for our ILP problems, which we then compare with the solutions found by our genetic algorithm approach.

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
