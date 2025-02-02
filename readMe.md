# Mary Elizabeth Brauns
## Senior Thesis with Dr. David Murphy  
### Hillsdale College Department of Mathematics  

# A Comparative Study of AI-Based Approaches to Linear Programming:  
## Classical Methods, Genetic Algorithms, and Neural Networks  

## Overview  
This project explores different approaches to solving linear programming (LP) problems. In particular, it compares the performance of a traditional exact solver (using SciPy's HiGHS simplex method) with two AI-based methods:

- A Genetic Algorithm (GA) implementation, which includes a warm-start technique to leverage information from previous similar problems.
- A Neural Network trained to approximate the optimal solution of LP problems.

The main goal is to evaluate each method in terms of solution quality, feasibility, and computational time. Additionally, the trade-offs between accuracy and speed are discussed in detail. Although AI-based methods offer flexibility and potential advantages in generalization and hybrid applications, they often come at the cost of increased computational overhead compared to specialized methods.

## Background  
### Linear Programming (LP)  
Linear Programming is an optimization technique used to find the best solution (maximum or minimum) for a linear objective function subject to a set of linear constraints. A general LP problem is formulated as follows:

#### Objective Function:  
\[ \text{Minimize: } Z = c_1 x_1 + c_2 x_2 + \dots + c_n x_n \]

#### Subject to Constraints:  
\[ a_{11} x_1 + a_{12} x_2 + \dots + a_{1n} x_n \leq b_1 \]
\[ a_{21} x_1 + a_{22} x_2 + \dots + a_{2n} x_n \leq b_2 \]
\[ \vdots \]
\[ a_{m1} x_1 + a_{m2} x_2 + \dots + a_{mn} x_n \leq b_m \]

#### And:  
\[ x_1, x_2, \dots, x_n \geq 0 \]

Classical LP solvers such as the Simplex Method or Interior-Point Methods exploit the problem’s linearity and convexity to obtain exact solutions rapidly.

### AI-Based Approaches  
While classical methods are highly optimized for linear problems, AI-based methods have their own merits, especially in more general or complex scenarios. Two notable approaches are:

#### Genetic Algorithms (GAs)  
GAs are heuristic, population-based search algorithms that evolve a set of candidate solutions over several generations using selection, crossover, and mutation operators. They are general-purpose and can be applied to a wide range of optimization problems but do not exploit the linear structure of LPs. When used on LPs, a GA typically requires many iterations to converge to a good solution, leading to increased computational time.

#### Neural Networks  
A neural network can be trained to learn the mapping from the parameters of an LP (such as the coefficients of the objective function and constraints) to the optimal solution. Once trained, the network can provide extremely fast approximations via a single forward pass. However, its accuracy is limited by the quality of the training data and the model architecture, and it does not guarantee exact optimality.

## Solution Methods  
### 1. Classical Solver: Simplex (HiGHS)  
The simplex method, as implemented in SciPy’s `linprog`, finds the optimal solution by moving along the vertices of the feasible region defined by the constraints. For small LP problems, the simplex method is extremely fast—often taking less than one millisecond.

#### Example Code:  
```python
from scipy.optimize import linprog
import time

def solve_lp_simplex(lp):
    c, A_ub, b_ub, bounds = lp["c"], lp["A_ub"], lp["b_ub"], lp["bounds"]
    start_time = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    solve_time = time.time() - start_time
    if res.success:
        return res.x, res.fun, solve_time
    else:
        return None, None, solve_time
```

### 2. Genetic Algorithm with Warm Starting  
A GA is employed to search for near-optimal solutions. In this implementation, a warm-start approach is used—meaning that the final population of one LP problem is passed as the initial population for the next. This can potentially reduce the number of generations needed for convergence.

### 3. Neural Network Surrogate Solver  
A neural network is trained to approximate the mapping from LP parameters to the optimal solution. This surrogate model is built using a feedforward network with a few hidden layers. 

#### Example Training Code:  
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

## Experimental Setup and Workflow  
### Dataset Generation:  
A large dataset of LP instances is generated. Each instance is solved using the simplex method to provide ground truth solutions.

### Testing and Comparison:  
For a set of test LP instances, the following is performed:

- Each LP is solved using the simplex method.
- The GA (with warm starting) is applied to find a near-optimal solution.
- The neural network provides an approximate solution via a forward pass.

The solution quality (objective value and feasibility) and computational time for each method are recorded.

## Discussion: Why Are Genetic Algorithms Slower?  
Despite their flexibility and general applicability, GAs tend to be slower than specialized methods like the simplex algorithm for several reasons:

- **Iterative Evolution**: Requires multiple generations to converge.
- **Fitness Evaluation Overhead**: Each candidate must be evaluated in each generation.
- **General-Purpose Nature**: Unlike simplex, GAs do not exploit the linear structure of LPs.
- **Algorithmic Overhead**: Involves additional operations like mutation and crossover.
- **Convergence Uncertainty**: No fixed iteration limit ensures convergence.

## Future Improvements  
- **Advanced GA Operators and Parallelization**: Implementing more sophisticated genetic operators and parallelizing fitness evaluations could reduce the overhead.
- **Optimizing Neural Network Inference**: Using TensorFlow Lite or ONNX Runtime may improve inference speed.
- **Hybrid Approaches**: Combining neural networks with classical solvers for improved performance.

## References  
- [SciPy linprog documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)  
- [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)  
- Literature on genetic algorithms, neural networks, and hybrid optimization methods.
