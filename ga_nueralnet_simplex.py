#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LP Solver Comparison Framework

This program compares three methods for solving linear programming problems:
1. Simplex (HiGHS) - Standard optimization approach
2. Neural Network - ML approach (trained specifically for each problem size)
3. Genetic Algorithm - Evolutionary optimization approach

IMPORTANT: Each neural network is trained and tested ONLY on problems with the
same number of variables and constraints. There is NO cross-testing between configurations.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os
import io
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras import layers, models, callbacks
from scipy.optimize import linprog
from scipy import stats
from multiprocessing import Pool, cpu_count

# Ensure TensorFlow warnings are minimized
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set up output directory
OUTPUT_DIR = "lp_solver_results"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_DIR, f"run_{RUN_TIMESTAMP}")

# LP problem parameters
LP_LOWER_BOUND = 0  # Lower bound for variables
LP_UPPER_BOUND = 10  # Upper bound for variables
DEFAULT_NUM_TRAIN = 100  # Default training samples per configuration
DEFAULT_NUM_TEST = 15  # Default test problems per configuration

# Neural network parameters
NN_EPOCHS = 20
NN_BATCH_SIZE = 64
NN_LAYERS = [128, 64, 32]  # Hidden layer sizes

# Genetic algorithm parameters
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 200
GA_MUTATION_RATE = 0.1
GA_EARLY_STOP = 20  # Stop if no improvement for this many generations


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def timestamp_msg(msg):
    """Add timestamp to message"""
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"


def print_header(msg, char='=', width=80):
    """Print a header with the message centered"""
    print("\n" + char * width)
    print(msg.center(width))
    print(char * width)


def print_section(msg):
    """Print a section header"""
    print(f"\n{'-' * 40}")
    print(f"{msg}")
    print(f"{'-' * 40}")


def save_config(config, filepath):
    """Save configuration as JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def setup_run(args):
    """Set up the run directory and save configuration"""
    # Create output directories
    ensure_dir(OUTPUT_DIR)
    ensure_dir(RUN_DIR)

    # Save run configuration
    config = {
        'timestamp': RUN_TIMESTAMP,
        'args': vars(args),
        'lp_params': {
            'lower_bound': LP_LOWER_BOUND,
            'upper_bound': LP_UPPER_BOUND,
            'num_train': args.train_size or DEFAULT_NUM_TRAIN,
            'num_test': args.test_size or DEFAULT_NUM_TEST
        },
        'nn_params': {
            'epochs': NN_EPOCHS,
            'batch_size': NN_BATCH_SIZE,
            'layers': NN_LAYERS
        },
        'ga_params': {
            'population_size': GA_POPULATION_SIZE,
            'generations': GA_GENERATIONS,
            'mutation_rate': GA_MUTATION_RATE,
            'early_stop': GA_EARLY_STOP
        }
    }

    save_config(config, os.path.join(RUN_DIR, 'run_config.json'))
    return config


def create_log(filepath):
    """Create a log file that will capture both stdout and file output"""

    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    import sys
    sys.stdout = Logger(filepath)
    print(timestamp_msg(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(f"Results will be saved to: {RUN_DIR}")


# ============================================================================
# LP PROBLEM GENERATION
# ============================================================================

def generate_lp_problem(n_vars, n_constraints, lower_bound=LP_LOWER_BOUND, upper_bound=LP_UPPER_BOUND):
    """
    Generate a random linear programming problem that is guaranteed to be feasible.

    The LP has the form:
    minimize     c^T x
    subject to   A_ub x <= b_ub
                x in [lower_bound, upper_bound]^n_vars

    Returns:
        Dictionary containing the LP parameters
    """
    # Generate a feasible point to ensure problem has a solution
    x_feasible = np.random.uniform(lower_bound, upper_bound, size=n_vars)

    # Generate constraint matrix with random coefficients
    A_ub = np.random.uniform(0, 10, size=(n_constraints, n_vars))

    # Add slack to ensure strict feasibility
    slack = np.random.uniform(0, 10, size=n_constraints)
    b_ub = A_ub.dot(x_feasible) + slack

    # Generate objective function coefficients
    c = np.random.uniform(-10, 10, size=n_vars)

    # Variable bounds
    bounds = [(lower_bound, upper_bound) for _ in range(n_vars)]

    # Calculate problem statistics
    density = np.count_nonzero(A_ub) / (n_constraints * n_vars)
    avg_constraint_value = np.mean(A_ub.dot(x_feasible))

    problem_stats = {
        "constraint_density": density,
        "avg_slack": np.mean(slack),
        "avg_constraint_value": avg_constraint_value,
        "objective_range": (np.min(c), np.max(c))
    }

    return {
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "bounds": bounds,
        "x_feasible": x_feasible,
        "stats": problem_stats
    }


def describe_lp(lp, include_matrices=False):
    """Return a string description of the LP problem"""
    n_vars = len(lp["c"])
    n_constraints = len(lp["b_ub"])

    description = [
        f"LP Problem: {n_vars} variables, {n_constraints} constraints",
        f"Objective range: {np.min(lp['c']):.2f} to {np.max(lp['c']):.2f}",
        f"Average slack: {np.mean(lp['b_ub'] - np.dot(lp['A_ub'], lp['x_feasible'])):.2f}",
        f"Variable bounds: {lp['bounds'][0]}"
    ]

    if include_matrices:
        description.extend([
            "\nObjective coefficients:",
            str(lp["c"]),
            "\nFirst few constraints:",
            str(lp["A_ub"][:min(3, n_constraints)]),
            "\nFirst few RHS values:",
            str(lp["b_ub"][:min(3, n_constraints)])
        ])

    return "\n".join(description)


# ============================================================================
# METHOD 1: SIMPLEX (HiGHS) SOLVER
# ============================================================================

def solve_simplex(lp, verbose=False):
    """
    Solve LP using the HiGHS simplex method

    Args:
        lp: Dictionary containing LP parameters
        verbose: Whether to print detailed info

    Returns:
        Tuple of (solution, objective, solve_time, simplex_stats)
    """
    c, A_ub, b_ub, bounds = lp["c"], lp["A_ub"], lp["b_ub"], lp["bounds"]

    if verbose:
        print("  Solving with Simplex (HiGHS) method...")

    # Solve the LP
    start_time = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    solve_time = time.time() - start_time

    if res.success:
        solution = res.x
        objective = res.fun
        iterations = res.nit
        status = res.message

        if verbose:
            print(f"    Simplex solved successfully in {solve_time:.6f} seconds, {iterations} iterations")
            print(f"    Objective value: {objective:.6f}")
    else:
        solution = None
        objective = None
        iterations = res.nit if hasattr(res, 'nit') else None
        status = res.message if hasattr(res, 'message') else "Failed"

        if verbose:
            print(f"    Simplex failed: {status}")

    simplex_stats = {
        "iterations": iterations,
        "status": status
    }

    return solution, objective, solve_time, simplex_stats


# ============================================================================
# NEURAL NETWORK APPROACH
# ============================================================================

def flatten_lp(lp, normalize=True):
    """
    Flatten LP parameters into a feature vector

    Args:
        lp: Dictionary containing LP parameters
        normalize: Whether to normalize features

    Returns:
        Numpy array with flattened features
    """
    # Extract the components
    c = lp["c"]
    A_ub = lp["A_ub"].flatten()
    b_ub = lp["b_ub"]

    # Concatenate into a single vector
    features = np.concatenate([c, A_ub, b_ub])

    # Optionally normalize
    if normalize:
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            features = (features - mean) / std

    return features


def create_training_data(n_vars, n_constraints, num_samples, verbose=True):
    """
    Create training data for neural network specifically for this problem size

    Args:
        n_vars: Number of variables
        n_constraints: Number of constraints
        num_samples: Number of training samples to generate
        verbose: Whether to print progress

    Returns:
        X, Y arrays for training
    """
    if verbose:
        print(f"Creating {num_samples} training samples for {n_vars}x{n_constraints} problems...")
        print(f"IMPORTANT: This training data will ONLY be used for {n_vars}x{n_constraints} problems")

    X = []
    Y = []

    # Use progress bar for larger samples
    progress_bar = tqdm(total=num_samples) if verbose else None

    while len(X) < num_samples:
        # Generate a random LP problem
        lp = generate_lp_problem(n_vars, n_constraints)

        # Solve it with simplex to get the "ground truth"
        solution, objective, _, _ = solve_simplex(lp)

        if solution is not None:
            # Add to training data
            X.append(flatten_lp(lp))
            Y.append(solution)

            # Update progress
            if progress_bar:
                progress_bar.update(1)

            # Print progress at intervals
            if verbose and len(X) % (max(1, num_samples // 10)) == 0:
                print(f"  Progress: {len(X)}/{num_samples} samples created")

    if progress_bar:
        progress_bar.close()

    if verbose:
        print(f"Training data generation complete: {len(X)} samples")

    return np.array(X), np.array(Y)


def build_nn_model(input_dim, output_dim, hidden_layers=NN_LAYERS):
    """
    Build neural network model for a specific LP configuration

    Args:
        input_dim: Input dimension (flattened LP size)
        output_dim: Output dimension (number of variables)
        hidden_layers: List of hidden layer sizes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()

    # Input layer
    model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation='relu'))

    # Output layer (linear activation for regression)
    model.add(layers.Dense(output_dim, activation='linear'))

    # Compile with mean squared error loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def train_nn_model(n_vars, n_constraints, num_train, verbose=True):
    """
    Train neural network for a specific problem configuration

    Args:
        n_vars: Number of variables
        n_constraints: Number of constraints
        num_train: Number of training samples
        verbose: Whether to print progress

    Returns:
        Trained model, training history, and metadata
    """
    if verbose:
        print_section(f"Training Neural Network for {n_vars}x{n_constraints} Configuration")
        print(f"This model will ONLY be used for {n_vars}x{n_constraints} problems")

    # Generate training data specifically for this configuration
    X_train, Y_train = create_training_data(n_vars, n_constraints, num_train, verbose)

    input_dim = X_train.shape[1]
    output_dim = n_vars

    if verbose:
        print(f"Building neural network with architecture: {input_dim} -> {NN_LAYERS} -> {output_dim}")

    # Build the model
    model = build_nn_model(input_dim, output_dim, NN_LAYERS)

    # Save model summary as string
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
    model_summary = summary_io.getvalue()

    if verbose:
        print(model_summary)

    # Add early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    training_start = time.time()

    if verbose:
        print(f"Training neural network with {NN_EPOCHS} epochs, batch size {NN_BATCH_SIZE}...")

    history = model.fit(
        X_train, Y_train,
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1 if verbose else 0
    )

    training_time = time.time() - training_start

    if verbose:
        print(f"Neural network training completed in {training_time:.2f} seconds")

    # Create metadata
    metadata = {
        "configuration": f"{n_vars}x{n_constraints}",
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_layers": NN_LAYERS,
        "epochs": NN_EPOCHS,
        "batch_size": NN_BATCH_SIZE,
        "training_samples": num_train,
        "training_time": training_time,
        "final_loss": history.history['loss'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "model_summary": model_summary
    }

    return model, history, metadata


def evaluate_nn(model, lp, verbose=False):
    """
    Evaluate neural network on an LP problem

    Args:
        model: Trained neural network model
        lp: Dictionary containing LP parameters
        verbose: Whether to print detailed info

    Returns:
        Tuple of (solution, objective, inference_time, is_feasible, constraint_stats)
    """
    if verbose:
        print("  Evaluating neural network model...")

    # Flatten and normalize LP
    X_input = flatten_lp(lp)
    X_input = np.expand_dims(X_input, axis=0)  # Add batch dimension

    # Time the prediction
    start_time = time.time()
    prediction = model.predict(X_input, verbose=0)
    inference_time = time.time() - start_time

    # Extract solution
    solution = prediction.flatten()

    # Clip solution to bounds
    bounds = lp["bounds"]
    for i in range(len(solution)):
        solution[i] = max(bounds[i][0], min(solution[i], bounds[i][1]))

    # Calculate objective value
    c = lp["c"]
    objective = np.dot(c, solution)

    # Check feasibility
    A_ub = lp["A_ub"]
    b_ub = lp["b_ub"]

    constraints = np.dot(A_ub, solution) - b_ub
    max_violation = np.max(constraints)
    is_feasible = max_violation <= 1e-6

    # Calculate constraint statistics
    constraint_stats = {
        "max_violation": max_violation,
        "avg_violation": np.mean(constraints[constraints > 0]) if np.any(constraints > 0) else 0,
        "num_violated": np.sum(constraints > 0)
    }

    if verbose:
        print(f"    NN inference time: {inference_time:.6f} seconds")
        print(f"    Objective value: {objective:.6f}")
        print(f"    Feasible: {is_feasible}, Max constraint violation: {max_violation:.6f}")

    return solution, objective, inference_time, is_feasible, constraint_stats


def plot_nn_training_history(history, config_dir):
    """
    Plot the neural network training history

    Args:
        history: Training history object
        config_dir: Directory to save plot

    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plot_path = os.path.join(config_dir, 'nn_training_history.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# ============================================================================
# GENETIC ALGORITHM APPROACH
# ============================================================================

def fitness_function(x, c, A_ub, b_ub, penalty_factor=1000):
    """
    Calculate fitness for genetic algorithm with penalty for constraint violations

    Args:
        x: Solution vector
        c: Objective coefficients
        A_ub: Constraint coefficients
        b_ub: Constraint right-hand sides
        penalty_factor: Factor for penalizing violations

    Returns:
        Fitness value (lower is better)
    """
    # Calculate objective value
    objective = np.dot(c, x)

    # Calculate constraint violations
    constraints = np.dot(A_ub, x) - b_ub
    violations = constraints[constraints > 0]

    # Apply penalty for violations
    penalty = 0
    if len(violations) > 0:
        penalty = penalty_factor * np.sum(violations ** 2)

    # Total fitness is objective plus penalty
    return objective + penalty


def solve_ga(lp, population_size=GA_POPULATION_SIZE, generations=GA_GENERATIONS,
             mutation_rate=GA_MUTATION_RATE, early_stop=GA_EARLY_STOP,
             initial_population=None, verbose=False):
    """
    Solve LP using genetic algorithm

    Args:
        lp: Dictionary containing LP parameters
        population_size: Size of GA population
        generations: Maximum number of generations
        mutation_rate: Probability of mutation
        early_stop: Stop if no improvement for this many generations
        initial_population: Optional warm start population
        verbose: Whether to print detailed info

    Returns:
        Tuple of (solution, objective, solve_time, population, ga_stats)
    """
    if verbose:
        print("  Solving with Genetic Algorithm...")

    c, A_ub, b_ub, bounds = lp["c"], lp["A_ub"], lp["b_ub"], lp["bounds"]
    n_vars = len(c)

    # Initialize population
    population = []

    # Use warm start if provided
    if initial_population is not None:
        if verbose:
            print(f"    Using warm start with {len(initial_population)} individuals")

        for individual in initial_population:
            clipped = [max(bounds[i][0], min(individual[i], bounds[i][1])) for i in range(n_vars)]
            population.append(clipped)

        # Fill remaining population with random individuals
        while len(population) < population_size:
            individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_vars)]
            population.append(individual)

        # Truncate if needed
        population = population[:population_size]
    else:
        if verbose:
            print(f"    Initializing random population of size {population_size}")

        for _ in range(population_size):
            individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_vars)]
            population.append(individual)

    # Mutation function with adaptive rate
    def mutate(individual, generation):
        # Decrease mutation strength over time
        scale = 1.0 - 0.5 * (generation / generations) if generations > 0 else 0.5

        for i in range(n_vars):
            if random.random() < mutation_rate:
                # Add Gaussian noise with decreasing scale
                individual[i] += random.gauss(0, scale)
                # Clip to bounds
                individual[i] = max(bounds[i][0], min(individual[i], bounds[i][1]))

        return individual

    # Crossover function (uniform crossover)
    def crossover(parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(n_vars):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]

        return child1, child2

    # Selection function (tournament selection)
    def tournament_selection(pop, tournament_size=3):
        candidates_idx = random.sample(range(len(pop)), tournament_size)
        candidates = [pop[idx] for idx in candidates_idx]
        fitnesses = [fitness_function(ind, c, A_ub, b_ub) for ind in candidates]
        min_idx = fitnesses.index(min(fitnesses))
        return candidates[min_idx]

    # Statistics tracking
    best_individual = None
    best_fitness = float('inf')
    start_time = time.time()
    generations_without_improvement = 0
    history = {
        "best_fitness": [],
        "avg_fitness": [],
        "constraint_violations": [],
        "diversity": []
    }

    # Main GA loop
    for gen in range(generations):
        # Calculate fitness for all individuals
        fitness_values = [fitness_function(ind, c, A_ub, b_ub) for ind in population]

        # Sort population by fitness
        population_with_fitness = list(zip(population, fitness_values))
        population_with_fitness.sort(key=lambda x: x[1])
        sorted_population = [ind for ind, _ in population_with_fitness]
        sorted_fitness = [fit for _, fit in population_with_fitness]

        # Update best individual
        current_best = sorted_population[0]
        current_fitness = sorted_fitness[0]

        # Track statistics
        avg_fitness = np.mean(sorted_fitness)

        # Calculate constraint violations
        violations = [np.sum(np.maximum(0, np.dot(A_ub, ind) - b_ub)) for ind in population]
        avg_violations = np.mean(violations)

        # Calculate population diversity (average pairwise distance)
        if len(population) > 1:
            sample_size = min(10, len(population))
            sample_indices = random.sample(range(len(population)), sample_size)
            sample_pop = [population[i] for i in sample_indices]
            distances = []

            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    dist = np.linalg.norm(np.array(sample_pop[i]) - np.array(sample_pop[j]))
                    distances.append(dist)

            diversity = np.mean(distances) if distances else 0
        else:
            diversity = 0

        # Update history
        history["best_fitness"].append(current_fitness)
        history["avg_fitness"].append(avg_fitness)
        history["constraint_violations"].append(avg_violations)
        history["diversity"].append(diversity)

        # Print progress at intervals
        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(f"    Generation {gen + 1}/{generations}: Best fitness = {current_fitness:.6f}, "
                  f"Avg fitness = {avg_fitness:.6f}, Violations = {avg_violations:.6f}")

        # Check for improvement
        improved = False
        improvement_threshold = 1e-6

        if current_fitness < best_fitness - improvement_threshold:
            best_fitness = current_fitness
            best_individual = current_best.copy()
            generations_without_improvement = 0
            improved = True
        else:
            generations_without_improvement += 1

        # Early stopping
        if generations_without_improvement >= early_stop:
            if verbose:
                print(f"    Early stopping after {gen + 1} generations (no improvement for {early_stop} generations)")
            break

        # Create new population
        new_population = []

        # Elitism: keep the best individuals
        elite_count = max(1, int(population_size * 0.1))
        new_population.extend(sorted_population[:elite_count])

        # Fill the rest with crossover and mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if parent1 != parent2:  # Avoid self-crossover
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, gen)
                child2 = mutate(child2, gen)
                new_population.extend([child1, child2])
            else:
                # If same parent selected, just mutate
                child = mutate(parent1.copy(), gen)
                new_population.append(child)

        # Truncate if needed
        population = new_population[:population_size]

    # End of GA
    solve_time = time.time() - start_time

    # Use best individual found during the run
    if best_individual is None:
        best_individual = sorted_population[0]

    # Check feasibility of best solution
    constraints = np.dot(A_ub, best_individual) - b_ub
    max_violation = np.max(constraints)
    is_feasible = max_violation <= 1e-6

    if is_feasible:
        objective = np.dot(c, best_individual)
        if verbose:
            print(f"    GA found feasible solution in {solve_time:.4f} seconds")
            print(f"    Objective value: {objective:.6f}")
    else:
        objective = None
        if verbose:
            print(f"    GA failed to find feasible solution, max violation: {max_violation:.6f}")

    # Compute additional stats
    constraint_stats = {
        "max_violation": max_violation,
        "avg_violation": np.mean(constraints[constraints > 0]) if np.any(constraints > 0) else 0,
        "num_violated": np.sum(constraints > 0)
    }

    ga_stats = {
        "generations": gen + 1,
        "history": history,
        "constraint_stats": constraint_stats,
        "early_stopped": generations_without_improvement >= early_stop
    }

    return best_individual, objective, solve_time, population, ga_stats


def plot_ga_convergence(ga_stats, config_dir):
    """
    Plot GA convergence history

    Args:
        ga_stats: Statistics from GA run
        config_dir: Directory to save plot

    Returns:
        Path to saved plot
    """
    history = ga_stats["history"]

    plt.figure(figsize=(12, 8))

    # Plot best fitness over generations
    plt.subplot(2, 2, 1)
    plt.plot(history["best_fitness"])
    plt.title('Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.grid(True)

    # Plot average fitness over generations
    plt.subplot(2, 2, 2)
    plt.plot(history["avg_fitness"])
    plt.title('Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.grid(True)

    # Plot constraint violations over generations
    plt.subplot(2, 2, 3)
    plt.plot(history["constraint_violations"])
    plt.title('Average Constraint Violations')
    plt.xlabel('Generation')
    plt.ylabel('Violation Magnitude')
    plt.grid(True)

    # Plot population diversity over generations
    plt.subplot(2, 2, 4)
    plt.plot(history["diversity"])
    plt.title('Population Diversity')
    plt.xlabel('Generation')
    plt.ylabel('Average Pairwise Distance')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(config_dir, 'ga_convergence.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# ============================================================================
# METRICS AND COMPARISON
# ============================================================================

def compute_comparison_metrics(method_time, method_obj, simplex_time, simplex_obj, method_constraints=None):
    """
    Compute comparison metrics between a method and simplex

    Args:
        method_time: Solution time for method
        method_obj: Objective value for method
        simplex_time: Solution time for simplex
        simplex_obj: Objective value for simplex
        method_constraints: Optional constraint statistics

    Returns:
        Dictionary of comparison metrics
    """
    # Time comparison
    time_diff = method_time - simplex_time
    time_ratio = method_time / simplex_time if simplex_time > 0 else float('inf')
    speedup = simplex_time / method_time if method_time > 0 else float('inf')

    # Objective comparison
    if simplex_obj is None or method_obj is None:
        accuracy = None
        obj_ratio = None
        obj_diff = None
    elif abs(simplex_obj) < 1e-6:
        # Handle case where simplex objective is very close to zero
        accuracy = 100.0 if abs(method_obj) < 1e-6 else 0.0
        obj_ratio = 1.0 if abs(method_obj) < 1e-6 else float('inf')
        obj_diff = abs(method_obj - simplex_obj)
    else:
        # Regular case
        accuracy = max(0.0, 100 - (abs(method_obj - simplex_obj) / abs(simplex_obj)) * 100)
        obj_ratio = method_obj / simplex_obj
        obj_diff = method_obj - simplex_obj

    # Quick comparison
    faster_than_simplex = method_time < simplex_time

    metrics = {
        "time_diff": time_diff,
        "time_ratio": time_ratio,
        "speedup": speedup,
        "objective_diff": obj_diff,
        "objective_ratio": obj_ratio,
        "accuracy": accuracy,
        "faster_than_simplex": faster_than_simplex
    }

    # Include constraint stats if available
    if method_constraints:
        metrics.update(method_constraints)

    return metrics


# ============================================================================
# EXPERIMENT FRAMEWORK
# ============================================================================

def run_experiment_for_configuration(n_vars, n_constraints, num_train, num_test, verbose=True):
    """
    Run a complete experiment for a specific problem configuration

    Args:
        n_vars: Number of variables
        n_constraints: Number of constraints
        num_train: Number of training samples
        num_test: Number of test problems
        verbose: Whether to print detailed info

    Returns:
        Tuple of (config_summary, detailed_results, problems_info, plots)
    """
    if verbose:
        print_header(f"EXPERIMENT: {n_vars} variables, {n_constraints} constraints")
        print("IMPORTANT: The neural network will be trained and tested ONLY on this configuration")

    # Create directory for this configuration
    config_dir = os.path.join(RUN_DIR, f"config_{n_vars}x{n_constraints}")
    ensure_dir(config_dir)

    # STEP 1: Train neural network for this specific configuration
    start_time = time.time()
    nn_model, nn_history, nn_metadata = train_nn_model(n_vars, n_constraints, num_train, verbose)

    # Plot training history
    nn_training_plot = plot_nn_training_history(nn_history, config_dir)

    # STEP 2: Test on random problems of the same configuration
    if verbose:
        print_section(f"Testing {num_test} problems for {n_vars}x{n_constraints} configuration")
        print(f"Using neural network trained ONLY for {n_vars}x{n_constraints} problems")

    # Data containers
    detailed_results = []
    problems_info = []
    summary_metrics = {
        "Simplex": {"times": [], "objectives": [], "iterations": []},
        "Neural Network": {"times": [], "objectives": [], "accuracies": [],
                           "faster": [], "constraint_violations": []},
        "Genetic Algorithm": {"times": [], "objectives": [], "accuracies": [],
                              "faster": [], "generations": [], "constraint_violations": []}
    }

    # For warm starting GA
    warm_start_population = None

    # Run tests
    for i in range(num_test):
        if verbose:
            print(f"\nTest Problem {i + 1}/{num_test} for {n_vars}x{n_constraints}")

        # Generate random LP problem
        lp = generate_lp_problem(n_vars, n_constraints)
        lp_id = f"Config_{n_vars}x{n_constraints}_Test_{i + 1}"

        # Record problem details
        problems_info.append({
            "LP_ID": lp_id,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "c": str(lp["c"]),
            "A_ub": str(lp["A_ub"].tolist()),
            "b_ub": str(lp["b_ub"].tolist()),
            "Bounds": str(lp["bounds"]),
            "Problem_Stats": str(lp["stats"])
        })

        # --- Solve using Simplex ---
        sol_s, obj_s, time_s, simplex_stats = solve_simplex(lp, verbose)

        summary_metrics["Simplex"]["times"].append(time_s)
        summary_metrics["Simplex"]["objectives"].append(obj_s)
        summary_metrics["Simplex"]["iterations"].append(simplex_stats["iterations"])

        detailed_results.append({
            "LP_ID": lp_id,
            "Configuration": f"{n_vars} vars, {n_constraints} constraints",
            "Method": "Simplex (HiGHS)",
            "Solution": str(sol_s),
            "Objective": obj_s,
            "SolveTime_sec": time_s,
            "Feasible": sol_s is not None,
            "SolveTimeDifference": 0.0,
            "AccuracyPercent": 100.0,
            "FasterThanSimplex": False,
            "Iterations": simplex_stats["iterations"],
            "Status": simplex_stats["status"]
        })

        # --- Solve using Neural Network ---
        if verbose:
            print(f"  Evaluating neural network model specifically trained for {n_vars}x{n_constraints}...")

        sol_nn, obj_nn, time_nn, feasible_nn, nn_constraint_stats = evaluate_nn(nn_model, lp, verbose)

        # Compute comparison metrics
        nn_metrics = compute_comparison_metrics(time_nn, obj_nn, time_s, obj_s, nn_constraint_stats)

        summary_metrics["Neural Network"]["times"].append(time_nn)
        summary_metrics["Neural Network"]["objectives"].append(obj_nn)
        summary_metrics["Neural Network"]["accuracies"].append(nn_metrics["accuracy"])
        summary_metrics["Neural Network"]["faster"].append(nn_metrics["faster_than_simplex"])
        summary_metrics["Neural Network"]["constraint_violations"].append(nn_constraint_stats["max_violation"])

        detailed_results.append({
            "LP_ID": lp_id,
            "Configuration": f"{n_vars} vars, {n_constraints} constraints",
            "Method": "Neural Network",
            "Solution": str(sol_nn),
            "Objective": obj_nn,
            "SolveTime_sec": time_nn,
            "Feasible": feasible_nn,
            "SolveTimeDifference": nn_metrics["time_diff"],
            "AccuracyPercent": nn_metrics["accuracy"],
            "FasterThanSimplex": nn_metrics["faster_than_simplex"],
            "Speedup": nn_metrics["speedup"],
            "MaxConstraintViolation": nn_constraint_stats["max_violation"],
            "AvgConstraintViolation": nn_constraint_stats["avg_violation"],
            "NumViolatedConstraints": nn_constraint_stats["num_violated"]
        })

        # --- Solve using Genetic Algorithm ---
        sol_ga, obj_ga, time_ga, final_population, ga_stats = solve_ga(
            lp, initial_population=warm_start_population, verbose=verbose
        )

        # Update warm start population for next problem
        warm_start_population = final_population

        # Compute comparison metrics
        ga_metrics = compute_comparison_metrics(time_ga, obj_ga, time_s, obj_s, ga_stats["constraint_stats"])

        summary_metrics["Genetic Algorithm"]["times"].append(time_ga)
        summary_metrics["Genetic Algorithm"]["objectives"].append(obj_ga)
        summary_metrics["Genetic Algorithm"]["accuracies"].append(ga_metrics["accuracy"])
        summary_metrics["Genetic Algorithm"]["faster"].append(ga_metrics["faster_than_simplex"])
        summary_metrics["Genetic Algorithm"]["generations"].append(ga_stats["generations"])
        summary_metrics["Genetic Algorithm"]["constraint_violations"].append(
            ga_stats["constraint_stats"]["max_violation"])

        detailed_results.append({
            "LP_ID": lp_id,
            "Configuration": f"{n_vars} vars, {n_constraints} constraints",
            "Method": "Genetic Algorithm",
            "Solution": str(sol_ga),
            "Objective": obj_ga,
            "SolveTime_sec": time_ga,
            "Feasible": (obj_ga is not None),
            "SolveTimeDifference": ga_metrics["time_diff"],
            "AccuracyPercent": ga_metrics["accuracy"],
            "FasterThanSimplex": ga_metrics["faster_than_simplex"],
            "Speedup": ga_metrics["speedup"],
            "Generations": ga_stats["generations"],
            "MaxConstraintViolation": ga_stats["constraint_stats"]["max_violation"],
            "AvgConstraintViolation": ga_stats["constraint_stats"]["avg_violation"],
            "NumViolatedConstraints": ga_stats["constraint_stats"]["num_violated"],
            "EarlyStopped": ga_stats["early_stopped"]
        })

        # Generate GA convergence plot for the first test problem
        if i == 0:
            ga_plot = plot_ga_convergence(ga_stats, config_dir)

    # STEP 3: Compute summary statistics
    if verbose:
        print_section(f"Computing summary statistics for {n_vars}x{n_constraints}")

    # Compute statistical tests
    def safe_ttest(a, b):
        """Run t-test with error handling"""
        if len(a) < 2 or len(b) < 2:
            return None, None
        try:
            t_stat, p_value = stats.ttest_rel(a, b)
            return t_stat, p_value
        except Exception:
            return None, None

    # Time comparison statistical tests
    nn_time_ttest = safe_ttest(summary_metrics["Neural Network"]["times"],
                               summary_metrics["Simplex"]["times"])
    ga_time_ttest = safe_ttest(summary_metrics["Genetic Algorithm"]["times"],
                               summary_metrics["Simplex"]["times"])

    # Objective comparison statistical tests (filter None values)
    nn_objectives = [o for o, s in zip(summary_metrics["Neural Network"]["objectives"],
                                       summary_metrics["Simplex"]["objectives"])
                     if o is not None and s is not None]
    simplex_objectives_nn = [s for o, s in zip(summary_metrics["Neural Network"]["objectives"],
                                               summary_metrics["Simplex"]["objectives"])
                             if o is not None and s is not None]

    ga_objectives = [o for o, s in zip(summary_metrics["Genetic Algorithm"]["objectives"],
                                       summary_metrics["Simplex"]["objectives"])
                     if o is not None and s is not None]
    simplex_objectives_ga = [s for o, s in zip(summary_metrics["Genetic Algorithm"]["objectives"],
                                               summary_metrics["Simplex"]["objectives"])
                             if o is not None and s is not None]

    nn_obj_ttest = safe_ttest(nn_objectives, simplex_objectives_nn)
    ga_obj_ttest = safe_ttest(ga_objectives, simplex_objectives_ga)

    # Compute averages
    avg_summary = {}

    # Simplex averages
    simplex_times = summary_metrics["Simplex"]["times"]
    simplex_objectives = [obj for obj in summary_metrics["Simplex"]["objectives"] if obj is not None]
    simplex_iterations = [it for it in summary_metrics["Simplex"]["iterations"] if it is not None]

    avg_summary["Simplex"] = {
        "Avg_SolveTime": np.mean(simplex_times) if simplex_times else None,
        "StdDev_SolveTime": np.std(simplex_times) if len(simplex_times) > 1 else None,
        "Avg_Objective": np.mean(simplex_objectives) if simplex_objectives else None,
        "Avg_Iterations": np.mean(simplex_iterations) if simplex_iterations else None,
        "Success_Rate": sum(1 for obj in summary_metrics["Simplex"]["objectives"] if obj is not None) / num_test
    }

    # Neural Network averages
    nn_times = summary_metrics["Neural Network"]["times"]
    nn_objectives = [obj for obj in summary_metrics["Neural Network"]["objectives"] if obj is not None]
    nn_accuracies = [acc for acc in summary_metrics["Neural Network"]["accuracies"] if acc is not None]
    nn_violations = summary_metrics["Neural Network"]["constraint_violations"]

    avg_summary["Neural Network"] = {
        "Avg_SolveTime": np.mean(nn_times) if nn_times else None,
        "StdDev_SolveTime": np.std(nn_times) if len(nn_times) > 1 else None,
        "Avg_Objective": np.mean(nn_objectives) if nn_objectives else None,
        "Avg_SolveTimeDiff": np.mean([time_nn - time_s for time_nn, time_s in
                                      zip(nn_times, simplex_times)]) if nn_times and simplex_times else None,
        "Avg_AccuracyPercent": np.mean(nn_accuracies) if nn_accuracies else None,
        "FasterThanSimplex_Ratio": np.mean(
            [1 if faster else 0 for faster in summary_metrics["Neural Network"]["faster"]]),
        "Avg_MaxConstraintViolation": np.mean(nn_violations) if nn_violations else None,
        "Feasibility_Rate": sum(1 for v in nn_violations if v <= 1e-6) / len(nn_violations) if nn_violations else 0,
        "Time_TTest_Statistic": nn_time_ttest[0],
        "Time_TTest_PValue": nn_time_ttest[1],
        "Obj_TTest_Statistic": nn_obj_ttest[0],
        "Obj_TTest_PValue": nn_obj_ttest[1],
        "Training_Time": nn_metadata["training_time"]
    }

    # Genetic Algorithm averages
    ga_times = summary_metrics["Genetic Algorithm"]["times"]
    ga_objectives = [obj for obj in summary_metrics["Genetic Algorithm"]["objectives"] if obj is not None]
    ga_accuracies = [acc for acc in summary_metrics["Genetic Algorithm"]["accuracies"] if acc is not None]
    ga_generations = summary_metrics["Genetic Algorithm"]["generations"]
    ga_violations = summary_metrics["Genetic Algorithm"]["constraint_violations"]

    avg_summary["Genetic Algorithm"] = {
        "Avg_SolveTime": np.mean(ga_times) if ga_times else None,
        "StdDev_SolveTime": np.std(ga_times) if len(ga_times) > 1 else None,
        "Avg_Objective": np.mean(ga_objectives) if ga_objectives else None,
        "Avg_SolveTimeDiff": np.mean([time_ga - time_s for time_ga, time_s in
                                      zip(ga_times, simplex_times)]) if ga_times and simplex_times else None,
        "Avg_AccuracyPercent": np.mean(ga_accuracies) if ga_accuracies else None,
        "FasterThanSimplex_Ratio": np.mean(
            [1 if faster else 0 for faster in summary_metrics["Genetic Algorithm"]["faster"]]),
        "Avg_Generations": np.mean(ga_generations) if ga_generations else None,
        "Avg_MaxConstraintViolation": np.mean(ga_violations) if ga_violations else None,
        "Feasibility_Rate": sum(1 for v in ga_violations if v <= 1e-6) / len(ga_violations) if ga_violations else 0,
        "Time_TTest_Statistic": ga_time_ttest[0],
        "Time_TTest_PValue": ga_time_ttest[1],
        "Obj_TTest_Statistic": ga_obj_ttest[0],
        "Obj_TTest_PValue": ga_obj_ttest[1]
    }

    # STEP 4: Generate visualizations
    if verbose:
        print_section(f"Generating visualizations for {n_vars}x{n_constraints}")

    plots = generate_comparison_plots(summary_metrics, n_vars, n_constraints, config_dir)

    # Calculate total time for configuration
    config_total_time = time.time() - start_time
    if verbose:
        print(f"Configuration {n_vars}x{n_constraints} completed in {config_total_time:.2f} seconds")

    # Package the results for this configuration
    config_summary = {
        "Configuration": f"{n_vars} vars, {n_constraints} constraints",
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "Simplex": avg_summary["Simplex"],
        "Neural Network": avg_summary["Neural Network"],
        "Genetic Algorithm": avg_summary["Genetic Algorithm"],
        "Model_Summary": nn_metadata["model_summary"],
        "Total_Config_Time": config_total_time
    }

    return config_summary, detailed_results, problems_info, plots


def generate_comparison_plots(summary_metrics, n_vars, n_constraints, config_dir):
    """
    Generate comparison plots for methods

    Args:
        summary_metrics: Dictionary of metrics
        n_vars: Number of variables
        n_constraints: Number of constraints
        config_dir: Directory to save plots

    Returns:
        Dictionary of plot paths
    """
    plots = {}

    # 1. Solve time comparison
    plt.figure(figsize=(10, 6))
    methods = ["Simplex", "Neural Network", "Genetic Algorithm"]
    avg_times = [np.mean(summary_metrics[m]["times"]) for m in methods]
    colors = ["blue", "green", "orange"]

    bars = plt.bar(methods, avg_times, color=colors)
    plt.ylabel('Average Solve Time (seconds)')
    plt.title(f'Average Solve Time Comparison ({n_vars}x{n_constraints})')

    # Add exact values on top of bars
    for bar, time in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{time:.5f}', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    solve_time_path = os.path.join(config_dir, f"solve_time_comparison.png")
    plt.savefig(solve_time_path)
    plt.close()
    plots["solve_time"] = solve_time_path

    # 2. Accuracy comparison (excluding simplex which is baseline)
    plt.figure(figsize=(10, 6))
    methods = ["Neural Network", "Genetic Algorithm"]
    accuracies = [np.mean([acc for acc in summary_metrics[m]["accuracies"] if acc is not None])
                  for m in methods]
    colors = ["green", "orange"]

    bars = plt.bar(methods, accuracies, color=colors)
    plt.ylabel('Average Accuracy (%)')
    plt.title(f'Average Solution Accuracy ({n_vars}x{n_constraints})')
    plt.ylim(0, 105)  # Set y-axis limit to 0-105%

    # Add exact values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    accuracy_path = os.path.join(config_dir, f"accuracy_comparison.png")
    plt.savefig(accuracy_path)
    plt.close()
    plots["accuracy"] = accuracy_path

    # 3. Feasibility rate comparison
    plt.figure(figsize=(10, 6))
    methods = ["Simplex", "Neural Network", "Genetic Algorithm"]

    simplex_feasible = sum(1 for obj in summary_metrics["Simplex"]["objectives"] if obj is not None) / len(
        summary_metrics["Simplex"]["objectives"])
    nn_feasible = sum(1 for v in summary_metrics["Neural Network"]["constraint_violations"] if v <= 1e-6) / len(
        summary_metrics["Neural Network"]["constraint_violations"])
    ga_feasible = sum(1 for v in summary_metrics["Genetic Algorithm"]["constraint_violations"] if v <= 1e-6) / len(
        summary_metrics["Genetic Algorithm"]["constraint_violations"])

    feasibility = [simplex_feasible, nn_feasible, ga_feasible]
    colors = ["blue", "green", "orange"]

    bars = plt.bar(methods, feasibility, color=colors)
    plt.ylabel('Feasibility Rate (%)')
    plt.title(f'Solution Feasibility Rate ({n_vars}x{n_constraints})')
    plt.ylim(0, 1.05)  # Set y-axis limit to 0-105%

    # Add exact values on top of bars
    for bar, rate in zip(bars, feasibility):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate * 100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    feasibility_path = os.path.join(config_dir, f"feasibility_comparison.png")
    plt.savefig(feasibility_path)
    plt.close()
    plots["feasibility"] = feasibility_path

    # 4. Boxplot of solve times
    plt.figure(figsize=(10, 6))
    data = [summary_metrics["Simplex"]["times"],
            summary_metrics["Neural Network"]["times"],
            summary_metrics["Genetic Algorithm"]["times"]]

    plt.boxplot(data, labels=methods)
    plt.ylabel('Solve Time (seconds)')
    plt.title(f'Solve Time Distribution ({n_vars}x{n_constraints})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add note about training
    plt.figtext(0.5, 0.01,
                f"Neural network was trained and tested ONLY on {n_vars}x{n_constraints} problems",
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    solve_time_box_path = os.path.join(config_dir, f"solve_time_boxplot.png")
    plt.savefig(solve_time_box_path)
    plt.close()
    plots["solve_time_box"] = solve_time_box_path

    return plots


def generate_cross_config_plots(summaries, output_dir):
    """
    Generate plots comparing performance across configurations

    Args:
        summaries: List of configuration summaries
        output_dir: Directory to save plots

    Returns:
        Dictionary of plot paths
    """
    plots = {}

    # Create DataFrame from summaries
    rows = []
    for summary in summaries:
        config = summary["Configuration"]
        n_vars = summary["n_vars"]
        n_constraints = summary["n_constraints"]

        for method in ["Simplex", "Neural Network", "Genetic Algorithm"]:
            method_data = summary[method]
            row = {
                "Configuration": config,
                "n_vars": n_vars,
                "n_constraints": n_constraints,
                "Method": method,
                "Avg_SolveTime": method_data.get("Avg_SolveTime"),
                "Avg_Objective": method_data.get("Avg_Objective"),
                "Avg_AccuracyPercent": method_data.get("Avg_AccuracyPercent", 100 if method == "Simplex" else None),
                "Feasibility_Rate": method_data.get("Feasibility_Rate", method_data.get("Success_Rate", None))
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # 1. Plot solution time scaling
    plt.figure(figsize=(12, 8))

    for method, color in zip(["Simplex", "Neural Network", "Genetic Algorithm"], ["blue", "green", "orange"]):
        method_data = df[df["Method"] == method]
        # Sort by number of variables
        method_data = method_data.sort_values("n_vars")

        plt.plot(method_data["n_vars"], method_data["Avg_SolveTime"], 'o-',
                 label=method, color=color, linewidth=2, markersize=8)

    plt.xlabel("Number of Variables")
    plt.ylabel("Average Solve Time (seconds)")
    plt.title("Solution Time Scaling by Problem Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add methodology note
    plt.figtext(0.5, 0.01,
                "IMPORTANT: Each neural network was trained and tested ONLY on problems of the same size.\n"
                "There is NO cross-testing between different problem sizes.",
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    time_scaling_path = os.path.join(output_dir, "time_scaling.png")
    plt.savefig(time_scaling_path, bbox_inches='tight')
    plt.close()
    plots["time_scaling"] = time_scaling_path

    # 2. Plot accuracy scaling
    plt.figure(figsize=(12, 8))

    for method, color in zip(["Neural Network", "Genetic Algorithm"], ["green", "orange"]):
        method_data = df[df["Method"] == method]
        # Sort by number of variables
        method_data = method_data.sort_values("n_vars")

        plt.plot(method_data["n_vars"], method_data["Avg_AccuracyPercent"], 'o-',
                 label=method, color=color, linewidth=2, markersize=8)

    plt.xlabel("Number of Variables")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Solution Accuracy by Problem Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    plt.legend()

    # Add methodology note
    plt.figtext(0.5, 0.01,
                "IMPORTANT: Each neural network was trained and tested ONLY on problems of the same size.\n"
                "There is NO cross-testing between different problem sizes.",
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    accuracy_scaling_path = os.path.join(output_dir, "accuracy_scaling.png")
    plt.savefig(accuracy_scaling_path, bbox_inches='tight')
    plt.close()
    plots["accuracy_scaling"] = accuracy_scaling_path

    # 3. Plot feasibility scaling
    plt.figure(figsize=(12, 8))

    for method, color in zip(["Simplex", "Neural Network", "Genetic Algorithm"], ["blue", "green", "orange"]):
        method_data = df[df["Method"] == method]
        # Sort by number of variables
        method_data = method_data.sort_values("n_vars")

        plt.plot(method_data["n_vars"], method_data["Feasibility_Rate"], 'o-',
                 label=method, color=color, linewidth=2, markersize=8)

    plt.xlabel("Number of Variables")
    plt.ylabel("Feasibility Rate")
    plt.title("Solution Feasibility by Problem Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()

    # Add methodology note
    plt.figtext(0.5, 0.01,
                "IMPORTANT: Each neural network was trained and tested ONLY on problems of the same size.\n"
                "There is NO cross-testing between different problem sizes.",
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    feasibility_scaling_path = os.path.join(output_dir, "feasibility_scaling.png")
    plt.savefig(feasibility_scaling_path, bbox_inches='tight')
    plt.close()
    plots["feasibility_scaling"] = feasibility_scaling_path

    return plots, df


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_excel_report(summaries, detailed_results, problems_info, plots, run_dir):
    """
    Generate Excel report with results

    Args:
        summaries: List of configuration summaries
        detailed_results: List of detailed results
        problems_info: List of problem details
        plots: Dictionary of plot paths
        run_dir: Directory for the run

    Returns:
        Path to the generated Excel file
    """
    # Create DataFrames
    df_summary = pd.DataFrame()
    df_detailed = pd.DataFrame()
    df_problems = pd.DataFrame()
    df_models = pd.DataFrame()

    # Process summaries
    rows = []
    model_rows = []

    for summary in summaries:
        config = summary["Configuration"]
        n_vars = summary["n_vars"]
        n_constraints = summary["n_constraints"]

        # Add Simplex results
        rows.append({
            "Configuration": config,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "Method": "Simplex",
            "Avg_SolveTime": summary["Simplex"]["Avg_SolveTime"],
            "StdDev_SolveTime": summary["Simplex"]["StdDev_SolveTime"],
            "Avg_Objective": summary["Simplex"]["Avg_Objective"],
            "Avg_SolveTimeDiff": 0.0,
            "Avg_AccuracyPercent": 100.0,
            "FasterThanSimplex_Ratio": None,
            "Avg_Iterations": summary["Simplex"]["Avg_Iterations"],
            "Success_Rate": summary["Simplex"]["Success_Rate"]
        })

        # Add Neural Network results
        rows.append({
            "Configuration": config,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "Method": "Neural Network",
            "Avg_SolveTime": summary["Neural Network"]["Avg_SolveTime"],
            "StdDev_SolveTime": summary["Neural Network"]["StdDev_SolveTime"],
            "Avg_Objective": summary["Neural Network"]["Avg_Objective"],
            "Avg_SolveTimeDiff": summary["Neural Network"]["Avg_SolveTimeDiff"],
            "Avg_AccuracyPercent": summary["Neural Network"]["Avg_AccuracyPercent"],
            "FasterThanSimplex_Ratio": summary["Neural Network"]["FasterThanSimplex_Ratio"],
            "Avg_MaxConstraintViolation": summary["Neural Network"]["Avg_MaxConstraintViolation"],
            "Feasibility_Rate": summary["Neural Network"]["Feasibility_Rate"],
            "Time_TTest_Statistic": summary["Neural Network"]["Time_TTest_Statistic"],
            "Time_TTest_PValue": summary["Neural Network"]["Time_TTest_PValue"],
            "Obj_TTest_Statistic": summary["Neural Network"]["Obj_TTest_Statistic"],
            "Obj_TTest_PValue": summary["Neural Network"]["Obj_TTest_PValue"],
            "Training_Time": summary["Neural Network"]["Training_Time"]
        })

        # Add Genetic Algorithm results
        rows.append({
            "Configuration": config,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "Method": "Genetic Algorithm",
            "Avg_SolveTime": summary["Genetic Algorithm"]["Avg_SolveTime"],
            "StdDev_SolveTime": summary["Genetic Algorithm"]["StdDev_SolveTime"],
            "Avg_Objective": summary["Genetic Algorithm"]["Avg_Objective"],
            "Avg_SolveTimeDiff": summary["Genetic Algorithm"]["Avg_SolveTimeDiff"],
            "Avg_AccuracyPercent": summary["Genetic Algorithm"]["Avg_AccuracyPercent"],
            "FasterThanSimplex_Ratio": summary["Genetic Algorithm"]["FasterThanSimplex_Ratio"],
            "Avg_Generations": summary["Genetic Algorithm"]["Avg_Generations"],
            "Avg_MaxConstraintViolation": summary["Genetic Algorithm"]["Avg_MaxConstraintViolation"],
            "Feasibility_Rate": summary["Genetic Algorithm"]["Feasibility_Rate"],
            "Time_TTest_Statistic": summary["Genetic Algorithm"]["Time_TTest_Statistic"],
            "Time_TTest_PValue": summary["Genetic Algorithm"]["Time_TTest_PValue"],
            "Obj_TTest_Statistic": summary["Genetic Algorithm"]["Obj_TTest_Statistic"],
            "Obj_TTest_PValue": summary["Genetic Algorithm"]["Obj_TTest_PValue"]
        })

        # Add model summary
        model_rows.append({
            "Configuration": config,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "Model_Summary": summary["Model_Summary"],
            "Total_Config_Time": summary["Total_Config_Time"]
        })

    # Create DataFrames
    df_summary = pd.DataFrame(rows)
    df_detailed = pd.DataFrame(detailed_results)
    df_problems = pd.DataFrame(problems_info)
    df_models = pd.DataFrame(model_rows)

    # Create an explanation DataFrame
    explanation_df = pd.DataFrame({
        "Methodology": [
            "IMPORTANT - EXPERIMENTAL METHODOLOGY:",
            "",
            "Each configuration (e.g., 5x5, 10x10, etc.) represents a completely separate experiment with:",
            "1. A neural network trained SPECIFICALLY for that exact problem size",
            "2. Testing done ONLY on problems matching the trained configuration",
            "3. No transfer or sharing of models between different problem sizes",
            "",
            "For example, the neural network tested on 5x5 problems was ONLY trained on 5x5 problems.",
            "The neural network tested on 10x10 problems was ONLY trained on 10x10 problems.",
            "Each size has its own dedicated model - there is NO CROSS-TESTING between different sizes.",
            "",
            "This ensures fair comparison of methods within each specific problem configuration."
        ]
    })

    # Write to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(run_dir, f"lp_solver_results_{timestamp}.xlsx")

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # First write the methodology explanation
        explanation_df.to_excel(writer, sheet_name="Methodology", index=False)

        # Write the summary data
        df_summary.to_excel(writer, sheet_name="All_Summaries", index=False)

        # Write individual sheets for each configuration
        for config in df_summary["Configuration"].unique():
            config_data = df_summary[df_summary["Configuration"] == config].copy()
            # Create a safe sheet name (Excel has 31 char limit)
            sheet_name = f"Config_{config.split()[0]}"[:31]
            config_data.to_excel(writer, sheet_name=sheet_name, index=False)

            # Add note about independent training/testing
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            worksheet.write(0, config_data.shape[1] + 1,
                            "NOTE: The neural network for this configuration was")
            worksheet.write(1, config_data.shape[1] + 1,
                            "trained and tested ONLY on problems of this exact size.")
            worksheet.write(2, config_data.shape[1] + 1,
                            "Each configuration has its own separate neural network.")

        # Write other sheets
        df_detailed.to_excel(writer, sheet_name="Detailed_Results", index=False)
        df_problems.to_excel(writer, sheet_name="Problems", index=False)
        df_models.to_excel(writer, sheet_name="Model_Summaries", index=False)

        # Add annotation to detailed results
        detailed_sheet = writer.sheets["Detailed_Results"]
        detailed_sheet.write(0, df_detailed.shape[1] + 1, "IMPORTANT NOTE:")
        detailed_sheet.write(1, df_detailed.shape[1] + 1, "Each neural network was trained and tested")
        detailed_sheet.write(2, df_detailed.shape[1] + 1, "ONLY on problems matching its configuration.")

    return excel_path


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiments(configurations, num_train, num_test, use_parallel=False, verbose=True):
    """
    Run experiments for all configurations

    Args:
        configurations: List of configurations to test
        num_train: Number of training samples per configuration
        num_test: Number of test problems per configuration
        use_parallel: Whether to use parallel processing
        verbose: Whether to print detailed info

    Returns:
        Tuple of (summaries, detailed_results, problems_info, plots)
    """
    # Print experiment overview
    if verbose:
        print_header(f"RUNNING EXPERIMENTS WITH {len(configurations)} CONFIGURATIONS")
        print(f"Training samples per configuration: {num_train}")
        print(f"Test problems per configuration: {num_test}")

    # Initialize containers
    all_summaries = []
    all_detailed_results = []
    all_problems_info = []
    all_plots = {}

    # Decide between parallel and sequential execution
    if use_parallel and len(configurations) > 1:
        if verbose:
            print(f"Using parallel processing with {min(cpu_count(), len(configurations))} workers")

        # Define worker function
        def worker(config):
            n_vars = config["n_vars"]
            n_constraints = config["n_constraints"]
            worker_verbose = False  # Disable verbose output in workers
            return run_experiment_for_configuration(n_vars, n_constraints, num_train, num_test, worker_verbose)

        # Run experiments in parallel
        with Pool(processes=min(cpu_count(), len(configurations))) as pool:
            results = pool.map(worker, configurations)

        # Process results
        for config, result in zip(configurations, results):
            summary, detailed, problems, plots = result
            all_summaries.append(summary)
            all_detailed_results.extend(detailed)
            all_problems_info.extend(problems)
            all_plots.update(plots)

            if verbose:
                print(f"Processed results for configuration {config['n_vars']}x{config['n_constraints']}")
    else:
        # Sequential execution
        for config in configurations:
            n_vars = config["n_vars"]
            n_constraints = config["n_constraints"]

            summary, detailed, problems, plots = run_experiment_for_configuration(
                n_vars, n_constraints, num_train, num_test, verbose
            )

            all_summaries.append(summary)
            all_detailed_results.extend(detailed)
            all_problems_info.extend(problems)
            all_plots.update(plots)

    # Generate cross-configuration plots
    if len(configurations) > 1:
        if verbose:
            print_section("Generating cross-configuration comparison plots")

        cross_plots, summary_df = generate_cross_config_plots(all_summaries, RUN_DIR)
        all_plots.update(cross_plots)

    return all_summaries, all_detailed_results, all_problems_info, all_plots


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LP Solver Comparison Framework")

    # Configuration options
    parser.add_argument("--n_vars", type=int, help="Number of variables (for single configuration)")
    parser.add_argument("--n_constraints", type=int, help="Number of constraints (for single configuration)")
    parser.add_argument("--train_size", type=int, help="Number of training samples", default=DEFAULT_NUM_TRAIN)
    parser.add_argument("--test_size", type=int, help="Number of test problems", default=DEFAULT_NUM_TEST)

    # Execution options
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with minimal parameters")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")

    # Parse args
    args = parser.parse_args()

    # Set up the run directory and save configuration
    config = setup_run(args)

    # Set up logging
    create_log(os.path.join(RUN_DIR, "run.log"))

    # Print welcome message
    print_header("LP SOLVER COMPARISON FRAMEWORK", char="*")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {RUN_DIR}")

    # Set verbose flag based on quiet option
    verbose = not args.quiet

    # Determine configurations to run
    if args.n_vars is not None and args.n_constraints is not None:
        # Single configuration mode
        configurations = [{"n_vars": args.n_vars, "n_constraints": args.n_constraints}]
        print(f"Running single configuration: {args.n_vars}x{args.n_constraints}")
    elif args.quick:
        # Quick test mode
        configurations = [
            {"n_vars": 5, "n_constraints": 5},
            {"n_vars": 10, "n_constraints": 10}
        ]
        args.train_size = min(args.train_size, 1000)
        args.test_size = min(args.test_size, 3)
        print("Quick test mode enabled with reduced parameters")
    else:
        # Default configurations
        configurations = [
            {"n_vars": 5, "n_constraints": 5},
            {"n_vars": 10, "n_constraints": 10},
            {"n_vars": 15, "n_constraints": 15},
            {"n_vars": 20, "n_constraints": 20},
            {"n_vars": 25, "n_constraints": 25},
            {"n_vars": 50, "n_constraints": 50},
            {"n_vars": 100, "n_constraints": 100}
        ]

    # Print configuration information
    print(f"Running {len(configurations)} configurations with:")
    print(f"  - {args.train_size} training samples per configuration")
    print(f"  - {args.test_size} test problems per configuration")
    print(f"  - Neural network layers: {NN_LAYERS}")
    print(f"  - Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")

    # Run the experiments
    start_time = time.time()

    summaries, detailed_results, problems_info, plots = run_experiments(
        configurations, args.train_size, args.test_size, args.parallel, verbose
    )

    # Generate Excel report
    if verbose:
        print_section("Generating comprehensive Excel report")

    excel_path = generate_excel_report(summaries, detailed_results, problems_info, plots, RUN_DIR)

    # Print completion message
    total_time = time.time() - start_time
    print_header("EXPERIMENT COMPLETED", char="*")
    print(f"Total execution time: {total_time / 60:.2f} minutes")
    print(f"Results saved to: {excel_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")