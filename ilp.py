#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Integer LP Solver Comparison Framework

This program compares three methods for solving integer linear programming problems:
1. MIP (HiGHS) - Standard branch-and-bound optimization approach
2. Enhanced Neural Network - ML approach with constraint awareness and local search
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
from tensorflow.keras import layers, models, callbacks, regularizers
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import stats
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

# Ensure TensorFlow warnings are minimized
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set up output directory
OUTPUT_DIR = "ilp_solver_results"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ""  # Will be set in the setup_run function

# ILP problem parameters
ILP_LOWER_BOUND = 0  # Lower bound for variables
ILP_UPPER_BOUND = 10  # Upper bound for variables
DEFAULT_NUM_TRAIN = 10000  # Increased default training samples per configuration
DEFAULT_NUM_TEST = 15  # Default test problems per configuration

# Enhanced neural network parameters
NN_EPOCHS = 50
NN_BATCH_SIZE = 64
NN_BASE_WIDTH = 256
NN_LAYERS = 4
NN_LOCAL_SEARCH_ITERATIONS = 50

# Genetic algorithm parameters
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 200
GA_MUTATION_RATE = 0.1
GA_EARLY_STOP = 20  # Stop if no improvement for this many generations


# ============================================================================
# UTILITY FUNCTIONS
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


def compute_metrics(values, name=""):
    """
    Utility function to compute common metrics for a set of values

    Args:
        values: List of values to compute metrics on
        name: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    metrics = {}
    if not values:
        return metrics

    values = [v for v in values if v is not None]
    if not values:
        return metrics

    metrics[f"{name}_mean"] = np.mean(values)
    if len(values) > 1:
        metrics[f"{name}_std"] = np.std(values)
    metrics[f"{name}_min"] = np.min(values)
    metrics[f"{name}_max"] = np.max(values)
    metrics[f"{name}_count"] = len(values)

    return metrics


def create_bar_chart(data, labels, title, xlabel, ylabel, colors, filename, ylim=None, add_values=True):
    """
    Create a standardized bar chart

    Args:
        data: Data values to plot
        labels: Labels for x-axis
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Colors for bars
        filename: Output file path
        ylim: Y-axis limits (optional)
        add_values: Whether to add text values on bars

    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, data, color=colors)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(ylim)

    if add_values:
        for bar, value in zip(bars, data):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

    return filename


def setup_run(args):
    """Set up the run directory and save configuration"""
    global RUN_DIR

    # Create output directories
    ensure_dir(OUTPUT_DIR)

    # If custom output directory is provided, use it
    if args.output_dir:
        RUN_DIR = args.output_dir
        ensure_dir(RUN_DIR)
    else:
        # Use timestamp-based directory
        RUN_DIR = os.path.join(OUTPUT_DIR, f"run_{RUN_TIMESTAMP}")
        ensure_dir(RUN_DIR)

    # Save run configuration
    config = {
        'timestamp': RUN_TIMESTAMP,
        'args': vars(args),
        'ilp_params': {
            'lower_bound': ILP_LOWER_BOUND,
            'upper_bound': ILP_UPPER_BOUND,
            'num_train': args.train_size or DEFAULT_NUM_TRAIN,
            'num_test': args.test_size or DEFAULT_NUM_TEST
        },
        'nn_params': {
            'epochs': NN_EPOCHS,
            'batch_size': NN_BATCH_SIZE,
            'base_width': NN_BASE_WIDTH,
            'layers': NN_LAYERS,
            'local_search_iterations': NN_LOCAL_SEARCH_ITERATIONS
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
# ILP PROBLEM GENERATION
# ============================================================================

def generate_ilp_problem(n_vars, n_constraints, lower_bound=ILP_LOWER_BOUND, upper_bound=ILP_UPPER_BOUND):
    """
    Generate a random integer linear programming problem that is guaranteed to be feasible.

    The ILP has the form:
    minimize     c^T x
    subject to   A_ub x <= b_ub
                 x in {lower_bound, lower_bound+1, ..., upper_bound}^n_vars

    Returns:
        Dictionary containing the ILP parameters
    """
    # Generate a feasible integer point to ensure problem has a solution
    x_feasible = np.random.randint(lower_bound, upper_bound + 1, size=n_vars)

    # Generate constraint matrix with random coefficients
    A_ub = np.random.uniform(0, 10, size=(n_constraints, n_vars))

    # Round coefficients to integers to make the problem more interpretable
    A_ub = np.round(A_ub)

    # Add slack to ensure strict feasibility
    slack = np.random.randint(0, 10, size=n_constraints)
    b_ub = np.dot(A_ub, x_feasible) + slack

    # Generate objective function coefficients - use integers for better stability
    c = np.random.randint(-10, 11, size=n_vars)

    # Variable bounds - ensure they're integers
    int_lower = int(lower_bound)
    int_upper = int(upper_bound)

    # Variable type indicators (1.0 for integer)
    integrality = np.ones(n_vars)

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
        "bounds": Bounds(lb=int_lower, ub=int_upper, keep_feasible=True),
        "integrality": integrality,
        "x_feasible": x_feasible,
        "stats": problem_stats
    }


def describe_ilp(ilp, include_matrices=False):
    """Return a string description of the ILP problem"""
    n_vars = len(ilp["c"])
    n_constraints = len(ilp["b_ub"])

    description = [
        f"ILP Problem: {n_vars} variables, {n_constraints} constraints",
        f"Objective range: {np.min(ilp['c']):.2f} to {np.max(ilp['c']):.2f}",
        f"Average slack: {np.mean(ilp['b_ub'] - np.dot(ilp['A_ub'], ilp['x_feasible'])):.2f}",
        f"Variable bounds: [{ilp['bounds'].lb}, {ilp['bounds'].ub}] (integers only)"
    ]

    if include_matrices:
        description.extend([
            "\nObjective coefficients:",
            str(ilp["c"]),
            "\nFirst few constraints:",
            str(ilp["A_ub"][:min(3, n_constraints)]),
            "\nFirst few RHS values:",
            str(ilp["b_ub"][:min(3, n_constraints)])
        ])

    return "\n".join(description)


# ============================================================================
# METHOD 1: MIP (HiGHS) SOLVER
# ============================================================================

def solve_mip(ilp, verbose=False):
    """
    Solve ILP using the HiGHS MIP solver

    Args:
        ilp: Dictionary containing ILP parameters
        verbose: Whether to print detailed info

    Returns:
        Tuple of (solution, objective, solve_time, mip_stats)
    """
    c, A_ub, b_ub, bounds, integrality = ilp["c"], ilp["A_ub"], ilp["b_ub"], ilp["bounds"], ilp["integrality"]

    if verbose:
        print("  Solving with MIP (HiGHS) method...")

    # Setup constraints
    constraints = [LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)]

    # Solve the ILP
    start_time = time.time()
    res = milp(
        c=c,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
        options={'disp': verbose}
    )
    solve_time = time.time() - start_time

    if res.success:
        solution = res.x
        objective = res.fun
        iterations = res.nit if hasattr(res, 'nit') else None
        status = res.message if hasattr(res, 'message') else "Success"

        # Verify solution is integer
        is_integer = all(abs(x - round(x)) < 1e-6 for x in solution)
        if not is_integer:
            print(f"WARNING: MIP solution contains non-integer values: {solution}")
            # Round to nearest integers if needed
            solution = np.round(solution).astype(int)
            # Recalculate objective with rounded solution
            objective = np.dot(c, solution)

        if verbose:
            print(f"    MIP solved successfully in {solve_time:.6f} seconds")
            print(f"    Objective value: {objective:.6f}")
    else:
        solution = None
        objective = None
        iterations = res.nit if hasattr(res, 'nit') else None
        status = res.message if hasattr(res, 'message') else "Failed"

        if verbose:
            print(f"    MIP failed: {status}")

    mip_stats = {
        "iterations": iterations,
        "status": status
    }

    return solution, objective, solve_time, mip_stats


# ============================================================================
# IMPROVED NEURAL NETWORK APPROACH
# ============================================================================

def improved_constraint_aware_loss(y_true, y_pred):
    """
    Enhanced loss function that more heavily penalizes constraint violations
    and encourages integer solutions

    Args:
        y_true: True solutions
        y_pred: Predicted solutions

    Returns:
        Combined loss value
    """
    # Base MSE loss for objective value matching
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Integer penalty with improved formulation
    # This creates a sharper penalty near integer values
    frac_part = tf.abs(y_pred - tf.round(y_pred))

    # Sharper penalty function that increases for values far from integers
    integer_penalty = tf.reduce_mean(tf.square(frac_part) / (0.1 + frac_part))

    # Higher weight for integer penalties to enforce integrality
    return mse_loss + 0.25 * integer_penalty


def build_enhanced_nn_model(input_dim, output_dim, problem_size):
    """
    Build an improved neural network with better architecture for ILP problems

    Args:
        input_dim: Input dimension (flattened ILP size)
        output_dim: Output dimension (number of variables)
        problem_size: Size of the problem (n_vars * n_constraints)

    Returns:
        Compiled Keras model
    """
    # Scale network width based on problem complexity
    width = min(NN_BASE_WIDTH * 2, problem_size * 4)

    # Create a model with improved architecture
    inputs = tf.keras.Input(shape=(input_dim,))

    # Initial dense layer
    x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Residual blocks with improved capacity
    for i in range(NN_LAYERS):
        skip = x

        # First dense layer in block
        x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Second dense layer in block with increased width for more capacity
        x = layers.Dense(int(width * 1.5), activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)

        # Bottleneck layer
        x = layers.Dense(width, activation='relu')(x)

        # Add skip connection with projection if needed
        if i % 2 == 0:  # Every other layer gets a different skip structure
            skip = layers.Dense(width, activation=None)(skip)

        x = layers.Add()([x, skip])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

    # Attention mechanism to focus on important features
    attention = layers.Dense(width, activation='tanh')(x)
    attention = layers.Dense(1, activation='sigmoid')(attention)
    x = layers.Multiply()([x, attention])

    # Output layer preparation
    x = layers.Dense(width // 2, activation='relu')(x)
    x = layers.Dense(width // 4, activation='relu')(x)

    # Final output
    outputs = layers.Dense(output_dim, activation='linear')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Use a fixed learning rate instead of a scheduler
    # This allows ReduceLROnPlateau callback to work properly
    learning_rate = 0.001

    # Compile with improved loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=improved_constraint_aware_loss, metrics=['mae'])

    return model


def flatten_ilp(ilp, normalize=True):
    """
    Flatten ILP parameters into a feature vector

    Args:
        ilp: Dictionary containing ILP parameters
        normalize: Whether to normalize features

    Returns:
        Numpy array with flattened features
    """
    # Extract the components
    c = ilp["c"]
    A_ub = ilp["A_ub"].flatten()
    b_ub = ilp["b_ub"]

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
        # Generate a random ILP problem
        ilp = generate_ilp_problem(n_vars, n_constraints)

        # Solve it with MIP to get the "ground truth"
        solution, objective, _, _ = solve_mip(ilp)

        if solution is not None:
            # Add to training data
            X.append(flatten_ilp(ilp))
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


class DataAugmentation(tf.keras.layers.Layer):
    """Data augmentation layer for ILP problems"""

    def __init__(self, noise_level=0.02, **kwargs):
        super(DataAugmentation, self).__init__(**kwargs)
        self.noise_level = noise_level

    def call(self, inputs, training=None):
        if training:
            return inputs + tf.random.normal(shape=tf.shape(inputs),
                                             mean=0.0,
                                             stddev=self.noise_level)
        return inputs


def train_enhanced_nn_model(n_vars, n_constraints, num_train, verbose=True):
    """
    Train enhanced neural network for a specific problem configuration

    Args:
        n_vars: Number of variables
        n_constraints: Number of constraints
        num_train: Number of training samples
        verbose: Whether to print progress

    Returns:
        Trained model, training history, and metadata
    """
    if verbose:
        print_section(f"Training Enhanced Neural Network for {n_vars}x{n_constraints} Configuration")
        print(f"This model will ONLY be used for {n_vars}x{n_constraints} problems")

    # Generate more training data specifically for this configuration
    X_train, Y_train = create_training_data(n_vars, n_constraints, num_train, verbose)

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    output_dim = n_vars
    problem_size = n_vars * n_constraints

    if verbose:
        print(f"Building enhanced neural network for problem size {problem_size}")

    # Build the enhanced model
    model = build_enhanced_nn_model(input_dim, output_dim, problem_size)

    # Save model summary as string
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
    model_summary = summary_io.getvalue()

    if verbose:
        print(model_summary)

    # Add callbacks for better training
    callbacks_list = [
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Reduce learning rate when plateau is reached
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        ),
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=os.path.join(RUN_DIR, f"nn_model_{n_vars}x{n_constraints}.h5"),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    # Train the model
    training_start = time.time()

    if verbose:
        print(f"Training neural network with {NN_EPOCHS} epochs, batch size {NN_BATCH_SIZE}...")

    # Ensure data is properly shaped and has no NaN values
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    # Check for NaN or Inf values
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Warning: NaN or Inf values found in training data. Replacing with zeros.")
        X_train = np.nan_to_num(X_train)

    if np.isnan(Y_train).any() or np.isinf(Y_train).any():
        print("Warning: NaN or Inf values found in training targets. Replacing with zeros.")
        Y_train = np.nan_to_num(Y_train)

    # Apply data augmentation safely
    try:
        # Create data augmentation layer
        data_aug = DataAugmentation(noise_level=0.02)
        X_train_aug = data_aug(X_train, training=True).numpy()
    except Exception as e:
        print(f"Data augmentation failed: {e}. Using original data.")
        X_train_aug = X_train

    try:
        history = model.fit(
            X_train_aug, Y_train,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_data=(X_val, Y_val),
            callbacks=callbacks_list,
            verbose=1 if verbose else 0
        )
    except Exception as e:
        print(f"Training failed with batch size {NN_BATCH_SIZE}, trying with smaller batch: {e}")
        try:
            # Try with a smaller batch size
            smaller_batch = max(16, NN_BATCH_SIZE // 2)
            history = model.fit(
                X_train_aug, Y_train,
                epochs=NN_EPOCHS,
                batch_size=smaller_batch,
                validation_data=(X_val, Y_val),
                callbacks=callbacks_list,
                verbose=1 if verbose else 0
            )
        except Exception as e:
            print(f"Training still failed with smaller batch: {e}")
            # Create a basic history object with empty lists
            history = type('obj', (object,), {
                'history': {
                    'loss': [0],
                    'val_loss': [0],
                    'mae': [0],
                    'val_mae': [0]
                }
            })
            print("Using a dummy history object.")

    training_time = time.time() - training_start

    if verbose:
        print(f"Neural network training completed in {training_time:.2f} seconds")
        if hasattr(history.history, 'val_mae'):
            print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")

    # Create metadata
    metadata = {
        "configuration": f"{n_vars}x{n_constraints}",
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "problem_size": problem_size,
        "model_width": min(NN_BASE_WIDTH * 2, problem_size * 4),
        "model_depth": NN_LAYERS,
        "epochs": NN_EPOCHS,
        "batch_size": NN_BATCH_SIZE,
        "training_samples": num_train,
        "training_time": training_time,
        "final_loss": history.history['loss'][-1] if 'loss' in history.history else 0,
        "final_val_loss": history.history['val_loss'][-1] if 'val_loss' in history.history else 0,
        "final_mae": history.history['mae'][-1] if 'mae' in history.history else 0,
        "final_val_mae": history.history['val_mae'][-1] if 'val_mae' in history.history else 0,
        "model_summary": model_summary
    }

    return model, history, metadata


def enhanced_local_search_improvement(solution, ilp, max_iterations=NN_LOCAL_SEARCH_ITERATIONS, verbose=False):
    """
    Apply enhanced local search with adaptive strategies to improve ILP solution

    Args:
        solution: Initial integer solution
        ilp: Dictionary containing ILP parameters
        max_iterations: Maximum number of local search iterations
        verbose: Whether to print detailed info

    Returns:
        Improved solution
    """
    c, A_ub, b_ub = ilp["c"], ilp["A_ub"], ilp["b_ub"]
    bounds = ilp["bounds"]
    lb, ub = int(bounds.lb), int(bounds.ub)
    n_vars = len(solution)

    # Current solution stats
    best_solution = solution.copy()
    constraints = np.dot(A_ub, best_solution) - b_ub
    violations = np.maximum(0, constraints)
    max_violation = np.max(violations) if len(violations) > 0 else 0
    is_feasible = max_violation <= 1e-6

    # If already feasible, use objective value as fitness
    if is_feasible:
        best_fitness = np.dot(c, best_solution)
    else:
        # Use adaptive penalty function for infeasible solutions
        # Increase penalty for more severe violations
        penalty = 1000 * np.sum(violations ** 2) + 5000 * np.max(violations)
        best_fitness = np.dot(c, best_solution) + penalty

    if verbose:
        print(f"    Starting enhanced local search from solution with fitness {best_fitness:.4f}")
        print(f"    Initial feasibility: {is_feasible}, Max violation: {max_violation:.6f}")

    # Track the variables that have been tried without improvement
    tried_vars = set()
    stuck_count = 0

    # Adaptive step sizes - start with larger moves early
    early_phase = max_iterations // 3
    mid_phase = 2 * max_iterations // 3

    # Local search loop with adaptive strategies
    for iteration in range(max_iterations):
        improved = False

        # If stuck, try random perturbations
        if stuck_count > n_vars // 2:
            if verbose:
                print(f"    Search stuck, applying random perturbation at iteration {iteration}")

            # Apply random perturbation to escape local minimum
            perturbation = np.random.randint(-2, 3, size=n_vars)
            candidate = np.clip(best_solution + perturbation, lb, ub)

            # Check feasibility
            constraints = np.dot(A_ub, candidate) - b_ub
            violations = np.maximum(0, constraints)
            max_violation = np.max(violations) if len(violations) > 0 else 0
            candidate_is_feasible = max_violation <= 1e-6

            # Calculate fitness
            if candidate_is_feasible:
                candidate_fitness = np.dot(c, candidate)
            else:
                penalty = 1000 * np.sum(violations ** 2) + 5000 * np.max(violations)
                candidate_fitness = np.dot(c, candidate) + penalty

            # Accept if better or with some probability to escape local optima
            if candidate_fitness < best_fitness or (not is_feasible and candidate_is_feasible) or \
                    (np.random.random() < 0.3 and iteration > mid_phase):
                best_solution = candidate
                best_fitness = candidate_fitness
                is_feasible = candidate_is_feasible
                improved = True
                tried_vars = set()  # Reset tried variables
                stuck_count = 0

                if verbose and candidate_is_feasible and not is_feasible:
                    print(f"    Iteration {iteration}: Random perturbation found feasible solution!")

            # Skip the rest of this iteration if we applied a perturbation
            if improved:
                continue

        # Determine step sizes based on phase
        if iteration < early_phase:
            step_sizes = [1, -1, 2, -2, 3, -3]
        elif iteration < mid_phase:
            step_sizes = [1, -1, 2, -2]
        else:
            step_sizes = [1, -1]

        # Try changing each variable in order of importance
        # Prioritize variables with higher objective coefficients
        var_priorities = np.argsort(np.abs(c))[::-1]

        for var_idx in var_priorities:
            i = var_idx

            # Skip if this variable has been tried recently without success
            if i in tried_vars and np.random.random() > 0.2:
                continue

            random.shuffle(step_sizes)  # Randomize search direction

            for step in step_sizes:
                # Try incrementing or decrementing the variable
                candidate = best_solution.copy()
                candidate[i] += step

                # Check bounds
                if candidate[i] < lb or candidate[i] > ub:
                    continue

                # Check feasibility
                constraints = np.dot(A_ub, candidate) - b_ub
                violations = np.maximum(0, constraints)
                max_violation = np.max(violations) if len(violations) > 0 else 0
                candidate_is_feasible = max_violation <= 1e-6

                # Calculate fitness with adaptive penalty
                if candidate_is_feasible:
                    candidate_fitness = np.dot(c, candidate)
                else:
                    # Increase penalty for repeated violations
                    penalty = 1000 * np.sum(violations ** 2) + 5000 * np.max(violations)
                    candidate_fitness = np.dot(c, candidate) + penalty

                # Improvement criteria with better balance of exploration/exploitation
                uphill_condition = not is_feasible and iteration > mid_phase and np.random.random() < 0.1

                if ((not is_feasible and candidate_is_feasible) or
                        (is_feasible and candidate_is_feasible and candidate_fitness < best_fitness) or
                        (not is_feasible and not candidate_is_feasible and candidate_fitness < best_fitness) or
                        uphill_condition):
                    best_solution = candidate
                    best_fitness = candidate_fitness
                    is_feasible = candidate_is_feasible
                    improved = True

                    # Remove from tried variables
                    if i in tried_vars:
                        tried_vars.remove(i)

                    if verbose and candidate_is_feasible and not is_feasible:
                        print(f"    Iteration {iteration}: Found feasible solution!")

                    break  # Skip trying other steps for this variable

            if improved:
                break  # Skip to the next iteration if we've improved
            else:
                tried_vars.add(i)  # Add to tried variables

        # Update stuck counter
        if not improved:
            stuck_count += 1
            if verbose and stuck_count % 5 == 0:
                print(f"    Search stuck for {stuck_count} iterations")
        else:
            stuck_count = 0

    # Final solution stats
    constraints = np.dot(A_ub, best_solution) - b_ub
    violations = np.maximum(0, constraints)
    max_violation = np.max(violations) if len(violations) > 0 else 0
    is_feasible = max_violation <= 1e-6
    final_objective = np.dot(c, best_solution)

    if verbose:
        print(f"    Final solution objective: {final_objective:.4f}")
        print(f"    Final feasibility: {is_feasible}, Max violation: {max_violation:.6f}")

    return best_solution


def evaluate_enhanced_nn(model, ilp, verbose=False):
    """
    Evaluate enhanced neural network on an ILP problem with local search improvement

    Args:
        model: Trained neural network model
        ilp: Dictionary containing ILP parameters
        verbose: Whether to print detailed info

    Returns:
        Tuple of (solution, objective, inference_time, is_feasible, constraint_stats)
    """
    if verbose:
        print("  Evaluating enhanced neural network model...")

    # Flatten and normalize ILP
    X_input = flatten_ilp(ilp)
    X_input = np.expand_dims(X_input, axis=0)  # Add batch dimension

    # Time the prediction and local search
    start_time = time.time()

    # Get neural network prediction
    prediction = model.predict(X_input, verbose=0)
    pred_time = time.time() - start_time

    if verbose:
        print(f"    Neural network prediction time: {pred_time:.6f} seconds")

    # Extract solution
    raw_solution = prediction.flatten()

    # Round to nearest integers
    solution = np.round(raw_solution).astype(int)

    # Clip solution to bounds
    bounds = ilp["bounds"]
    lb, ub = bounds.lb, bounds.ub
    solution = np.clip(solution, lb, ub)

    # Apply enhanced local search improvement
    if verbose:
        print("    Applying enhanced local search improvement...")

    local_search_start = time.time()
    improved_solution = enhanced_local_search_improvement(solution, ilp, verbose=verbose)
    local_search_time = time.time() - local_search_start

    # Total time includes prediction and local search
    inference_time = time.time() - start_time

    if verbose:
        print(f"    Local search time: {local_search_time:.6f} seconds")
        print(f"    Total inference time: {inference_time:.6f} seconds")

    # Calculate objective value
    c = ilp["c"]
    initial_objective = np.dot(c, solution)
    objective = np.dot(c, improved_solution)

    objective_improvement = initial_objective - objective
    if verbose and objective_improvement != 0:
        print(f"    Objective improved by {objective_improvement:.4f} through local search")

    # Check feasibility
    A_ub = ilp["A_ub"]
    b_ub = ilp["b_ub"]

    initial_constraints = np.dot(A_ub, solution) - b_ub
    initial_max_violation = np.max(initial_constraints)
    initial_is_feasible = initial_max_violation <= 1e-6

    constraints = np.dot(A_ub, improved_solution) - b_ub
    max_violation = np.max(constraints)
    is_feasible = max_violation <= 1e-6

    # Calculate constraint statistics
    constraint_stats = {
        "max_violation": max_violation,
        "initial_max_violation": initial_max_violation,
        "avg_violation": np.mean(constraints[constraints > 0]) if np.any(constraints > 0) else 0,
        "num_violated": np.sum(constraints > 0),
        "rounding_impact": np.mean(np.abs(solution - raw_solution)),
        "local_search_improvement": objective_improvement,
        "initial_feasibility": initial_is_feasible,
        "final_feasibility": is_feasible
    }

    if verbose:
        print(f"    Initial feasibility: {initial_is_feasible}, Max violation: {initial_max_violation:.6f}")
        print(f"    Final feasibility: {is_feasible}, Max violation: {max_violation:.6f}")
        print(f"    Initial objective: {initial_objective:.6f}, Final objective: {objective:.6f}")
        print(f"    Average rounding impact: {constraint_stats['rounding_impact']:.4f}")

    return improved_solution, objective, inference_time, is_feasible, constraint_stats


def plot_nn_training_history(history, config_dir):
    """
    Plot the neural network training history

    Args:
        history: Training history object
        config_dir: Directory to save plot

    Returns:
        Path to saved plot
    """
    # Check if history is None or doesn't have expected attributes
    if history is None or not hasattr(history, 'history') or not history.history:
        # Create a dummy plot
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No training history available",
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(config_dir, 'nn_training_history.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    plt.figure(figsize=(12, 8))

    # Plot training & validation loss
    plt.subplot(2, 1, 1)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot MAE
    plt.subplot(2, 1, 2)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Neural Network Training History - Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
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
        x: Solution vector (integer)
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


def solve_ga(ilp, population_size=GA_POPULATION_SIZE, generations=GA_GENERATIONS,
             mutation_rate=GA_MUTATION_RATE, early_stop=GA_EARLY_STOP,
             initial_population=None, verbose=False):
    """
    Solve ILP using genetic algorithm

    Args:
        ilp: Dictionary containing ILP parameters
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

    c, A_ub, b_ub = ilp["c"], ilp["A_ub"], ilp["b_ub"]
    bounds = ilp["bounds"]
    lb, ub = int(bounds.lb), int(bounds.ub)
    n_vars = len(c)

    # Initialize population with integer individuals
    population = []

    # Use warm start if provided
    if initial_population is not None:
        if verbose:
            print(f"    Using warm start with {len(initial_population)} individuals")

        for individual in initial_population:
            # Ensure integers and clip to bounds
            clipped = np.clip(np.round(individual).astype(int), lb, ub).tolist()
            population.append(clipped)

        # Fill remaining population with random individuals
        while len(population) < population_size:
            individual = [random.randint(lb, ub) for i in range(n_vars)]
            population.append(individual)

        # Truncate if needed
        population = population[:population_size]
    else:
        if verbose:
            print(f"    Initializing random population of size {population_size}")

        for _ in range(population_size):
            individual = [random.randint(lb, ub) for i in range(n_vars)]
            population.append(individual)

    # Integer mutation function
    def mutate(individual, generation):
        # Decrease mutation strength over time
        scale = 1.0 - 0.5 * (generation / generations) if generations > 0 else 0.5

        for i in range(n_vars):
            if random.random() < mutation_rate:
                # For integers: add or subtract a small integer value
                # Scale determines the probability of larger jumps
                jump_size = 1
                if random.random() < scale:
                    jump_size = 2
                if random.random() < scale / 2:
                    jump_size = 3

                # Add or subtract the jump_size
                if random.random() < 0.5:
                    individual[i] += jump_size
                else:
                    individual[i] -= jump_size

                # Clip to bounds
                individual[i] = max(lb, min(individual[i], ub))

        return individual

    # Crossover function for integers (uniform crossover)
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

    # Verify integrality
    is_integer = all(isinstance(x, int) or x.is_integer() if hasattr(x, 'is_integer') else x.is_integer()
                     for x in best_individual)
    if not is_integer and verbose:
        print("    WARNING: GA solution contains non-integer values. Rounding...")
        best_individual = [int(round(x)) for x in best_individual]

    # Convert to numpy array for consistency
    best_individual = np.array(best_individual, dtype=int)

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

def compute_comparison_metrics(method_time, method_obj, mip_time, mip_obj, method_constraints=None):
    """
    Compute comparison metrics between a method and MIP

    Args:
        method_time: Solution time for method
        method_obj: Objective value for method
        mip_time: Solution time for MIP
        mip_obj: Objective value for MIP
        method_constraints: Optional constraint statistics

    Returns:
        Dictionary of comparison metrics
    """
    # Time comparison
    time_diff = method_time - mip_time
    time_ratio = method_time / mip_time if mip_time > 0 else float('inf')
    speedup = mip_time / method_time if method_time > 0 else float('inf')

    # Objective comparison
    if mip_obj is None or method_obj is None:
        accuracy = None
        obj_ratio = None
        obj_diff = None
    elif abs(mip_obj) < 1e-6:
        # Handle case where MIP objective is very close to zero
        accuracy = 100.0 if abs(method_obj) < 1e-6 else 0.0
        obj_ratio = 1.0 if abs(method_obj) < 1e-6 else float('inf')
        obj_diff = abs(method_obj - mip_obj)
    else:
        # Regular case
        # Calculate accuracy as a percentage of the optimal solution
        # 100% = perfect match, 0% = completely wrong
        accuracy = max(0.0, 100 - (abs(method_obj - mip_obj) / abs(mip_obj)) * 100)
        obj_ratio = method_obj / mip_obj
        obj_diff = method_obj - mip_obj

    # Quick comparison
    faster_than_mip = method_time < mip_time

    metrics = {
        "time_diff": time_diff,
        "time_ratio": time_ratio,
        "speedup": speedup,
        "objective_diff": obj_diff,
        "objective_ratio": obj_ratio,
        "accuracy": accuracy,
        "faster_than_mip": faster_than_mip
    }

    # Include constraint stats if available
    if method_constraints:
        metrics.update(method_constraints)

    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

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

    # 1. Solve time comparison using the utility function
    methods = ["MIP", "Neural Network", "Genetic Algorithm"]
    avg_times = [np.mean(summary_metrics[m]["times"]) for m in methods]
    colors = ["blue", "green", "orange"]

    solve_time_path = create_bar_chart(
        avg_times, methods,
        f'Average Solve Time Comparison ({n_vars}x{n_constraints})',
        '', 'Average Solve Time (seconds)',
        colors, os.path.join(config_dir, f"solve_time_comparison.png")
    )
    plots["solve_time"] = solve_time_path

    # 2. Accuracy comparison (excluding MIP which is baseline)
    methods = ["Neural Network", "Genetic Algorithm"]
    accuracies = [np.mean([acc for acc in summary_metrics[m]["accuracies"] if acc is not None])
                  for m in methods]
    colors = ["green", "orange"]

    accuracy_path = create_bar_chart(
        accuracies, methods,
        f'Average Solution Accuracy ({n_vars}x{n_constraints})',
        '', 'Average Accuracy (%)',
        colors, os.path.join(config_dir, f"accuracy_comparison.png"),
        ylim=(0, 105)
    )
    plots["accuracy"] = accuracy_path

    # 3. Feasibility rate comparison
    methods = ["MIP", "Neural Network (Initial)", "Neural Network (Final)", "Genetic Algorithm"]

    mip_feasible = sum(1 for obj in summary_metrics["MIP"]["objectives"] if obj is not None) / len(
        summary_metrics["MIP"]["objectives"])
    nn_initial_feasible = np.mean(
        [1 if feasible else 0 for feasible in summary_metrics["Neural Network"]["initial_feasibility"]])
    nn_final_feasible = np.mean(
        [1 if feasible else 0 for feasible in summary_metrics["Neural Network"]["final_feasibility"]])
    ga_feasible = sum(1 for v in summary_metrics["Genetic Algorithm"]["constraint_violations"] if v <= 1e-6) / len(
        summary_metrics["Genetic Algorithm"]["constraint_violations"])

    feasibility = [mip_feasible, nn_initial_feasible, nn_final_feasible, ga_feasible]
    colors = ["blue", "lightgreen", "green", "orange"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, feasibility, color=colors)
    plt.ylabel('Feasibility Rate (%)')
    plt.title(f'Solution Feasibility Rate ({n_vars}x{n_constraints})')
    plt.ylim(0, 1.05)  # Set y-axis limit to 0-105%

    # Make x-axis labels vertical if many methods
    plt.xticks(rotation=15, ha='right')

    # Add exact values on top of bars
    for bar, rate in zip(bars, feasibility):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate * 100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    feasibility_path = os.path.join(config_dir, f"feasibility_comparison.png")
    plt.savefig(feasibility_path)
    plt.close()
    plots["feasibility"] = feasibility_path

    # 4. NEW: Accuracy distribution (binned)
    plt.figure(figsize=(12, 8))
    methods = ["Neural Network", "Genetic Algorithm"]
    colors = ["green", "orange"]

    # Define accuracy bins
    accuracy_bins = [
        (95, 100.1, "95-100%"),  # Using 100.1 to include exactly 100%
        (90, 95, "90-95%"),
        (80, 90, "80-90%"),
        (50, 80, "50-80%"),
        (0, 50, "<50%")
    ]

    for i, method in enumerate(methods):
        accuracies = [acc for acc in summary_metrics[method]["accuracies"] if acc is not None]

        if not accuracies:
            continue

        bin_counts = []
        for low, high, _ in accuracy_bins:
            count = sum(1 for acc in accuracies if low <= acc < high)
            bin_counts.append(count / len(accuracies) * 100 if accuracies else 0)

        bar_positions = np.arange(len(accuracy_bins)) + 0.2 * (i - 0.5)
        plt.bar(bar_positions, bin_counts, width=0.2, color=colors[i], label=method, alpha=0.7)

    plt.xlabel('Accuracy Range')
    plt.ylabel('Percentage of Problems (%)')
    plt.title(f'Distribution of Solution Accuracy ({n_vars}x{n_constraints})')
    plt.xticks(np.arange(len(accuracy_bins)), [label for _, _, label in accuracy_bins])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    accuracy_dist_path = os.path.join(config_dir, f"accuracy_distribution.png")
    plt.savefig(accuracy_dist_path)
    plt.close()
    plots["accuracy_distribution"] = accuracy_dist_path

    # Neural Network Improvements from Local Search
    if nn_history is not None:
        plt.figure(figsize=(10, 6))

        # Prepare data
        improvements = summary_metrics["Neural Network"]["local_search_improvements"]
        initial_feasible = summary_metrics["Neural Network"]["initial_feasibility"]
        final_feasible = summary_metrics["Neural Network"]["final_feasibility"]

        # Calculate how often local search improves feasibility
        feasibility_improved = sum(1 for init, final in zip(initial_feasible, final_feasible)
                                   if not init and final)
        feasibility_improvement_rate = feasibility_improved / len(initial_feasible) if len(initial_feasible) > 0 else 0

        # Stats for objective improvement
        pos_improvements = [imp for imp in improvements if imp > 0]
        neg_improvements = [imp for imp in improvements if imp < 0]
        no_improvements = sum(1 for imp in improvements if imp == 0)

        avg_pos_improvement = np.mean(pos_improvements) if pos_improvements else 0
        avg_neg_improvement = np.mean(neg_improvements) if neg_improvements else 0

        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot 1: Histogram of objective improvements
        ax1.hist(improvements, bins=10, color='green', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.set_xlabel('Objective Improvement')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Local Search Improvements')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Summary statistics
        labels = ['Improved\nObjective', 'Worsened\nObjective', 'No Change', 'Improved\nFeasibility']
        values = [len(pos_improvements) / len(improvements) if len(improvements) > 0 else 0,
                  len(neg_improvements) / len(improvements) if len(improvements) > 0 else 0,
                  no_improvements / len(improvements) if len(improvements) > 0 else 0,
                  feasibility_improvement_rate]

        ax2.bar(labels, values, color=['green', 'red', 'gray', 'blue'], alpha=0.7)
        ax2.set_ylabel('Proportion')
        ax2.set_title('Local Search Effects')
        ax2.set_ylim(0, 1.0)
        for i, v in enumerate(values):
            ax2.text(i, v + 0.02, f'{v:.2f}', ha='center')

        plt.tight_layout()
        ls_path = os.path.join(config_dir, f"local_search_effects.png")
        plt.savefig(ls_path)
        plt.close()
        plots["local_search"] = ls_path

    # 6. Neural Network Rounding Impact
    if "rounding_impacts" in summary_metrics["Neural Network"]:
        plt.figure(figsize=(10, 6))
        rounding_values = summary_metrics["Neural Network"]["rounding_impacts"]

        plt.hist(rounding_values, bins=10, color='green', alpha=0.7)
        plt.axvline(np.mean(rounding_values), color='red', linestyle='dashed', linewidth=2)

        plt.xlabel('Average Rounding Impact')
        plt.ylabel('Frequency')
        plt.title(f'Neural Network Rounding Impact Distribution ({n_vars}x{n_constraints})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add text for mean value
        plt.text(np.mean(rounding_values) * 1.05, plt.ylim()[1] * 0.9,
                 f'Mean: {np.mean(rounding_values):.4f}',
                 color='red', fontsize=10)

        rounding_path = os.path.join(config_dir, f"nn_rounding_impact.png")
        plt.savefig(rounding_path)
        plt.close()
        plots["rounding_impact"] = rounding_path

    # 7. Boxplot of solve times
    plt.figure(figsize=(10, 6))
    data = [summary_metrics["MIP"]["times"],
            summary_metrics["Neural Network"]["times"],
            summary_metrics["Genetic Algorithm"]["times"]]

    plt.boxplot(data, labels=["MIP", "Neural Network", "Genetic Algorithm"])
    plt.ylabel('Solve Time (seconds)')
    plt.title(f'Solve Time Distribution ({n_vars}x{n_constraints})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add note about training
    plt.figtext(0.5, 0.01,
                f"Neural network was trained and tested ONLY on {n_vars}x{n_constraints} integer problems",
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
        Dictionary of plot paths, DataFrame with summary data
    """
    plots = {}

    # Create DataFrame from summaries
    rows = []
    for summary in summaries:
        config = summary["Configuration"]
        n_vars = summary["n_vars"]
        n_constraints = summary["n_constraints"]

        for method in ["MIP", "Neural Network", "Genetic Algorithm"]:
            method_data = summary[method]
            row = {
                "Configuration": config,
                "n_vars": n_vars,
                "n_constraints": n_constraints,
                "Method": method,
                "Avg_SolveTime": method_data.get("Avg_SolveTime"),
                "Avg_Objective": method_data.get("Avg_Objective"),
                "Avg_AccuracyPercent": method_data.get("Avg_AccuracyPercent", 100 if method == "MIP" else None),
                "Feasibility_Rate": method_data.get("FinalFeasibility_Rate",
                                                    method_data.get("Feasibility_Rate",
                                                                    method_data.get("Success_Rate", None)))
            }

            # Add method-specific metrics
            if method == "Neural Network":
                row.update({
                    "Avg_RoundingImpact": method_data.get("Avg_RoundingImpact"),
                    "Avg_LocalSearchImprovement": method_data.get("Avg_LocalSearchImprovement"),
                    "InitialFeasibility_Rate": method_data.get("InitialFeasibility_Rate"),
                    "FinalFeasibility_Rate": method_data.get("FinalFeasibility_Rate"),
                    "Feasibility_Improvement": method_data.get("Feasibility_Improvement")
                })

            rows.append(row)

    df = pd.DataFrame(rows)

    # 1. Plot solution time scaling
    plt.figure(figsize=(12, 8))

    for method, color in zip(["MIP", "Neural Network", "Genetic Algorithm"], ["blue", "green", "orange"]):
        method_data = df[df["Method"] == method]
        # Sort by number of variables
        method_data = method_data.sort_values("n_vars")

        plt.plot(method_data["n_vars"], method_data["Avg_SolveTime"], 'o-',
                 label=method, color=color, linewidth=2, markersize=8)

    plt.xlabel("Number of Variables")
    plt.ylabel("Average Solve Time (seconds)")
    plt.title("Solution Time Scaling by Problem Size (Integer Programming)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add methodology note
    plt.figtext(0.5, 0.01,
                "IMPORTANT: Each neural network was trained and tested ONLY on integer problems of the same size.\n"
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
    plt.title("Solution Accuracy by Problem Size (Integer Programming)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    plt.legend()

    # Add methodology note
    plt.figtext(0.5, 0.01,
                "IMPORTANT: Each neural network was trained and tested ONLY on integer problems of the same size.\n"
                "There is NO cross-testing between different problem sizes.",
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    accuracy_scaling_path = os.path.join(output_dir, "accuracy_scaling.png")
    plt.savefig(accuracy_scaling_path, bbox_inches='tight')
    plt.close()
    plots["accuracy_scaling"] = accuracy_scaling_path

    # 3. Plot feasibility scaling
    plt.figure(figsize=(12, 8))

    for method, color, marker in zip(
            ["MIP", "Neural Network", "Genetic Algorithm"],
            ["blue", "green", "orange"],
            ["o", "s", "^"]
    ):
        method_data = df[df["Method"] == method]
        # Sort by number of variables
        method_data = method_data.sort_values("n_vars")

        plt.plot(method_data["n_vars"], method_data["Feasibility_Rate"], marker + '-',
                 label=method, color=color, linewidth=2, markersize=8)

    plt.xlabel("Number of Variables")
    plt.ylabel("Feasibility Rate")
    plt.title("Solution Feasibility by Problem Size (Integer Programming)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()

    # Add methodology note
    plt.figtext(0.5, 0.01,
                "IMPORTANT: Each neural network was trained and tested ONLY on integer problems of the same size.\n"
                "There is NO cross-testing between different problem sizes.",
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    feasibility_scaling_path = os.path.join(output_dir, "feasibility_scaling.png")
    plt.savefig(feasibility_scaling_path, bbox_inches='tight')
    plt.close()
    plots["feasibility_scaling"] = feasibility_scaling_path

    # 4. NEW: Accuracy distribution across all configurations
    try:
        plt.figure(figsize=(12, 8))

        # Define accuracy bins
        accuracy_bins = [
            (95, 100.1, "95-100%"),  # Using 100.1 to include exactly 100%
            (90, 95, "90-95%"),
            (80, 90, "80-90%"),
            (50, 80, "50-80%"),
            (0, 50, "<50%")
        ]

        methods = ["Neural Network", "Genetic Algorithm"]
        all_method_data = {}

        for method in methods:
            # Get all accuracy values directly from the detailed results
            accs = []
            for summary in summaries:
                if method in summary and "accuracies" in summary[method]:
                    # Get individual accuracy values if available
                    if isinstance(summary[method]["accuracies"], list):
                        method_accs = [a for a in summary[method]["accuracies"] if a is not None]
                        accs.extend(method_accs)
                    # Otherwise use the average
                    elif "Avg_AccuracyPercent" in summary[method]:
                        avg_acc = summary[method]["Avg_AccuracyPercent"]
                        if avg_acc is not None:
                            accs.append(avg_acc)

            all_method_data[method] = accs

        # Create the plot
        bar_width = 0.35
        colors = ["green", "orange"]

        # Create bars for each method and each bin
        for i, method in enumerate(methods):
            accuracies = all_method_data[method]
            bin_counts = []

            if accuracies:
                for low, high, _ in accuracy_bins:
                    count = sum(1 for acc in accuracies if low <= acc < high)
                    bin_counts.append(count / len(accuracies) * 100)

                x = np.arange(len(accuracy_bins))
                plt.bar(x + i * bar_width - bar_width / 2, bin_counts, width=bar_width,
                        label=method, color=colors[i], alpha=0.8)

        plt.xlabel('Accuracy Range')
        plt.ylabel('Percentage of Problems (%)')
        plt.title('Overall Distribution of Solution Accuracy Across All Configurations')
        plt.xticks(np.arange(len(accuracy_bins)), [label for _, _, label in accuracy_bins])
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        accuracy_overall_path = os.path.join(output_dir, "accuracy_overall_distribution.png")
        plt.savefig(accuracy_overall_path, bbox_inches='tight')
        plt.close()
        plots["accuracy_overall_distribution"] = accuracy_overall_path
    except Exception as e:
        print(f"Error generating overall accuracy distribution: {e}")

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

        # Add MIP results
        rows.append({
            "Configuration": config,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "Method": "MIP",
            "Avg_SolveTime": summary["MIP"]["Avg_SolveTime"],
            "StdDev_SolveTime": summary["MIP"]["StdDev_SolveTime"],
            "Avg_Objective": summary["MIP"]["Avg_Objective"],
            "Avg_SolveTimeDiff": 0.0,
            "Avg_AccuracyPercent": 100.0,
            "FasterThanMIP_Ratio": None,
            "Avg_Iterations": summary["MIP"]["Avg_Iterations"],
            "Success_Rate": summary["MIP"]["Success_Rate"]
        })

        # Add Neural Network results
        nn_data = {
            "Configuration": config,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "Method": "Enhanced Neural Network",
            "Avg_SolveTime": summary["Neural Network"]["Avg_SolveTime"],
            "StdDev_SolveTime": summary["Neural Network"]["StdDev_SolveTime"],
            "Avg_Objective": summary["Neural Network"]["Avg_Objective"],
            "Avg_SolveTimeDiff": summary["Neural Network"]["Avg_SolveTimeDiff"],
            "Avg_AccuracyPercent": summary["Neural Network"]["Avg_AccuracyPercent"],
            "FasterThanMIP_Ratio": summary["Neural Network"]["FasterThanMIP_Ratio"],
            "Avg_MaxConstraintViolation": summary["Neural Network"]["Avg_MaxConstraintViolation"],
            "InitialFeasibility_Rate": summary["Neural Network"]["InitialFeasibility_Rate"],
            "FinalFeasibility_Rate": summary["Neural Network"]["FinalFeasibility_Rate"],
            "Feasibility_Improvement": summary["Neural Network"]["Feasibility_Improvement"],
            "Time_TTest_Statistic": summary["Neural Network"]["Time_TTest_Statistic"],
            "Time_TTest_PValue": summary["Neural Network"]["Time_TTest_PValue"],
            "Obj_TTest_Statistic": summary["Neural Network"]["Obj_TTest_Statistic"],
            "Obj_TTest_PValue": summary["Neural Network"]["Obj_TTest_PValue"],
            "Training_Time": summary["Neural Network"]["Training_Time"]
        }

        # Add rounding impact and local search improvement if available
        if "Avg_RoundingImpact" in summary["Neural Network"]:
            nn_data["Avg_RoundingImpact"] = summary["Neural Network"]["Avg_RoundingImpact"]
        if "Avg_LocalSearchImprovement" in summary["Neural Network"]:
            nn_data["Avg_LocalSearchImprovement"] = summary["Neural Network"]["Avg_LocalSearchImprovement"]

        rows.append(nn_data)

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
            "FasterThanMIP_Ratio": summary["Genetic Algorithm"]["FasterThanMIP_Ratio"],
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
            "1. A neural network trained SPECIFICALLY for that exact integer problem size",
            "2. Testing done ONLY on integer problems matching the trained configuration",
            "3. No transfer or sharing of models between different problem sizes",
            "",
            "The Enhanced Neural Network approach includes:",
            "- Improved architecture with attention mechanism and dynamic skip connections",
            "- Sharper constraint-aware loss function to better enforce integer solutions",
            "- Enhanced local search with adaptive strategies and random perturbations",
            "- Variable prioritization based on objective coefficients",
            "- More training data and advanced training techniques",
            "",
            "All problems involve integer variables, making them more complex than continuous LP problems.",
            "",
            "This ensures fair comparison of methods within each specific problem configuration."
        ]
    })

    # Write to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(run_dir, f"enhanced_ilp_solver_results_{timestamp}.xlsx")

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
                            "trained and tested ONLY on integer problems of this exact size.")
            worksheet.write(2, config_data.shape[1] + 1,
                            "Each configuration has its own separate neural network.")
            worksheet.write(3, config_data.shape[1] + 1,
                            "Enhanced local search refinement is applied to neural network solutions.")

        # Write other sheets
        df_detailed.to_excel(writer, sheet_name="Detailed_Results", index=False)
        df_problems.to_excel(writer, sheet_name="Problems", index=False)
        df_models.to_excel(writer, sheet_name="Model_Summaries", index=False)

        # Add annotation to detailed results
        detailed_sheet = writer.sheets["Detailed_Results"]
        detailed_sheet.write(0, df_detailed.shape[1] + 1, "IMPORTANT NOTE:")
        detailed_sheet.write(1, df_detailed.shape[1] + 1, "Each neural network was trained and tested")
        detailed_sheet.write(2, df_detailed.shape[1] + 1, "ONLY on integer problems matching its configuration.")
        detailed_sheet.write(3, df_detailed.shape[1] + 1,
                             "The enhanced neural network includes improved local search refinement.")

    return excel_path


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

    try:
        nn_model, nn_history, nn_metadata = train_enhanced_nn_model(n_vars, n_constraints, num_train, verbose)

        # Plot training history
        nn_training_plot = plot_nn_training_history(nn_history, config_dir)
    except Exception as e:
        print(f"Error during neural network training: {str(e)}")
        print("Continuing with experiment using simplified model...")

        # Create a simplified fallback model
        input_dim = n_vars + n_constraints * n_vars + n_constraints

        # Simple model with fewer layers and no complex architecture
        inputs = tf.keras.Input(shape=(input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(n_vars, activation='linear')(x)
        nn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        nn_history = None
        nn_metadata = {
            "configuration": f"{n_vars}x{n_constraints}",
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "input_dim": input_dim,
            "output_dim": n_vars,
            "problem_size": n_vars * n_constraints,
            "model_width": 128,
            "model_depth": 2,
            "epochs": 0,
            "batch_size": NN_BATCH_SIZE,
            "training_samples": num_train,
            "training_time": 0,
            "final_loss": 0,
            "final_val_loss": 0,
            "final_mae": 0,
            "final_val_mae": 0,
            "model_summary": "Fallback model due to training error"
        }

    # STEP 2: Test on random problems of the same configuration
    if verbose:
        print_section(f"Testing {num_test} problems for {n_vars}x{n_constraints} configuration")
        print(f"Using neural network trained ONLY for {n_vars}x{n_constraints} problems")

    # Data containers
    detailed_results = []
    problems_info = []
    summary_metrics = {
        "MIP": {"times": [], "objectives": [], "iterations": []},
        "Neural Network": {"times": [], "objectives": [], "accuracies": [],
                           "faster": [], "constraint_violations": [], "rounding_impacts": [],
                           "local_search_improvements": [], "initial_feasibility": [], "final_feasibility": []},
        "Genetic Algorithm": {"times": [], "objectives": [], "accuracies": [],
                              "faster": [], "generations": [], "constraint_violations": []}
    }

    # For warm starting GA
    warm_start_population = None

    # Run tests
    for i in range(num_test):
        if verbose:
            print(f"\nTest Problem {i + 1}/{num_test} for {n_vars}x{n_constraints}")

        # Generate random ILP problem
        ilp = generate_ilp_problem(n_vars, n_constraints)
        ilp_id = f"Config_{n_vars}x{n_constraints}_Test_{i + 1}"

        # Record problem details
        problems_info.append({
            "ILP_ID": ilp_id,
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "c": str(ilp["c"]),
            "A_ub": str(ilp["A_ub"].tolist()),
            "b_ub": str(ilp["b_ub"].tolist()),
            "Bounds": f"[{ilp['bounds'].lb}, {ilp['bounds'].ub}] (integers only)",
            "Problem_Stats": str(ilp["stats"])
        })

        # --- Solve using MIP ---
        sol_m, obj_m, time_m, mip_stats = solve_mip(ilp, verbose)

        summary_metrics["MIP"]["times"].append(time_m)
        summary_metrics["MIP"]["objectives"].append(obj_m)
        summary_metrics["MIP"]["iterations"].append(mip_stats["iterations"])

        detailed_results.append({
            "ILP_ID": ilp_id,
            "Configuration": f"{n_vars} vars, {n_constraints} constraints",
            "Method": "MIP (HiGHS)",
            "Solution": str(sol_m),
            "Objective": obj_m,
            "SolveTime_sec": time_m,
            "Feasible": sol_m is not None,
            "SolveTimeDifference": 0.0,
            "AccuracyPercent": 100.0,
            "FasterThanMIP": False,
            "Iterations": mip_stats["iterations"],
            "Status": mip_stats["status"]
        })

        # --- Solve using Enhanced Neural Network ---
        if verbose:
            print(f"  Evaluating enhanced neural network model specifically trained for {n_vars}x{n_constraints}...")

        sol_nn, obj_nn, time_nn, feasible_nn, nn_constraint_stats = evaluate_enhanced_nn(nn_model, ilp, verbose)

        # Compute comparison metrics
        nn_metrics = compute_comparison_metrics(time_nn, obj_nn, time_m, obj_m, nn_constraint_stats)

        summary_metrics["Neural Network"]["times"].append(time_nn)
        summary_metrics["Neural Network"]["objectives"].append(obj_nn)
        summary_metrics["Neural Network"]["accuracies"].append(nn_metrics["accuracy"])
        summary_metrics["Neural Network"]["faster"].append(nn_metrics["faster_than_mip"])
        summary_metrics["Neural Network"]["constraint_violations"].append(nn_constraint_stats["max_violation"])
        summary_metrics["Neural Network"]["rounding_impacts"].append(nn_constraint_stats["rounding_impact"])
        summary_metrics["Neural Network"]["local_search_improvements"].append(
            nn_constraint_stats["local_search_improvement"])
        summary_metrics["Neural Network"]["initial_feasibility"].append(nn_constraint_stats["initial_feasibility"])
        summary_metrics["Neural Network"]["final_feasibility"].append(nn_constraint_stats["final_feasibility"])

        detailed_results.append({
            "ILP_ID": ilp_id,
            "Configuration": f"{n_vars} vars, {n_constraints} constraints",
            "Method": "Enhanced Neural Network",
            "Solution": str(sol_nn),
            "Objective": obj_nn,
            "SolveTime_sec": time_nn,
            "Feasible": feasible_nn,
            "SolveTimeDifference": nn_metrics["time_diff"],
            "AccuracyPercent": nn_metrics["accuracy"],
            "FasterThanMIP": nn_metrics["faster_than_mip"],
            "Speedup": nn_metrics["speedup"],
            "MaxConstraintViolation": nn_constraint_stats["max_violation"],
            "AvgConstraintViolation": nn_constraint_stats["avg_violation"],
            "NumViolatedConstraints": nn_constraint_stats["num_violated"],
            "RoundingImpact": nn_constraint_stats["rounding_impact"],
            "LocalSearchImprovement": nn_constraint_stats["local_search_improvement"],
            "InitialFeasibility": nn_constraint_stats["initial_feasibility"],
            "FinalFeasibility": nn_constraint_stats["final_feasibility"]
        })

        # --- Solve using Genetic Algorithm ---
        sol_ga, obj_ga, time_ga, final_population, ga_stats = solve_ga(
            ilp, initial_population=warm_start_population, verbose=verbose
        )

        # Update warm start population for next problem
        warm_start_population = final_population

        # Compute comparison metrics
        ga_metrics = compute_comparison_metrics(time_ga, obj_ga, time_m, obj_m, ga_stats["constraint_stats"])

        summary_metrics["Genetic Algorithm"]["times"].append(time_ga)
        summary_metrics["Genetic Algorithm"]["objectives"].append(obj_ga)
        summary_metrics["Genetic Algorithm"]["accuracies"].append(ga_metrics["accuracy"])
        summary_metrics["Genetic Algorithm"]["faster"].append(ga_metrics["faster_than_mip"])
        summary_metrics["Genetic Algorithm"]["generations"].append(ga_stats["generations"])
        summary_metrics["Genetic Algorithm"]["constraint_violations"].append(
            ga_stats["constraint_stats"]["max_violation"])

        detailed_results.append({
            "ILP_ID": ilp_id,
            "Configuration": f"{n_vars} vars, {n_constraints} constraints",
            "Method": "Genetic Algorithm",
            "Solution": str(sol_ga),
            "Objective": obj_ga,
            "SolveTime_sec": time_ga,
            "Feasible": (obj_ga is not None),
            "SolveTimeDifference": ga_metrics["time_diff"],
            "AccuracyPercent": ga_metrics["accuracy"],
            "FasterThanMIP": ga_metrics["faster_than_mip"],
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

    # Time comparison statistical tests
    nn_time_ttest = safe_ttest(summary_metrics["Neural Network"]["times"],
                               summary_metrics["MIP"]["times"])
    ga_time_ttest = safe_ttest(summary_metrics["Genetic Algorithm"]["times"],
                               summary_metrics["MIP"]["times"])

    # Objective comparison statistical tests (filter None values)
    nn_objectives = [o for o, s in zip(summary_metrics["Neural Network"]["objectives"],
                                       summary_metrics["MIP"]["objectives"])
                     if o is not None and s is not None]
    mip_objectives_nn = [s for o, s in zip(summary_metrics["Neural Network"]["objectives"],
                                           summary_metrics["MIP"]["objectives"])
                         if o is not None and s is not None]

    ga_objectives = [o for o, s in zip(summary_metrics["Genetic Algorithm"]["objectives"],
                                       summary_metrics["MIP"]["objectives"])
                     if o is not None and s is not None]
    mip_objectives_ga = [s for o, s in zip(summary_metrics["Genetic Algorithm"]["objectives"],
                                           summary_metrics["MIP"]["objectives"])
                         if o is not None and s is not None]

    nn_obj_ttest = safe_ttest(nn_objectives, mip_objectives_nn)
    ga_obj_ttest = safe_ttest(ga_objectives, mip_objectives_ga)

    # Compute averages
    avg_summary = {}

    # MIP averages
    mip_times = summary_metrics["MIP"]["times"]
    mip_objectives = [obj for obj in summary_metrics["MIP"]["objectives"] if obj is not None]
    mip_iterations = [it for it in summary_metrics["MIP"]["iterations"] if it is not None]

    avg_summary["MIP"] = {
        "Avg_SolveTime": np.mean(mip_times) if mip_times else None,
        "StdDev_SolveTime": np.std(mip_times) if len(mip_times) > 1 else None,
        "Avg_Objective": np.mean(mip_objectives) if mip_objectives else None,
        "Avg_Iterations": np.mean(mip_iterations) if mip_iterations else None,
        "Success_Rate": sum(1 for obj in summary_metrics["MIP"]["objectives"] if obj is not None) / num_test
    }

    # Neural Network averages
    nn_times = summary_metrics["Neural Network"]["times"]
    nn_objectives = [obj for obj in summary_metrics["Neural Network"]["objectives"] if obj is not None]
    nn_accuracies = [acc for acc in summary_metrics["Neural Network"]["accuracies"] if acc is not None]
    nn_violations = summary_metrics["Neural Network"]["constraint_violations"]
    nn_rounding = summary_metrics["Neural Network"]["rounding_impacts"]
    nn_ls_improvements = summary_metrics["Neural Network"]["local_search_improvements"]
    nn_initial_feasible = summary_metrics["Neural Network"]["initial_feasibility"]
    nn_final_feasible = summary_metrics["Neural Network"]["final_feasibility"]

    avg_summary["Neural Network"] = {
        "Avg_SolveTime": np.mean(nn_times) if nn_times else None,
        "StdDev_SolveTime": np.std(nn_times) if len(nn_times) > 1 else None,
        "Avg_Objective": np.mean(nn_objectives) if nn_objectives else None,
        "Avg_SolveTimeDiff": np.mean([time_nn - time_m for time_nn, time_m in
                                      zip(nn_times, mip_times)]) if nn_times and mip_times else None,
        "Avg_AccuracyPercent": np.mean(nn_accuracies) if nn_accuracies else None,
        "FasterThanMIP_Ratio": np.mean(
            [1 if faster else 0 for faster in summary_metrics["Neural Network"]["faster"]]),
        "Avg_MaxConstraintViolation": np.mean(nn_violations) if nn_violations else None,
        "Avg_RoundingImpact": np.mean(nn_rounding) if nn_rounding else None,
        "Avg_LocalSearchImprovement": np.mean(nn_ls_improvements) if nn_ls_improvements else None,
        "InitialFeasibility_Rate": np.mean([1 if feasible else 0 for feasible in nn_initial_feasible]),
        "FinalFeasibility_Rate": np.mean([1 if feasible else 0 for feasible in nn_final_feasible]),
        "Feasibility_Improvement": np.mean([1 if final and not initial else 0
                                            for initial, final in zip(nn_initial_feasible, nn_final_feasible)]),
        "Time_TTest_Statistic": nn_time_ttest[0] if nn_time_ttest is not None else None,
        "Time_TTest_PValue": nn_time_ttest[1] if nn_time_ttest is not None else None,
        "Obj_TTest_Statistic": nn_obj_ttest[0] if nn_obj_ttest is not None else None,
        "Obj_TTest_PValue": nn_obj_ttest[1] if nn_obj_ttest is not None else None,
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
        "Avg_SolveTimeDiff": np.mean([time_ga - time_m for time_ga, time_m in
                                      zip(ga_times, mip_times)]) if ga_times and mip_times else None,
        "Avg_AccuracyPercent": np.mean(ga_accuracies) if ga_accuracies else None,
        "FasterThanMIP_Ratio": np.mean(
            [1 if faster else 0 for faster in summary_metrics["Genetic Algorithm"]["faster"]]),
        "Avg_Generations": np.mean(ga_generations) if ga_generations else None,
        "Avg_MaxConstraintViolation": np.mean(ga_violations) if ga_violations else None,
        "Feasibility_Rate": sum(1 for v in ga_violations if v <= 1e-6) / len(ga_violations) if ga_violations else 0,
        "Time_TTest_Statistic": ga_time_ttest[0] if ga_time_ttest is not None else None,
        "Time_TTest_PValue": ga_time_ttest[1] if ga_time_ttest is not None else None,
        "Obj_TTest_Statistic": ga_obj_ttest[0] if ga_obj_ttest is not None else None,
        "Obj_TTest_PValue": ga_obj_ttest[1] if ga_obj_ttest is not None else None
    }

    # STEP 4: Generate visualizations
    if verbose:
        print_section(f"Generating visualizations for {n_vars}x{n_constraints}")

    try:
        plots = generate_plots_v2(summary_metrics, n_vars, n_constraints, config_dir)
    except Exception as e:
        print(f"Error generating plots: {e}")
        # Create an empty plots dictionary if visualization fails
        plots = {}
        print("Continuing without visualizations")

    # Calculate total time for configuration
    config_total_time = time.time() - start_time
    if verbose:
        print(f"Configuration {n_vars}x{n_constraints} completed in {config_total_time:.2f} seconds")

    # Package the results for this configuration
    config_summary = {
        "Configuration": f"{n_vars} vars, {n_constraints} constraints",
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "MIP": avg_summary["MIP"],
        "Neural Network": avg_summary["Neural Network"],
        "Genetic Algorithm": avg_summary["Genetic Algorithm"],
        "Model_Summary": nn_metadata["model_summary"],
        "Total_Config_Time": config_total_time
    }

    return config_summary, detailed_results, problems_info, plots


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

        try:
            cross_plots, summary_df = generate_cross_config_plots(all_summaries, RUN_DIR)
            all_plots.update(cross_plots)
        except Exception as e:
            print(f"Error generating cross-configuration plots: {e}")
            print("Continuing without cross-configuration plots")

    return all_summaries, all_detailed_results, all_problems_info, all_plots


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Integer LP Solver Comparison Framework")

    # Configuration options
    parser.add_argument("--n_vars", type=int, help="Number of variables (for single configuration)")
    parser.add_argument("--n_constraints", type=int, help="Number of constraints (for single configuration)")
    parser.add_argument("--train_size", type=int, help="Number of training samples", default=DEFAULT_NUM_TRAIN)
    parser.add_argument("--test_size", type=int, help="Number of test problems", default=DEFAULT_NUM_TEST)
    parser.add_argument("--output_dir", type=str, help="Custom output directory")

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
    print_header("ENHANCED INTEGER LP SOLVER COMPARISON FRAMEWORK", char="*")
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
        args.train_size = min(args.train_size, 50)
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
    print(f"  - Neural network base width: {NN_BASE_WIDTH}, layers: {NN_LAYERS}")
    print(f"  - Local search iterations: {NN_LOCAL_SEARCH_ITERATIONS}")
    print(f"  - Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"  - All variables constrained to INTEGER values")

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
