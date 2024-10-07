import numpy as np
import math
from sklearn.preprocessing import StandardScaler

# 1. Simple Polynomial Function
def simple_polynomial(x):
    return x**2

# 2. Sine Function with Noise
def sine_with_noise(x, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=x.shape)
    return np.sin(x) + noise

# 3. Gamma Function using Python's math library
def gamma_function(z):
    return math.gamma(z)

def sphere_function(X):
    return np.sum(X**2, axis=1)  # Sum of squares for Sphere function

def rastrigin_function(X):
    n = X.shape[1]
    return 10 * n + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)  # Rastrigin function

def alpine_function(X):
    return np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1) 

def generate_input_grid_and_evaluate(func, num_points_per_dim, num_dims, min_val, max_val):
    # Generate grid points based on the specified range
    grid = np.linspace(min_val, max_val, num_points_per_dim)

    # Create a meshgrid for the specified number of dimensions
    grid_mesh = np.meshgrid(*[grid] * num_dims, indexing='ij')

    # Stack and reshape the meshgrid to create a matrix of input points
    X = np.stack(grid_mesh, axis=-1).reshape(-1, num_dims)

    # Evaluate the function at each point in X to get Y values
    y = func(X).reshape(-1, 1)

    return X, y

# Function to scale X and y using StandardScaler
def scale_X_y(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled