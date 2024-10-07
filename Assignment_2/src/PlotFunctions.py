import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set Seaborn theme for better styling
sns.set(style='whitegrid')

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

# Parameters
num_points = 100  # Number of points in the grid

# Set different ranges for the functions
min_val_sphere_alpine = -10
max_val_sphere_alpine = 10
min_val_rastrigin = -5.12
max_val_rastrigin = 5.12

# Generate input grids and evaluate functions
X_sphere, y_sphere = generate_input_grid_and_evaluate(sphere_function, num_points, 1, min_val_sphere_alpine, max_val_sphere_alpine)
X_alpine, y_alpine = generate_input_grid_and_evaluate(alpine_function, num_points, 1, min_val_sphere_alpine, max_val_sphere_alpine)
X_rastrigin, y_rastrigin = generate_input_grid_and_evaluate(rastrigin_function, num_points, 1, min_val_rastrigin, max_val_rastrigin)

# Standard scale X and y values
X_sphere_scaled, y_sphere_scaled = scale_X_y(X_sphere, y_sphere)
X_alpine_scaled, y_alpine_scaled = scale_X_y(X_alpine, y_alpine)
X_rastrigin_scaled, y_rastrigin_scaled = scale_X_y(X_rastrigin, y_rastrigin)

# Plot and save Sphere function as PDF
plt.figure()
sns.lineplot(x=X_sphere_scaled.flatten(), y=y_sphere_scaled.flatten(), color='b')
sns.scatterplot(x=X_sphere_scaled.flatten(), y=y_sphere_scaled.flatten(), color='b', s=10)
plt.xlabel('Scaled X')
plt.ylabel('Scaled f(X)')
plt.grid()
plt.savefig('sphere_function_scaled.pdf', format='pdf', bbox_inches='tight')

# Plot and save Alpine function as PDF
plt.figure()
sns.lineplot(x=X_alpine_scaled.flatten(), y=y_alpine_scaled.flatten(), color='g')
sns.scatterplot(x=X_alpine_scaled.flatten(), y=y_alpine_scaled.flatten(), color='g', s=10)
plt.xlabel('Scaled X')
plt.ylabel('Scaled f(X)')
plt.grid()
plt.savefig('alpine_function_scaled.pdf', format='pdf', bbox_inches='tight')

# Plot and save Rastrigin function as PDF
plt.figure()
sns.lineplot(x=X_rastrigin_scaled.flatten(), y=y_rastrigin_scaled.flatten(), color='r')
sns.scatterplot(x=X_rastrigin_scaled.flatten(), y=y_rastrigin_scaled.flatten(), color='r', s=10)
plt.xlabel('Scaled X')
plt.ylabel('Scaled f(X)')
plt.grid()
plt.savefig('rastrigin_function_scaled.pdf', format='pdf', bbox_inches='tight')

# Close all figures to free up memory
plt.close('all')
