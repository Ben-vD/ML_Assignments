import FunctionAprox as fa
import numpy as np
from sklearn import datasets
import NeuralNetwork as nn
import UtilFunctions as uf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns

plt.rc('text', usetex=True)

X, y = fa.generate_input_grid_and_evaluate(fa.rastrigin_function, 100, 1, -5.12, 5.12)
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y)

# Define fitting functions
fitting_functions = [0, 1, 2, 3, 4]  # Your fitting function indices
fitting_labels = [r"PSO", r"PSO_{s}", r"PSO_{r}", r"dPSO", r"mPSO"]  # Use raw string for subscripts

# Initialize lists to hold average fitness, diversity, and best models for each fitting function
all_average_fitness = []
all_average_diversity = []
all_labels = []
all_mses = []
best_models_per_algorithm = []  # To store the best models for each algorithm

# Define a worker function for each experiment
def run_experiment(experiment_index, fit_function):
    np.random.seed(experiment_index)

    # Initialize the neural network with the specified fitting function
    neural_network = nn.NeuralNetwork(1, [32, 1], False, 0.001, ["sigmoid", "linear"])

    # Train the neural network and get fitness and diversity arrays
    fitness, diversity = neural_network.fit(X, y, fit_function)

    mse = neural_network.score(X, y)

    return fitness, diversity, mse, neural_network

# Run experiments for each fitting function
if __name__ == '__main__':
    num_experiments = 10  # Set to 10 experiments

    for fit_index, fit_function in enumerate(fitting_functions):
        fitness_results = []
        diversity_results = []
        mses = []
        models = []

        # Use ProcessPoolExecutor to run experiments in parallel
        with ProcessPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = [executor.submit(run_experiment, i, fit_function) for i in range(num_experiments)]

            # Collect the results as the futures complete
            for future in as_completed(futures):
                fitness, diversity, mse, model = future.result()
                fitness_results.append(fitness)
                diversity_results.append(diversity)
                mses.append(mse)
                models.append(model)

        # Convert lists to arrays for easy averaging
        fitness_results = np.array(fitness_results)
        diversity_results = np.array(diversity_results)
        mses = np.array(mses)
        models = np.array(models)

        # Compute the average fitness and diversity across the experiments
        average_fitness = np.mean(fitness_results, axis=0)
        average_diversity = np.mean(diversity_results, axis=0)

        # Store the results for plotting later
        all_average_fitness.append(average_fitness)
        all_average_diversity.append(average_diversity)
        all_labels.append(fitting_labels[fit_index])

        # Compute the average and standard deviation of test accuracy
        average_test_accuracy = np.mean(mses)
        std_test_accuracy = np.std(mses)

        # Print the final average test accuracy and its standard deviation
        print(f'Fitting Function {fitting_labels[fit_index]}:')
        print(f'Average Test Accuracy: {average_test_accuracy:.4f}')
        print(f'Standard Deviation of Test Accuracies: {std_test_accuracy:.4f}\n')

        # Select the model with the lowest MSE for this fitting function
        best_model_index = np.argmin(mses)
        best_model = models[best_model_index]
        best_models_per_algorithm.append(best_model)  # Store the best model for this algorithm

    # Plotting all average fitness on one graph
    plt.figure(figsize=(8, 5))
    for idx, avg_fitness in enumerate(all_average_fitness):
        plt.plot(avg_fitness, label=all_labels[idx])
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.grid()
    plt.legend(loc='upper right', fontsize=10)  # Position the legend at the top right
    plt.savefig('average_fitness.pdf')  # Save the fitness plot as PDF
    plt.close()  # Close the plot

    # Plotting all average diversity on another graph
    plt.figure(figsize=(8, 5))
    for idx, avg_diversity in enumerate(all_average_diversity):
        plt.plot(avg_diversity, label=all_labels[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Diversity')
    plt.grid()
    plt.legend(loc='upper right', fontsize=10)  # Position the legend at the top right
    plt.savefig('average_diversity.pdf')  # Save the diversity plot as PDF
    plt.close()  # Close the plot

    # Plotting the best approximation for each algorithm
    plt.figure(figsize=(8, 5))

    # Plot the true function (e.g., Sphere function)
    plt.plot(X, y, label='True Function', color='black', linewidth=2)

    # Plot the best model for each algorithm
    for fit_index, best_model in enumerate(best_models_per_algorithm):
        # Use the best model to predict on the input data
        y_pred = best_model.predict(X)

        # Plot the predictions of the best model
        plt.plot(X, y_pred, label=fitting_labels[fit_index])

    # Set plot labels and legend
    plt.xlabel('X values')
    plt.ylabel('Function Approximation')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid()

    # Save and display the plot
    plt.savefig('best_approximation.pdf')  # Save the plot as PDF
    plt.close()  # Close the plot

    print("Plots saved successfully!")
