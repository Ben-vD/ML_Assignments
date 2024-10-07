import numpy as np
from sklearn import datasets
import NeuralNetwork as nn
import UtilFunctions as uf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Enable LaTeX formatting for text rendering
plt.rc('text', usetex=True)

data = np.loadtxt("../data/DryBeansNorm.csv", delimiter=',')
X = data[:, :-1]
y = data[:, -1].astype(int)

# Define fitting functions
fitting_functions = [0, 1, 2, 3, 4]  # Your fitting function indices
fitting_labels = [r"PSO", r"PSO_{s}", r"PSO_{r}", r"dPSO", r"mPSO"]  # Use raw string for subscripts

fitting_functions = [3, 4]  # Your fitting function indices
fitting_labels = [r"dPSO", r"mPSO"]  # Use raw string for subscripts

# Initialize lists to hold average fitness and diversity for each fitting function
all_average_fitness = []
all_average_diversity = []
all_labels = []

# Define a worker function for each experiment
def run_experiment(experiment_index, fit_function):
    np.random.seed(experiment_index)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Initialize the neural network with the specified fitting function
    neural_network = nn.NeuralNetwork(16, [8, 7], True, 0.001, ["sigmoid", "sigmoid"])

    # Train the neural network and get fitness and diversity arrays
    fitness, diversity = neural_network.fit(X_train, uf.oheLabels(y_train), fit_function)

    test_accuracy = neural_network.score(X_test, y_test)
    return fitness, diversity, test_accuracy

# Run experiments for each fitting function
if __name__ == '__main__':
    num_experiments = 10

    for fit_index, fit_function in enumerate(fitting_functions):
        fitness_results = []
        diversity_results = []
        test_accuracies = []

        # Use ProcessPoolExecutor to run experiments in parallel
        with ProcessPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = [executor.submit(run_experiment, i, fit_function) for i in range(num_experiments)]

            # Collect the results as the futures complete
            for future in as_completed(futures):
                fitness, diversity, test_accuracy = future.result()
                fitness_results.append(fitness)
                diversity_results.append(diversity)
                test_accuracies.append(test_accuracy)

        # Convert lists to arrays for easy averaging
        fitness_results = np.array(fitness_results)
        diversity_results = np.array(diversity_results)

        # Compute the average fitness and diversity across the experiments
        average_fitness = np.mean(fitness_results, axis=0)
        average_diversity = np.mean(diversity_results, axis=0)

        # Store the results for plotting later
        all_average_fitness.append(average_fitness)
        all_average_diversity.append(average_diversity)
        all_labels.append(fitting_labels[fit_index])

        # Compute the average and standard deviation of test accuracy
        average_test_accuracy = np.mean(test_accuracies)
        std_test_accuracy = np.std(test_accuracies)

        # Print the final average test accuracy and its standard deviation
        print(f'Fitting Function {fitting_labels[fit_index]}:')
        print(f'Average Test Accuracy: {average_test_accuracy:.2f}')
        print(f'Standard Deviation of Test Accuracies: {std_test_accuracy:.2f}\n')

    # Plotting all average fitness on one graph
    plt.figure(figsize=(8, 5))
    for idx, avg_fitness in enumerate(all_average_fitness):
        plt.plot(avg_fitness, label=all_labels[idx])
    #plt.title('Average Fitness Over Time')
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
    #plt.title('Average Diversity Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Diversity')
    plt.grid()
    plt.legend(loc='upper right', fontsize=10)  # Position the legend at the top right
    plt.savefig('average_diversity.pdf')  # Save the diversity plot as PDF
    plt.close()  # Close the plot

    print("Plots saved successfully!")
