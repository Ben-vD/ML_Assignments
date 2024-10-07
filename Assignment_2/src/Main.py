import numpy as np
from sklearn import datasets
import NeuralNetwork as nn
import UtilFunctions as uf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import FunctionAprox as fa
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import sys


iris = datasets.load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

#cancer = datasets.load_breast_cancer()
#X = StandardScaler().fit_transform(cancer.data)
#y = cancer.target

#data = np.loadtxt("../data/DryBeansNorm.csv", delimiter=',')
#X = data[:, :-1]
#y = data[:, -1].astype(int)

X, y = fa.generate_input_grid_and_evaluate(fa.rastrigin_function, 100, 1, -10, 10)
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y)
sns.lineplot(x = X[:, 0], y = y[:, 0])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
neural_network = nn.NeuralNetwork(1, [50, 1], False, 0.001, ["sigmoid", "linear"])
neural_network.fit(X, y, 4)
print(neural_network.score(X, y))

sns.lineplot(x = X[:, 0], y = neural_network.predict(X)[:, 0])
sns.lineplot(x = X[:, 0], y = y[:, 0])
plt.show()
sys.exit()

# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement
delta = 0.01  # Minimum change to consider an improvement
max_hidden_neurons = 50  # Maximum number of hidden neurons to test

# Function to train and evaluate the model for one fold
def train_and_evaluate_fold(train_index, val_index, hidden_neurons, X_train, y_train):
    neural_network = nn.NeuralNetwork(4, [hidden_neurons, 3], True, 0.001, ["sigmoid", "sigmoid"])
    
    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    # Fit the model on the training fold
    neural_network.fit(X_fold_train, uf.oheLabels(y_fold_train), 0) #

    # Calculate the score on the validation fold
    score = neural_network.score(X_fold_val, y_fold_val)
    
    return score

# Function to train and evaluate with early stopping
def train_with_early_stopping(hidden_neurons, X_train, y_train):
    kf = KFold(n_splits=5, shuffle=True)
    scores = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(train_and_evaluate_fold, train_index, val_index, hidden_neurons, X_train, y_train)
            for train_index, val_index in kf.split(X_train)
        ]
        
        scores = [future.result() for future in futures]

    print(scores)
    return np.mean(scores)

# Main loop to find optimal number of hidden neurons
best_score = -np.inf
best_neurons = 0
patience_counter = 0

for hidden_neurons in range(1, max_hidden_neurons + 1):
    current_score = train_with_early_stopping(hidden_neurons, X_train, y_train)
    print(f"Hidden neurons: {hidden_neurons}, Validation score: {current_score}")

    if current_score > best_score + delta:
        best_score = current_score
        best_neurons = hidden_neurons
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1

    # Check for early stopping
    if patience_counter >= patience:
        print(f"Early stopping triggered at {hidden_neurons} hidden neurons.")
        break

print(f"Best number of hidden neurons: {best_neurons} with score: {best_score}")