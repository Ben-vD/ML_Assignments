import GA_feature_selection as ga
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from multiprocessing import Pool

generations = int(sys.argv[1])
populationSize = int(sys.argv[2])
feature_mut_rate = float(sys.argv[3])
hyper_mut_rate = float(sys.argv[4])
hyper_n = int(sys.argv[5])
hyperMin = np.array(sys.argv[6].split(","), dtype = float)
hyperMax = np.array(sys.argv[7].split(","), dtype = float)
hyper_mut_vars_start = np.array(sys.argv[8].split(","), dtype = float)

print("Generation", generations)
print("Population Size", populationSize)
print("Feature Mutation Rate", feature_mut_rate)
print("Hyperparameter Mutation Rate", hyper_mut_rate)
print("Number of Hyperparameters", hyper_n)
print("Hyperparameter Min", hyperMin)
print("Hyperparameter Max", hyperMax)
print("Hyperparameter Vars", hyper_mut_vars_start)

#iris = load_iris()
#iris_data = iris.data
#iris_labels = iris.target
#
#breast_cancer = load_breast_cancer()
#breast_cancer_data = breast_cancer.data
#breast_cancer_labels = breast_cancer.target

data = np.loadtxt("CleanDataNorm.csv", dtype = str, delimiter = ',')
beans_lables = data[1:, 23].astype(int)
beans_data = data[1:, :23].astype(float)

X_train, X_test, y_train, y_test = train_test_split(beans_data, beans_lables, test_size = 0.25, random_state = 42)

results = []
pool = Pool(processes=10)
for exp in range(10):
    result = pool.apply_async(ga.evolution, args = (X_train, y_train, X_test, y_test, generations, populationSize, feature_mut_rate,
                                                    hyper_mut_rate, hyper_n, hyperMin, hyperMax, hyper_mut_vars_start, exp))
    results.append(result)

pool.close()
pool.join()

resF = [r.get() for r in results]

with open('results0_9.txt', 'w') as file:
    for res in resF:
        file.write(f"{res}\n")

#ga.evolution(X_train, y_train, X_test, y_test, generations, populationSize, feature_mut_rate,
#             hyper_mut_rate, hyper_n, hyperMin, hyperMax, hyper_mut_vars_start, 0)