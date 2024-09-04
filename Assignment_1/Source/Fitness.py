from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

def get_pop_fitness(X_train, y_train, X_test, y_test, feat_chrs, hyp_chrs, pop_n):

    fitness_arr = np.zeros(pop_n)
    test_acc_arr = np.zeros(pop_n)
    
    for i in range(pop_n):
            features = feat_chrs[i,:] == 1
            fitness_arr[i], test_acc_arr[i] = fitness(X_train[:, features], y_train, X_test[:, features], y_test, hyp_chrs[i, 0], hyp_chrs[i, 1], hyp_chrs[i, 2])
    
    return fitness_arr, test_acc_arr
    

def fitness(X_train, y_train, X_test, y_test, neighbours, metric, weights):
    
    metric_str = ["euclidean", "manhattan", "cosine"][int(metric)]
    weights_str = ["uniform", "distance"][int(weights)]

    k_model = KNeighborsClassifier(n_neighbors = int(neighbours), metric = metric_str, weights = weights_str)
    k_model.fit(X_train, y_train)
    return np.mean(cross_val_score(k_model, X_train, y_train, cv = 10)), accuracy_score(y_test, k_model.predict(X_test))