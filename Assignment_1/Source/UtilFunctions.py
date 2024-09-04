import numpy as np
import Fitness as fit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def lin_interp_mut_vars(hyper_mut_vars, gen, gen_end):

    lin_interp_mut_vars_arr = np.zeros(len(hyper_mut_vars))
    for i, hyper_start in enumerate(hyper_mut_vars):
        lin_interp_mut_vars_arr[i] = hyper_start + ((gen - 0) / (gen_end - 0)) * (0 - hyper_start)

    return lin_interp_mut_vars_arr

def lin_interp_mut(mut_r_start, gen, gen_end):
    return mut_r_start + ((gen - 0) / (gen_end - 0)) * (0 - mut_r_start)


def fix_invalid_chr(feat_chrs, hyp_chrs, hyperMin, hyperMax):

    populationSize, n_features = feat_chrs.shape
    n_hypers = hyp_chrs.shape[1]

    for i in range(populationSize):
        
        # Fix no feature selected
        if ((1 in feat_chrs[i]) == False):
            rand_feature = np.random.randint(0, n_features)
            feat_chrs[i, rand_feature] = 1
            
        # Fix Range issues ofr hyperparameters
        for j in range(n_hypers):
            if (hyp_chrs[i, j] < hyperMin[j]):
                hyp_chrs[i, j] = hyperMin[j]
            elif (hyp_chrs[i, j] > hyperMax[j]):
                hyp_chrs[i, j] = hyperMax[j] - 1


def create_initial_pop(X_train, y_train, X_test, y_test, n_features, populationSize, hyper_n, hyperMin, hyperMax):
    
    hyp_chrs = np.random.uniform(hyperMin[0], hyperMax[0], populationSize)
    for i in range(1, hyper_n):
        hyp_chrs = np.column_stack((hyp_chrs, np.random.uniform(hyperMin[i], hyperMax[i], populationSize)))
    
    feat_chrs = np.random.randint(0, 2, (populationSize, n_features))

    fix_invalid_chr(feat_chrs, hyp_chrs, hyperMin, hyperMax)
    
    fitness_arr, test_acc_arr = fit.get_pop_fitness(X_train, y_train, X_test, y_test, feat_chrs, hyp_chrs, populationSize)

    return feat_chrs, hyp_chrs, fitness_arr, test_acc_arr