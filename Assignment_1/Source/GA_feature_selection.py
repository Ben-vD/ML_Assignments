import Mutation as mut
import Crossover as cross
import Selection as sel
import UtilFunctions as uf
import Fitness as fit
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from tqdm import tqdm

def evolution(X_train, y_train, X_test, y_test, generations, populationSize, feature_mut_rate,
              hyper_mut_rate, hyper_n, hyperMin, hyperMax, hyper_mut_vars_start, exp):
    
    np.random.seed()


    n_features = X_train.shape[1]

    # Create initial population
    p_feat_chr, p_hyp_chr, p_fitness, p_test_acc_arr = uf.create_initial_pop(X_train, y_train, X_test, y_test, n_features, populationSize, hyper_n, hyperMin, hyperMax)

    best_indx = np.argmax(p_test_acc_arr)
    best_feat_chr = p_feat_chr[best_indx]
    best_hyp_chr = p_hyp_chr[best_indx]
    best_fitness = p_fitness[best_indx]
    best_test_acc = p_test_acc_arr[best_indx]

    # Run Evolution    
    for g in tqdm (range(generations), desc="Exp {}, Generation".format(exp)):
        
        # Generate offspring (Crossover)
        off_feat_chr, off_hyp_chr = cross.gen_offspring_pop(p_feat_chr, p_hyp_chr, p_fitness, populationSize)
        

        # Mutation
        mut.mut_feat(off_feat_chr, uf.lin_interp_mut(feature_mut_rate, g, generations))
        mut.mut_hyp(off_hyp_chr, uf.lin_interp_mut(hyper_mut_rate, g, generations), uf.lin_interp_mut_vars(hyper_mut_vars_start, g, generations))
        
        # Fix invalid offspring and get offspring fitness
        uf.fix_invalid_chr(off_feat_chr, off_hyp_chr, hyperMin, hyperMax)
        off_fitness, off_test_acc_arr = fit.get_pop_fitness(X_train, y_train, X_test, y_test, off_feat_chr, off_hyp_chr, populationSize)
        
        # Combine parents and offspring
        comb_feat_chr = np.concatenate((p_feat_chr, off_feat_chr), axis = 0)
        comb_hyp_chr = np.concatenate((p_hyp_chr, off_hyp_chr), axis = 0)
        comb_fitness = np.concatenate((p_fitness, off_fitness))
        comp_test_acc_arr = np.concatenate((p_test_acc_arr, off_test_acc_arr))

        # Selection
        new_parent_idxs = sel.selection(comb_fitness, 2, populationSize)

        # Update Generation
        p_feat_chr = comb_feat_chr[new_parent_idxs]
        p_hyp_chr = comb_hyp_chr[new_parent_idxs]
        p_fitness = comb_fitness[new_parent_idxs]
        p_test_acc_arr = comp_test_acc_arr[new_parent_idxs]

        # Get new best model
        if (np.max(p_test_acc_arr) > best_test_acc):
            best_indx = np.argmax(p_test_acc_arr)
            best_feat_chr = p_feat_chr[best_indx]
            best_hyp_chr = p_hyp_chr[best_indx]
            best_fitness = p_fitness[best_indx]
            best_test_acc = p_test_acc_arr[best_indx]

        #print(np.mean(p_fitness), best_test_acc, best_fitness)
        #print(best_feat_chr)
        #print(best_hyp_chr)
        #print()

    #print(best_feat_chr)
    #print(best_hyp_chr)
    #print(best_test_acc, best_fitness)

    # Get test accuracy from best model
    features = (best_feat_chr == 1)

    X_test_sel_feats = X_test[:, features]
    X_train_sel_feats = X_train[:, features]

    metric_str = ["euclidean", "manhattan", "minkowski", "cosine"][int(best_hyp_chr[1])]      
    weights_str = ["uniform", "distance"][int(best_hyp_chr[2])]

    k_model = KNeighborsClassifier(n_neighbors = int(best_hyp_chr[0]), metric = metric_str, weights = weights_str)
    k_model.fit(X_train_sel_feats, y_train)
    return [accuracy_score(y_test, k_model.predict(X_test_sel_feats)), best_test_acc, best_fitness, best_feat_chr, best_hyp_chr]  