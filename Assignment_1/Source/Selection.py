import numpy as np

# Done
# Can select same individual multiple times (Asexual)
def selection(fitness_arr, tourn_size, select_n):
    
    selected_idxs = np.zeros(select_n, dtype = int)
    
    for tourn_nr in range(select_n):
        
        tournament_idxs = np.random.choice(len(fitness_arr), tourn_size, replace = False)
        tournament_fitness = fitness_arr[tournament_idxs]
        winner_idx = tournament_idxs[np.argmax(tournament_fitness)]
    
        selected_idxs[tourn_nr] = winner_idx
    
    return selected_idxs