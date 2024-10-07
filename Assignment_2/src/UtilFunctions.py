import numpy as np

def oheLabels(y):
    y_ohe = np.zeros((len(y), np.max(y) + 1))
        
    for i in range(len(y)):
        y_ohe[i, y[i]] = 1

    return y_ohe

def random_ints_sum_to_n(n, count):
    nums = np.ones(count, dtype = int)
    remaining = n - count

    random_indices = np.random.randint(0, count, size = remaining)
    for index in random_indices:
        nums[index] += 1

    return nums
    