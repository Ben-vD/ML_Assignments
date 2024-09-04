import numpy as np

def mut_feat(feat_chr, m_r):
    
    for chr in feat_chr:
        for i in range(len(chr)):
            if np.random.rand() < m_r:
                chr[i] = 1 - chr[i]
                
def mut_hyp(hyp_chr, hyper_mut_rate, hyper_mut_vars):
    
    hyper_n = hyp_chr.shape[1]

    for chr in hyp_chr:
        mut_hyper_arr = np.random.uniform(size = hyper_n) < hyper_mut_rate

        for i, mut_hyper in enumerate(mut_hyper_arr):
            if (mut_hyper):
                chr[i] += np.random.normal(0, hyper_mut_vars[i])
