
generations=200
populationSize=100
feature_mut_rate_start=1.0
hyper_mut_rate_start=1.0
n_hyperP=3
# KNN (n_neighbors,metric,weight)
hyperMin=1,0,0
hyperMax=350,3,2
hyper_mut_vars_start=25,1.5,1.0


python Main.py $generations $populationSize $feature_mut_rate_start $hyper_mut_rate_start $n_hyperP $hyperMin $hyperMax $hyper_mut_vars_start
