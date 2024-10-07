from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

#cancer = datasets.load_breast_cancer()
#X = cancer.data
#y = cancer.target

#wine = datasets.load_wine()
#X = wine.data
#y = wine.target

digits = datasets.load_digits()
X = digits.data
y = digits.target

feature_count = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2)

print(feature_count)
for i in range(1, feature_count + 1):
    rf = RandomForestClassifier(100, max_depth = None, max_features = i)
    print(i, np.mean(cross_val_score(rf, X, y, cv = 10, n_jobs = 10)))