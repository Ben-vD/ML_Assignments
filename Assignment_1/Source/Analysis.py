import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

data = pd.read_csv("CleanDataDT.csv")
beans_lables = data["Class"].to_numpy()
beans_data = data.drop(["Class"], axis = 1).to_numpy()

iris = load_iris()
iris_data = iris.data
iris_labels = iris.target

breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_labels = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(beans_data, beans_lables, test_size = 0.25)

clf = tree.DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()