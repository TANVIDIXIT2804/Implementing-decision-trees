import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read dataset
# ...
# 

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

#for 2a splitting 70% data to train and test on remaining
X_train = pd.DataFrame(X[0:70,:])
y_train = pd.Series(y[0:70], dtype = "category")

X_test = pd.DataFrame(X[70:100,:])
y_test = pd.Series(y[70:100], dtype = "category")
y_test_uni=y_test.unique()


for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for data in y_test_uni:
        print('Precision: ', precision(y_hat, y_test, data))
        print('Recall: ', recall(y_hat, y_test, data))
        
#2b using kfold cross-validation for ig and gi

from sklearn.model_selection import KFold

def computeOptDepth(X, y, folds = 5, depths = [1,2,3,4]):
    kf = KFold(n_splits = 5)
    kf_copy = kf #preserving the data for nested cv
    kf.get_n_splits(X) #Returns the number of splitting iterations in the cross-validator
    avg_accu = {"information_gain":0, "gini_index":0}
    for train_id, test_id in kf.split(X):
        X_train = pd.DataFrame(X[train_id])
        y_train = pd.Series(y[train_id], dtype = "category")
        X_test = pd.DataFrame(X[test_id])
        y_test = pd.Series(y[test_id], dtype = "category")

        for x in ['information_gain', 'gini_index']:
            tree = DecisionTree(criterion=x) #Split based on Inf. Gain and gi
            tree.fit(X_train, y_train)
            y_hat = tree.predict(X_test)
            #tree.plot()
            print('Criteria :', x)
            print('Accuracy: ', accuracy(y_hat, y_test))
            avg_accu[x] += accuracy(y_hat, y_test)

    for x in ['information_gain', 'gini_index']:
        avg_accu[x] /= folds
    print("KFold avg accuracy on info_gain: {:.2f} and on gini_index = {:.2f}".format(avg_accu["information_gain"], avg_accu["gini_index"]))
    
    #nested cross validation 
    no_of_folds = 0
    kf = kf_copy
    kf.get_n_splits(X)
    for train_id, test_id in kf.split(X):
        X_train = X[train_id]
        y_train = y[train_id]
        opt_accu = {"information_gain":0, "gini_index":0}
        opt_depth = {"information_gain":0, "gini_index":0}
        for d in depths:
            kf_val = KFold(n_splits = folds)
            kf_val.get_n_splits(X_train)
            curr_avg_accu = {"information_gain":0, "gini_index":0}
            for train_nest, test_nest in kf_val.split(X_train):
                X_train_nest = pd.DataFrame(X_train[train_nest])
                y_train_nest = pd.Series(y_train[train_nest], dtype = "category")
                X_val = pd.DataFrame(X_train[test_nest])
                y_val = pd.Series(y_train[test_nest], dtype = "category")
                for x in ['information_gain', 'gini_index']:
                    tree = DecisionTree(criterion=x) #Split based on Inf. Gain
                    tree.fit(X_train_nest, y_train_nest)
#                     X_val = pd.DataFrame(X_val)
                    y_hat = tree.predict(X_val)
#                     tree.plot()
                    curr_avg_accu[x] += accuracy(y_hat, y_val)
            for x in ['information_gain', 'gini_index']:
                curr_avg_accu[x] /= folds
                if curr_avg_accu[x] > opt_accu[x]:
                    opt_accu = curr_avg_accu
                    opt_depth[x] = d
        
        no_of_folds += 1 #finding the optimal depth as we increase fold till 5
        print("Optimal Depth for fold {} using info_gain = {}".format(no_of_folds,opt_depth["information_gain"]))
        print("Optimal Depth for fold {} using gini_index = {}".format(no_of_folds, opt_depth["gini_index"]))

            

computeOptDepth(X, y, folds = 5, depths = list(range(0, 10))) # calling the func to print most optimal depth using 5 fold nested cross validation

