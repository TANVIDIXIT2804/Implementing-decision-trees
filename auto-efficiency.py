
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read auto-mpg data set
data = pd.read_csv("/content/auto-mpg .csv")

# replacing hp values containing ? with mean value of hps
data.hp = data.hp.str.replace('?', 'NaN').astype(float)
data.hp.fillna(data.hp.mean(), inplace=True)

d_hp = data.hp
data.hp = d_hp.astype(int)

#carname col of no use -> so delete
data = data.drop('carname', axis=1) 

#rename col
s,t = data.shape
for i, attr in enumerate(data):
    if i == 0:                # first iteration, no renaming reqd -> so continue
        continue
    else:
        data.rename(columns={str(attr):i-1}, inplace=True)

# Data preprocessing
random = data.sample(frac=1).reset_index(drop=True) #shuffling data
X = (random.iloc[:, 1:]).squeeze()
y = (random.iloc[:, 0:1]).T.squeeze()

# Split data for training and testing
set_split = int(0.7*len(y))
X_train=pd.DataFrame(X.iloc[:set_split])
X_test=pd.DataFrame(X.iloc[set_split:])
y_train=pd.Series(y[:set_split])
y_test=pd.Series(y[set_split:])

max_depth = 5 

# training on created Decision Tree:
tree_own = DecisionTree(max_depth=max_depth)
tree_own.fit(X_train, y_train)
tree_own.plot()

y_hat_1 = pd.Series(tree_own.predict(X_test))
y_test_1 = y_test.reset_index(drop=True)

RMSE_own_dt = rmse(y_hat_1, y_test_1)
MAE_own_dt = mae(y_hat_1, y_test_1)

print('RMSE for MyTree: {}'.format(RMSE_own_dt))
print('MAE for MyTree: {}'.format(MAE_own_dt))

# Training on SkLearn decision tree
tree_skl = DecisionTreeRegressor(max_depth=max_depth)
tree_skl.fit(X_train, y_train)

y_hat = pd.Series(tree_skl.predict(X_test))
y_test_ = y_test.reset_index(drop=True)

RMSE_skl = rmse(y_hat, y_test_)
MAE_skl = mae(y_hat, y_test_)

print('RMSE for SklTree: {}'.format(RMSE_skl))
print('MAE for SklTree: {}'.format(MAE_skl))

#final comparison
if RMSE_own_dt <= RMSE_skl:
    print("RMSE for my tree is lesse than that of the Skltree")
else:
    print("RMSE for SkTree is lesser than that of MyTree")
if MAE_own_dt <= MAE_skl:
    print("MAE for my tree is lesser than that of SklTree")
else:
    print("MAE for SkTree is lesser than that of MyTree")
