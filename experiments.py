
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions
P=5

# create a function to generate fake data
def fakeData(N,M,io):

    if io == "DIRO":
        d = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        X = d.reset_index()  # to avoid indexing errors
        y = pd.Series(np.random.randn(N))

    elif io == "RIDO":
        d = pd.DataFrame(np.random.randn(N, M))
        X = d.reset_index()
        y = pd.Series(np.random.randint(P, size=N), dtype="category")

    elif io == "DIDO":
        d = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        X = d.reset_index()
        y = pd.Series(np.random.randint(P, size=N), dtype="category")

    else:
        d = pd.DataFrame(np.random.randn(N, M))
        X = d.reset_index()
        y = pd.Series(np.random.randn(N))

    return X, y
    
# function to measure the time for learning and predicting
def computeTime(N,M,io):
    
    
    learn_list=[]
    predict_list=[]
    
    for i in range(1,N+1):
      learning_time = []
      predicting_time = []
      features=[]
      for j in range(1, M+1):

          X,y = fakeData(i,j,io)
          tree = DecisionTree()

          # computing time for learning
          start = time.time()
          tree = DecisionTree(criterion="information_gain")
          tree.fit(X, y)
          finish = time.time()
          learning_time.append((finish-start))
          #print(learning_time)

          # computing time for predicting
          start = time.time()
          y_hat = tree.predict(X)
          finish = time.time()
          predicting_time.append((finish - start))

          features.append(j)
      learn_list.append(learning_time)
      predict_list.append(predicting_time)

    #print(learn_list)
   
    # Plotting the graph
    for lt in range(1,len(learn_list)+1):
      plt.plot(features, learn_list[lt-1], label='sample %s' %lt,marker='.')
    plt.title("Number of features vs time taken to learn")
    plt.xlabel("Number of features")
    plt.ylabel("Time taken") 
    plt.legend()
    plt.show()

    for lt in range(1,len(predict_list)+1):
      plt.plot(features, predict_list[lt-1], label='sample %s' %lt,marker='.')
    plt.title("Number of features vs time taken to predict")
    plt.xlabel("Number of features")
    plt.ylabel("Time taken") 
    plt.legend()
    plt.show()

if __name__ =='__main__':
    computeTime( 3,20, "DIRO") #on changing the io to other forms we can get the graphs for dido rido and riro
