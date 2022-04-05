import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

df = pd.read_csv('gpu_running_time.csv')

data = df.to_numpy()
X = data[:,:15]
y = np.mean(data[:,15:],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4361)

mse, mae = 0, 0 
e = []
  
for i in range (100): 
  #builds 100 trees
  model = DecisionTreeRegressor(splitter= "random")
  model.fit(X_train,y_train)
  pred = model.predict(X_test)

  mse += mean_squared_error(pred,y_test)
  mae += mean_absolute_error(pred,y_test)

  e.append(model)
  
print("mse: ", mse/len(e))
print("mae: ", mae/len(e))