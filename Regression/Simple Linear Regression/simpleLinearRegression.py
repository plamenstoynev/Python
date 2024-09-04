import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataSet = pd.read_csv("../../Simple Linear Regression/simpleLinearRegression.py")
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size= 0.2, random_state= 0)

regressor = LinearRegression()
regressor.fit(XTrain, yTrain)