import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

dataSet = pd.read_csv('Regression/Multiple Linear Regression/50_Startups.csv')
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(XTrain, yTrain)

yPred = regressor.predict(XTest)
np.set_printoptions(precision=2)
print(np.concatenate((yPred.reshape(len(yPred), 1), yTest.reshape(len(yTest),1)), 1))
