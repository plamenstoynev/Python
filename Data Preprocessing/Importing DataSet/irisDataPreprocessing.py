# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Loading the Iris dataset
dataset = pd.read_csv('/Users/plamenstoynev/OneDrive/Python/Data Preprocessing/iris.csv')

# Creating the matrix of features (X) and the dependent variable vector (y)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Printing the matrix of features and the dependent variable vector
print(y)
print(x)