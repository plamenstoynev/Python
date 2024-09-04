import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/Users/plamenstoynev/OneDrive/Python/Python/Data Preprocessing/Importing DataSet/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
XTrain [:, 3:] = sc.fit_transform(XTrain[:, 3:])
XTest [:, 3:] = sc.transform(XTest[:, 3:])

print ("X Train")
print (XTrain)

print ("X Test")
print (XTest)