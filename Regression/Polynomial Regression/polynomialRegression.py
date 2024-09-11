import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataSet = pd.read_csv('/Users/plamenstoynev/OneDrive/Python/Python/Regression/Polynomial Regression/Position_Salaries.csv')
X = dataSet.iloc[:, 1:-1].values
y = dataSet.iloc[:, -1].values

linReg = LinearRegression()
linReg.fit(X,y)

polynamialReg = PolynomialFeatures(degree=4)
polynamialReg.fit_transform(X)
xPolyReg = polynamialReg.fit_transform(X)
linReg2 = LinearRegression()

linReg2.fit(xPolyReg, y)

plt.scatter(X,y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.title('Truth of Bluff (Linear Regression)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,y, color = 'red')
plt.plot(X, linReg2.predict(polynamialReg.fit_transform(X)), color = 'blue')
plt.title('Truth of Bluff (Polynomial Regression)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

XGrid = np.arange(min(X), max(X), 0.1)
XGrid = XGrid.reshape((len(XGrid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(XGrid, linReg2.predict(polynamialReg.fit_transform(X)), color = 'blue')
plt.title('Truth of Bluff (Polynomial Regression)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

print(linReg.predict([[6.5]]))

print(linReg2.predict(polynamialReg.fit_transform([[6.5]])))