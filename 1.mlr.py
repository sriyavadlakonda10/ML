import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\Download\Investment.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

X = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor .fit(X_train,y_train)

y_pred = regressor.predict(x_test)

#we build mlr model

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
#ols
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#ols
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1]]
#ols
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(x_test, y_test)
variance

