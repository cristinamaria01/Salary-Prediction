import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def line(x, a, b):
    return a*x + b

dataframe = pd.read_csv('Salary.csv', delimiter=',')
print('Dataframe info:')
dataframe.info()

# YearsExperience
x = dataframe['YearsExperience'].values                 # ndarray - N dimensional array
x = x.reshape(-1,1)                                     # 2 dimensional array -> 35, 1

# Salary
y = dataframe['Salary'].values                          # 1 dimensional array -> 35,

# Normalizarea datelor
# y_mean = y.mean()
# y_std  = y.std()                                        # Standard Deviation
# y      = (y - y_mean)/y_std


# Impartirea datelor
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size= 0.2)
# Overfitting !!! - Impartirea datelor nu este buna - Regularizare ridge regression
# # X_train, X_test = X[:2], X[2:]
# # y_train, y_test = y[:2], y[2:]

# Plotam datele
plt.scatter(x_train, y_train, c= 'orange', label = 'Date de antrenare')
plt.scatter(x_test, y_test, c = 'c', label = 'Date de testare')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# Correlation Matrix
correlation_matrix = dataframe.corr()
print(correlation_matrix)
# sns.heatmap(correlation_matrix, annot=True)
# plt.show()

model = LinearRegression()
model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred = model.predict(x_test)

# ridge_model = Ridge(alpha= 0.1)
# ridge_model.fit(x_train, y_train)
# y_pred_train_r = ridge_model.predict(X_train)
# y_pred_r = ridge_model.predict(X_test)

# Scores
print("R square for train {}".format(model.score(x_train, y_train)))
print("MSE for train: {}".format(mean_squared_error(y_train, y_pred_train)))
print("R squared for test: {}".format(model.score(x_test, y_test)))
print("MSE for test: {}".format(mean_squared_error(y_test, y_pred)))
#
# # print("R square for train Ridge {}".format(ridge_model.score(X_train, y_train)))
# # print("MSE for train Ridge: {}".format(mean_squared_error(y_train, y_pred_train_r)))
# # print("R squared for test Ridge: {}".format(ridge_model.score(X_test, y_test)))
# # print("MSE for test Ridge: {}".format(mean_squared_error(y_test, y_pred_r)))

a = model.coef_[0]
b = model.intercept_
#
# a_ridge = ridge_model.coef_[0]
# b_ridge = ridge_model.intercept_
#
points = np.array([x.min(), x.max()])
plt.plot(points, line(points, a, b), c='red')
plt.legend()
plt.show()

# Daca lucram cu date normalizate
# y_predicted = model.predict(x_test)
# y_predicted = y_predicted * y_std

f = lambda x : a*x + b
print(dataframe)
