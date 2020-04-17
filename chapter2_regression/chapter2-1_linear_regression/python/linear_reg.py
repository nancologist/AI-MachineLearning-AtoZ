# Importing Libraries +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing Dataset ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dataset = pandas.read_csv('./data/Salary_Data.csv')
X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1].values

# Splitting Dataset For Training and Test ++++++++++++++++++++++++++++++++++++++++++++
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Model ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predicting using Test Set +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
y_pred = linear_regressor.predict(X_test)

# Visualising Data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Training Set
pyplot.scatter(X_train, y_train, color='red')
pyplot.plot(X_train, linear_regressor.predict(X_train), color='blue')
pyplot.title('Salary & XP (Training Set)')
pyplot.xlabel('Years of Experience')
pyplot.ylabel('Salary')
pyplot.show()

# Test Set
pyplot.scatter(X_test, y_test, color='red')
pyplot.plot(X_train, linear_regressor.predict(X_train), color='blue')
pyplot.title('Salary & XP (Test Set)')
pyplot.xlabel('Years of Experience')
pyplot.ylabel('Salary')
pyplot.show()