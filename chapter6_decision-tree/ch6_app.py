import pandas
import matplotlib.pyplot
import numpy
from sklearn.tree import DecisionTreeRegressor

dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Train the decision-tree regression model:
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X, y)
