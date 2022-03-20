import pandas
from matplotlib import pyplot
import numpy
from sklearn.ensemble import RandomForestRegressor

dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(X, y)

# Predict a new result:
res = rf_regressor.predict([[6.5]])  # => 167K $ - A Good Prediction!

# Visualize the Random Forest regression results (high resolution)
X_grid = numpy.arange(min(X), max(X), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
pyplot.scatter(X, y, color='red')
pyplot.plot(X_grid, rf_regressor.predict(X_grid), color='blue')
pyplot.title('Truth or Bluff (Random Forest Regression)')
pyplot.xlabel('Position level')
pyplot.ylabel('Salary')
pyplot.show()
