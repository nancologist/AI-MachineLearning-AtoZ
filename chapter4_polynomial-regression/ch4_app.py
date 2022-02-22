import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot

dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values


# Train the Linear Regression Model on the whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)


# Train the Polynomial Regression Model:
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)  # X_poly is the matrix of features composed of Position-levels and
# the squares of the Position-levels.
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)


# Visualize the Linear Regression Results:
# pyplot.scatter(X, y, color="red")  # plot the real dataset
# pyplot.plot(X, linear_regressor.predict(X), color="blue")  # Plot linear-regression prediction
# pyplot.title("Truth or Bluff (Linear Regression)")
# pyplot.xlabel("Position Level")
# pyplot.ylabel("Salary")
# pyplot.show()


# Visualize the Polynomial Regression:
pyplot.scatter(X, y, color="red")  # plot the real dataset
pyplot.plot(
    X,
    linear_regressor_2.predict(
        polynomial_regressor.fit_transform(X)
    ),
    color="blue"
)  # Plot linear-regression prediction
pyplot.title("Truth or Bluff (Polynomial Regression degree=2)")
pyplot.xlabel("Position Level")
pyplot.ylabel("Salary")
pyplot.show()
