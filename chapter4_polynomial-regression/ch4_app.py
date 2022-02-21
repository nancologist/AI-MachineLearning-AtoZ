import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values


# Train the Linear Regression Model on the whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)


# Train the Polynomial Regression Model:
polynomial_regressor = PolynomialFeatures(degree=2)
X_poly = polynomial_regressor.fit_transform(X)  # X_poly is the matrix of features composed of Position-levels and
# the squares of the Position-levels.
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

