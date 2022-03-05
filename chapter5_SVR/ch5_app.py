import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot
import numpy


# Import dataset
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # [[ 1], [ 2], [ 3], [ 4], [ 5], [ 6], [ 7], [ 8], [ 9], [10]]
y = dataset.iloc[:, -1].values  # [  45000   50000   60000   80000  110000  150000  200000  300000  500000, 1000000]

# fit_transform() needs a 2D array so we should make y vertical (i.e. 10 rows in 1 column):
y = y.reshape(len(y), 1)  # [[  45000], [  50000], [  60000], [  80000], [ 110000], [ 150000], [ 200000], [ 300000], [ 500000], [1000000]]


# Apply feature-scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)  # [[-1.5666989 ], [-1.21854359], [-0.87038828], [-0.52223297], [-0.17407766], [ 0.17407766], [ 0.52223297], [ 0.87038828], [ 1.21854359], [ 1.5666989 ]]
y = sc_y.fit_transform(y)  # [[-0.72004253], [-0.70243757], [-0.66722767], [-0.59680786], [-0.49117815], [-0.35033854], [-0.17428902], [ 0.17781001], [ 0.88200808], [ 2.64250325]]


# Train the SVR model on the whole dataset:
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X, y)


# Predict a new result:
x_target = [[6.5]]
x_target = sc_x.transform(x_target)  # As we have already applied the feature-scaling on our X, so we should apply it here too.
y_predict = svr_regressor.predict(x_target).reshape(-1, 1)
y_predict = sc_y.inverse_transform(y_predict)  # [[170370.0204065]] => 170K $


# Visualize the SVR results:
pyplot.scatter(
    sc_x.inverse_transform(X),
    sc_y.inverse_transform(y),
    color="red"
)  # Plot scattered dataset

pyplot.plot(
    sc_x.inverse_transform(X),
    sc_y.inverse_transform(
        svr_regressor.predict(X).reshape(-1, 1)
    ),
    color="blue"
)  # Plot SVR-regression prediction

pyplot.title("Truth or Bluff (Support Vector Regression)")
pyplot.xlabel("Position Level")
pyplot.ylabel("Salary")
pyplot.show()


# Visualise the SVR results (for high resolution and smoother curve)
X_grid = numpy.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
pyplot.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
pyplot.plot(X_grid, svr_regressor.predict(sc_x.inverse_transform(X_grid)), color='blue')
pyplot.title('Truth or Bluff (SVR)')
pyplot.xlabel('Position level')
pyplot.ylabel('Salary')
# pyplot.show()
