import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualising the Training-Set results
pyplot.scatter(X_train, y_train, color='red')  # plot the datasets as points
pyplot.plot(X_train, regressor.predict(X_train), color='blue')  # plot the predicted linear regression
pyplot.title('Salary vs. XP (Training Set)')
pyplot.xlabel('Years of XP')
pyplot.ylabel('Salary')
pyplot.show()

# Visualising the Test-Set results
pyplot.scatter(X_test, y_test, color='red')
pyplot.plot(X_train, regressor.predict(X_train), color='blue')
pyplot.title('Salary vs. XP (Test Set)')
pyplot.xlabel('Years of XP')
pyplot.ylabel('Salary')
pyplot.show()
