import pandas
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
