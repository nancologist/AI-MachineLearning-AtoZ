# Importing Libraries
import numpy
import matplotlib.pyplot
import pandas
from sklearn.model_selection import train_test_split

# Importing Dataset
dataset = pandas.read_csv('PATH_TO_FILE/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Dataset For Training and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
