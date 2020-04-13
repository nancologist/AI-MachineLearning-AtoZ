import numpy # as np (in Tutorial)
import matplotlib.pyplot as pyplot
import pandas
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Importing Data and store it as a Data Frame:
dataset = pandas.read_csv('./data/Data.csv')

# Features (Independent Variables) - conventionally the var name is X.
X = dataset.iloc[:, :-1].values

# The Prediction (Dependent Variable) going to be compared with this (the last column of dataset):
y = dataset.iloc[:,-1].values

# Taking Care of Missing Data
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
# numpy.nan: Empty Values in Dataset
# strategy='mean' : Replace the missing_values with the average

imputer.fit(X=X[:, 1:3])
# This looks for all the missing_values
# [1:3] -> Becaue the first columnet X[0] is string data type and it can cause error if we pass it to the .fit()

X[:, 1:3] = imputer.transform(X=X[:, 1:3])
# .transform do the REPLACING process of missing_values

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# This function makes a random splitting with the given test_size
# Attention : Just for the purpose of getting the same result we are going to set the "random_state" , otherwise we shouldn't do that.
