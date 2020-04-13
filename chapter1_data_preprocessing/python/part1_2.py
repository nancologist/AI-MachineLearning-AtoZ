import numpy # as np (in Tutorial)
import matplotlib.pyplot as pyplot
import pandas
from sklearn.impute import SimpleImputer

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

