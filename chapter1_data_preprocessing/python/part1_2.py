import numpy
import matplotlib.pyplot as pyplot
import pandas

# Importing Data and store it as a Data Frame:
dataset = pandas.read_csv('./data/Data.csv')

# Features (Independent Variables) - conventionally the var name is X.
X = dataset.iloc[:, :-1].values

# The Prediction (Dependent Variable) going to be compared with this (the last column of dataset):
y = dataset.iloc[:,-1].values