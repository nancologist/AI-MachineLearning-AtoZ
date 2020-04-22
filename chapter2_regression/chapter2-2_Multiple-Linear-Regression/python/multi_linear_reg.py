import pandas
import numpy
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Import Data ++++++++++++++++++++++++++++++++++++++++++++++++++++
dataset = pandas.read_csv('./data/50_StartUps.csv')

# Separate Dep. (y) and Indep. (X) Variables +++++++++++++++++++++
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode Categorical Data ++++++++++++++++++++++++++++++++++++++++
transformer = ('encoder', OneHotEncoder(), [3])  # X[-1] : The State of StartUps
colTransformer = ColumnTransformer([transformer], remainder='passthrough')

X = numpy.array(colTransformer.fit_transform(X))

# Splitting Data Sets for Training and Test ++++++++++++++++++++++++
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
