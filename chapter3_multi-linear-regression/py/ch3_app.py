import pandas
import numpy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Import data and set x (indep. variables) and y (dependent variable):
data_set = pandas.read_csv('50_Startups.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

# Encode categorical variable (State which is at Column=3) one-hot:
colTransformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [3])
    ],
    remainder='passthrough'
)
X = numpy.array(
    colTransformer.fit_transform(X)
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
