import pandas
import numpy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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


# Creating and training the multi linear regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predict the test-set results
y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape(len(y_pred), 1)  # make horizental vector vertical
y_test = y_test.reshape(len(y_test), 1)  # make horizental vector vertical

numpy.set_printoptions(precision=2)  # number of decimals to display (calculation will not be affected)
print(
    numpy.concatenate(
        (y_pred, y_test),
        1  # axis=1, it means concatanating the vectors horizentally
    )
)
