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


# Making a single prediction
# (for example the profit of a startup
# with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(
    regressor.predict(
        [
            [1, 0, 0, 160000, 130000, 300000]
        ]
    )
)  # => [181566.92]
'''
Therefore, our model predicts that the profit of a Californian startup which spent 160000 in R&D,
130000 in Administration and 300000 in Marketing is $ 181566,92.
'''

'''
Important note 1:
Notice that the values of the features were all input in a double pair of square brackets.
That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into
a double pair of square brackets makes the input exactly a 2D array. Simply put:

1,0,0,160000,130000,300000 → scalars

[1,0,0,160000,130000,300000] → 1D array

[[1,0,0,160000,130000,300000]] → 2D array

Important note 2:
Notice also that the "California" state was not input as a string in the last column but as "1, 0, 0" 
in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, 
and as we see in the second row of the matrix of features X, "California" was encoded as "1, 0, 0". And be careful to 
include these values in the first three columns, not the last three ones, because the dummy variables are always created 
in the first columns.
'''

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)  # [ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
print(regressor.intercept_)  # 42467.52924853204


'''
Therefore, the equation of our multiple linear regression model is:

Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3+0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object.
Attributes in Python are different than methods and usually return a simple value or an array of values.
'''