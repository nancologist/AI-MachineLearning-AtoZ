import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1)
        ),
        1
    )
)


# =============================================================
# ============= EVALUATING THE MODEL PERFORMANCE: =============
# =============================================================
score = r2_score(y_test, y_pred)  # => 0.9325315554761302
