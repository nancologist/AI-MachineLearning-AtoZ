import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression


# Importing Dataset
dataset = pandas.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting dataset into training- and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Training the Logistic Regression model on Training Set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# Predict a new result
res = classifier.predict(
    sc.fit_transform([[30, 87000]])  # IMPORTANT: As we have already applied the Feature Scaling on Indep. Vars. above, we should use the same StandardScaler instance also here!
)
print(res)  # returns 0 - it means the Person wouldn't buy the SUV , which is right when we look at CSV file.


# Predict Test Set Result:
y_pred = classifier.predict(X_test)
print(
    numpy.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1),
        ),
        1
    )
)
