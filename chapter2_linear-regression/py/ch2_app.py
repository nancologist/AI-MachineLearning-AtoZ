import pandas
from sklearn.model_selection import train_test_split

dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
