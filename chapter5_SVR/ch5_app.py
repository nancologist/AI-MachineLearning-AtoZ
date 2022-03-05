import pandas
from sklearn.preprocessing import StandardScaler


# Import dataset
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # [[ 1], [ 2], [ 3], [ 4], [ 5], [ 6], [ 7], [ 8], [ 9], [10]]
y = dataset.iloc[:, -1].values  # [  45000   50000   60000   80000  110000  150000  200000  300000  500000, 1000000]

# fit_transform() needs a 2D array so we should make y vertical (i.e. 10 rows in 1 column):
y = y.reshape(len(y), 1)  # [[  45000], [  50000], [  60000], [  80000], [ 110000], [ 150000], [ 200000], [ 300000], [ 500000], [1000000]]


# Apply feature-scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)  # [[-1.5666989 ], [-1.21854359], [-0.87038828], [-0.52223297], [-0.17407766], [ 0.17407766], [ 0.52223297], [ 0.87038828], [ 1.21854359], [ 1.5666989 ]]
y = sc_y.fit_transform(y)  # [[-0.72004253], [-0.70243757], [-0.66722767], [-0.59680786], [-0.49117815], [-0.35033854], [-0.17428902], [ 0.17781001], [ 0.88200808], [ 2.64250325]]
