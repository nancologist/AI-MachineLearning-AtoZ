# Data Preprocessing 

# Importing the dataset ++++++++++++++++++++++++++++++++++++++
dataset <- read.csv('Data.csv')
# dataset <- dataset[, 2:3]

# Splitting Dataset ++++++++++++++++++++++++++++++++++++++++
# install.packages('caTools')
library(caTools)
set.seed(123)
splitter <- sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set <- subset(dataset, splitter == TRUE)
test_set <- subset(dataset, splitter == FALSE)

# Feature Scaling +++++++++++++++++++++++++++++++++++++++++
# training_set[, 2:3] <- scale(training_set[, 2:3])
# test_set[, 2:3] <- scale(test_set[, 2:3])