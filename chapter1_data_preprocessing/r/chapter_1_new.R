# Data Preprocessing 

# Importing the dataset ++++++++++++++++++++++++++++++++++++++
dataset <- read.csv('Data.csv')


# Handling Missing Data ++++++++++++++++++++++++++++++++++++++
dataset$Age <- ifelse(
    is.na(dataset$Age),
    ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Age
)

dataset$Salary <- ifelse(
    is.na(dataset$Salary),
    ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Salary
)


# Encoding Categorical Data ++++++++++++++++++++++++++++++++++
dataset$Country = factor(
    dataset$Country,
    levels = c('France', 'Spain', 'Germany'),
    labels = c(1, 2, 3)
)

dataset$Purchased = factor(
    dataset$Purchased,
    levels = c('No', 'Yes'),
    labels = c(0, 1)
)

# One-Hot-Encode
#for(unique_value in unique(dataset$Country)){
#    dataset[paste("Country", unique_value, sep = ".")] <- ifelse(dataset$Country == unique_value, 1, 0)
#}


# Splitting Dataset ++++++++++++++++++++++++++++++++++++++++
# install.packages('caTools')
library(caTools)
set.seed(123) # To get the same result as in the Course.

splitter <- sample.split(dataset$Purchased, SplitRatio = 0.8)

training_set <- subset(dataset, splitter == TRUE)
test_set <- subset(dataset, splitter == FALSE)

# Feature Scaling +++++++++++++++++++++++++++++++++++++++++

# Mori: "I personally would like to run this part before
#       Data Splitting, so that we already have Scaled Features
#       For training_set and test_set! (Save one line code!)

training_set[, 2:3] <- scale(training_set[, 2:3])
test_set[, 2:3] <- scale(test_set[, 2:3])
# The columns Country and Purchased are excluded!


