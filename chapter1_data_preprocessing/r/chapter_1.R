# Data Preprocessing

# Importing the dataset ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dataset <- read.csv('Data.csv') 
# ERROR : If it's not working check the "getwd()" and then "setwd(PROJECT_DIR)"
# setwd("~/IdeaProjects/AI/Learning_MachineLearning_AtoZ/chapter1_data_preprocessing/r")



# Taking care of missing data ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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



# Encoding categorical data ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dataset$Country <- factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
dataset$Purchased <- factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))



# Splitting the dataset into the Training set and Test set ++++++++++++++++++++++++++++++++++++++++++++++++++
install.packages('caTools')
library(caTools)
set.seed(123) # Just as in the tutorial to get the same result (like random_state in Python)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)

# training_set <- dataset[split,] - my way... (it worked but for safety i use the next line of Tut)
training_set <- subset(dataset, split == TRUE)

test_set <- subset(dataset, split == FALSE)

