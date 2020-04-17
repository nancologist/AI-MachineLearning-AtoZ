# Importing Data +++++++++++++++++++++++++++++++++++++++++++++++++
# First set the Current Directory to Working Directory
dataset <- read.csv('Salary_Data.csv')

# Splitting Data for Training and Test +++++++++++++++++++++++++++
# install.packages('caTools')
library(caTools)
splitter <- sample.split(dataset$Salary, SplitRatio = 0.8)

training_set <- subset(dataset, splitter == TRUE)
test_set <- subset(dataset, splitter == FALSE)

# R-Package will take care of Feature Scaling...