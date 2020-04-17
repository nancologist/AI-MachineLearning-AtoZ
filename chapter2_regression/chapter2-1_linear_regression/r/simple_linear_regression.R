# Importing Data +++++++++++++++++++++++++++++++++++++++++++++++++
# First set the Current Directory to Working Directory
dataset <- read.csv('Salary_Data.csv')

# Splitting Data for Training and Test +++++++++++++++++++++++++++
# install.packages('caTools')
library(caTools)
splitter <- sample.split(dataset$Salary, SplitRatio = 0.8)

training_set <- subset(dataset, splitter == TRUE)
test_set <- subset(dataset, splitter == FALSE)

# R-Package will take care of Feature Scaling


# Fit Simple Linear Regression to Training Set ++++++++++++++++++++
linear_regressor <- lm(formula = Salary ~ YearsExperience, data = training_set)
summary(linear_regressor)


# Predicting ++++++++++++++++++++++++++++++++++++++++++++++++++++++
y_pred <- predict(linear_regressor, newdata = test_set)


# Visaulising Data and Prediction +++++++++++++++++++++++++++++++++
# install.packages('ggplots2')
library(ggplot2)

# Training-Set Plot +++++++++++++++++++++++++
ggplot() +
    
    geom_point( # Scattered Points of Observations in Training Set
        aes(
            x = training_set$YearsExperience,
            y = training_set$Salary
        ),
        colour = 'red'
    ) +
    
    geom_line( # Linear Regression Line
        aes(
            x = training_set$YearsExperience,
            y = predict(linear_regressor, newdata = training_set)
        ),
        colour = 'blue'
    ) +
    
    ggtitle('Salary & XP (Training Set)') +
    
    xlab('Years of Experience') +
    
    ylab('Salary')


# Test-Set Plot +++++++++++++++++++++++++
ggplot() +
    
    geom_point( # Scattered Points of Observations in Training Set
        aes(
            x = test_set$YearsExperience,
            y = test_set$Salary
        ),
        colour = 'red'
    ) +
    
    geom_line( # Linear Regression Line
        aes(
            x = training_set$YearsExperience,
            y = predict(linear_regressor, newdata = training_set)
        ),
        colour = 'blue'
    ) +
    
    ggtitle('Salary & XP (Test Set)') +
    
    xlab('Years of Experience') +
    
    ylab('Salary')




