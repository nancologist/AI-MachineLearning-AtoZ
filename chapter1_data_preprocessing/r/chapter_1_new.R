# Data Preprocessing 

# Importing the dataset
dataset <- read.csv('Data.csv')

# Handling Missing Data
dataset$Age <- ifelse(
    is.na(dataset$Age),
    ave(dataset&Age, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Age
)
