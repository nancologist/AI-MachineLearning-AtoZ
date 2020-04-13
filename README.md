# Machine Learning A-Z: Hands-On Python & R In Data Science

# 1. Welcome to the course
## 1.3. Why Machin Learning is the future?
* For the Dawn of Time until 2005 the human race have created 130 EXABYTES!
* until 2010 that number was : 1,200 EXABYTES
* until 2015 : 7,900 EXABYTES
* estimated for 2020 : 40,900 EXABYTES

Maching Learning can help to use this huge Data more and better.
____

## 1.4. Installing R and R Studio
1. https://cran.r-project.org/

2. Download and Install R

3. www.rstudio.com (IDE for R)

4. Download and install RStudio
___

## 1.5. Meet your instructors
Hi there,

Hope you are enjoying the course so far!

Not so long ago Hadelin and I did an interview on the SDS Podcast. This is the best place to start if you would like to learn more about his background... and a bit about me too if this is your first course with me :)

Link: 

http://www.superdatascience.com/2

Some of the things you will learn in this podcast:
What is Machine Learning
Mastering Data Science through online courses
What are Recommender Systems
Million dollar question: R vs Python (vs Julia)
What Grand project Hadelin and I are currently working on
Plus you will get an overview of:
Regressions
Classifications
Clustering
Association rule learning
Reinforcement learning
Deep learning


See you in class!

Sincerely,

Kirill Eremenko
___
___
___

# 1. Data Preprocessing
## 1.1. Module Introduction
In each section we first start with Python and then do it with the R.
___

## 1.2. Importing the Libraries (Python)
For Machine Learning in Python we always need at least these three libraries:

* __Numpy__ : To work with arrays and do mathematical operations
* __matplotlib.pyplot__ : Which allows you to create charts
* __pandas__ : which allows you to import dataset and create easily matrices and vectors.

### Installation
So we should install these libraries which we need into our Virtual Env.:
* ``pip install numpy``
* ``pip install matplotlib``
* ``pip install pandas``
___
## 1.3. Importing Dataset (Python)
Data.csv : This file here is like a Retail Company which analysis "Which client purchased one of their products, so these the rows (observations) i this dataset correspond to the different customers of this employee, and their infos... and the last column is about if they bought the Product or not."

``pandas.read_csv()``: This creates a Data Frame from a .csv file.

### Next Steps
After importing Dataset and store it as a Data Frame we need to do these as the next steps:
1. Creating Matrix of Features.
2. Dependent Variable, a Vector.

### Important Principle in Machine Learning
_IN ANY DATASET WHICH YOU ARE GOING TO TRAIN A MACHINE LEARNING MODEL YOU HAVE THE ENTITIES WHICH ARE THE FEATURES AND THE DEPENDENT VARIABLE (two above steps)._

### Features (Independent Variables)
Features are the columns, which are the independent informations (here: Country, Age, Salary), with them you are going to PREDICT the DEPENDENT VARIABLE [Mori: For the future!].

``.ilock`` : To locate indexes  
``.iloc[:,:-1]`` : This takes all the Rows and also all the columns of the dataset, EXCEPT the last column (which is going to be predicted)

### Dependent Variable
This is usually the last column of the dataset (here: PURCHASED)! Because as you may guess, this company will PREDICT some future customers are going to buy the same product based on these informations.

___

## 1.5. Handling Missing Data
There are some missing data, which is normal in Machine Learning! If we look at the dataset we see that the Salary in Row 4 and the Age in the Raw 6.

You can not leave it like that, because it will cause error by training the model, therefore you must handle them. There are actually several ways to handle them:

1. The 1st way is just to IGNORE those Observations which hav missing data and deleting them. And that would be OK if you have a LARGE dataset so if you have for example 1% Missing data, you know that removing 1% of observations won't change much the LEARNING QUALITY of your MODEL. But sometimes you have a lot of missing data and therefore you must handle them the right way.

2. The 2nd way is to REPLACING the missing data by THE AVERAGE OF ALL THE VALUES IN THAT COLUMN (FEATURE), in which the data is missing.

3. Other ways could be to replace the missing value with the Median of that Column (Feature) or with the maximum frequent value in that column.

### Our Goals
* We want to replace the missing salary by the average of all the salaries, this is a CLASSIC WAY of handling missing data.

### Package ``Scikit Learn``
This is one of the important packages by the Data Preprocessing and it has very good tools for that. We will use it a lot in this course.

``pip install scikit-learn``

### How we handle it?
The class from the package Scikit-Learn is called ``SimpleImputer`` , we are actually going to first import that class, then we will create an Instance of that class, this object will allow us to exactly replace the missing salary, by the average of salaries

``fit(NUMERICAL_VALUES)`` It looks for the missing value and also calculate the replacement.
___

## 1.6. Categorical Data
___

## 1.8. Splitting the Dataset into the Training and Test Set
___

## 1.9. Feature Scaling
___

## 1.10. Our Data Processing Template
___
___
___