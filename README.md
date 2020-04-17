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

### Features (Independent Variables) - INPUT VALUES
Features are the columns, which are the independent informations (here: Country, Age, Salary), with them you are going to PREDICT the DEPENDENT VARIABLE [Mori: For the future!].

``.ilock`` : To locate indexes  
``.iloc[:,:-1]`` : This takes all the Rows and also all the columns of the dataset, EXCEPT the last column (which is going to be predicted)

### Dependent Variable - TARGET VALUE
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
What are categorical variables and how to encode categorical data, which is illustrated in Python by LabelEncoder and OneHotEncoder class from sklearn.preprocessing library, and in R the factor function to transform categorical data into numerical variables.

### Label Encoder vs. One Hot Encoder (More)
So in the first columnt (Country) we have some countries which can't be understood by our ML Mode. So we should make them number, we could have do that with Label Encoder which gives for example to French, Germany, Spain these numbers: 1,2,3  

Now the Problem is that our MODEL is going to misunderstand this, because it's going to compare these values! But we know that the Countries can not be compared like that! So what can help us here is ``One Hot Encoder``! This makes the string data numeric without allowing them to be compared!

IN OUR CASE the OneHotEncoder splits the Country-Column into THREE columns, because we have also 3 countries (If we had 5 Countries, the Country-Col would have been splitted into 5 columns).

THE ONE-HOT-ENCODER CREATES ``BINARY VECTORS`` FOR EACH COUNTRY.

### 1.6.1. Encoding Independent Variable (Features - X)
#### ``sklearn.compose.ColumnTransformer``
* Instructor about Line-32 to Line-34 : We have to enter two arguments:
    1. ``transfomers`` : with that we specify what kind of transformation we are going to do and which indexes of column we want to transform
    2. ``remainder='passthrough''`` : which specifies we actually want to keep the columns which won't get this transformation, meaning "Age" and "Salary" untouched!
    
    3. More details: ``tansformer=[(KIND_OF_TRANSFORMATION, TYPE_OF_ENCODING, COLUMNS_TO_BE_APPLIED)]``
    
    4. fit_transform() : It does the both FITTING and TRANSFORMING at once! (This was not possible by ``imputer``, so there we have first used ``.fit()`` and then ``.transform()`` in LINE-24 and LINE-28)
    
    5. LINE-38 : We use ``numpy.array()`` because ``ColumnTransformer()`` does not do that for us. And it should be a Numpy-Array, otherwise our Model can not train with it!
    
    6. _You now know to apply the One-Hot-Encoding when you have several categories in the matrix of features, but also you can do a simple Label-Encoding when you have two classes which you can directly encode to 0 and 1, in other word BINARY OUTCOME._

* https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

* This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated TO FORM A SINGLE FEATURE SPACE. This is useful for heterogeneous or columnar data, to combine several feature extraction mechanisms or transformations into a single transformer.

* Parameter "transformers" : list of tuples  
List of (name, transformer, column(s)) tuples specifying the transformer objects to be applied to subsets of the data.

* Parameter "remainder" , also called ESTIMATOR.

#### ``sklearn.preprocessing.OneHotEncoder``
* https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder

* Encode categorical features as a one-hot numeric array.

* The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)
___

### 1.6.2. Encoding Dependent Variable (y)
#### ``sklearn.preprocessing.LabelEncoder``
* https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.htm  

* This transformer should be used to encode target values, i.e. ``y`` and not the input ``X``.

* It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.
___

## 1.7. Feature Scaling
### WHAT - Definition
__Feature Scaling is a technique that will put all your features in the SAME RANGE.__    

If we look at our ``Data.csv`` we can clearly see that the values of the Age-feature are NOT in the same RANGE as the values of the Salary-feature the Age-range is from like 0 to 100 and the Salary-Range is from 0 to 100 thousand.

Now we want to put the Age-Range and the Salary-Range in a same range using FEATURE SCALING technique.


### WHY - should we apply Feature Scaling?
For some of the ML Models (not all of them) if your different Features have a huge difference in range of their values, this can cause a __BIAS IN THE CORRELATIONS COMPUTATIONS__.  
In another word, the features that have higher values compared to the other ones will DOMINATE the other features so that these other features may NOT BE CONSIDERED in the Correlation Computation.

So depends on our Model sometimes we need to apply the Feature Scaling and sometimes it is not necessary, because the Model automatically can detect this issue and they fix this with ADAPTING the COEFFICIENTS (For example you see that with the Linear Regression. Linear Regression has some coefficients for each of the Features, so the Features with super HIGH VALUES will get a very LOW COEFFICIENTS. But for other regressions like Logistic Regressions or also the ML Model in R, we should apply the Feature Scaling)

### HOW - Feature Scaling Methods
* Standardisation : taking each value of the feature and subtract it by the Mean and then divided by the Standard Deviation. This puts all the values in range of usually between -3 and +3.

* Normalisation : In this we subtract the values of Feature by the Minimum value of the Feature and then divided by the Range (Max - Min). This will put all the values in the Feature between 0 and 1.

![feature scaling](./images/feature_scaling.png)

#### Standardisation vs. Normalisation
_Hadelin: I have had tons of experience with both and I did not observe much difference in the final accuracy and result between these two techniques._

``StandardScaler()`` keeps the X as numpy.array so we don't need to apply ``numpy.array()`` in LINE-48. 
___

___
## 1.8. Splitting Dataset

## 1.8.1 Splitting the Dataset into the Training and Test Set (Python)
In Machine Learning we split our data to a Training-Set and a Test-Set.  
You know that this is about the machine which is going to learn something to make predictions.

Imagine your machine learn to much on a dataset. Then we are not sure if its performance is great on a new set with a slightly different corrolations.

So we should always keep a part of data for the Test!

The performance of the Machine should not be that much different on the Test-Set comparing to Training-Set, so then we can conclude that this Model can understand Correlations (And he did not learn it by heart!) and so it can adapt the new sets of data in new situations.

BETWEEN 20% TO 30% OF DATA IS A GOOD CHOICE FOR THE TEST-PART!

### How Machine Learns Now?
Now the Machine Learning Model is going to find a CORRELATION between the X_train and y_train and with this Correlation it can predict a new_y for a new_X! or we can test it the quality of its Prediction with the X_test and y_test

___
### WARNING - Update (Part 1-8)
WARNING - Update
Dear students,

in the following tutorial, the first line of code we will type will be:

from sklearn.cross_validation import train_test_split 

However the "cross_validation" name is now deprecated and was replaced by "model_selection" inside the new anaconda versions.

Therefore you might get a warning or even an error if you run this line of code above.

To avoid this, you just need to replace:

``from sklearn.cross_validation import train_test_split`` 

by
``from sklearn.model_selection import train_test_split`` 
___

## 1.8.2 How is all these in R? 
### Importing Data
* ``read.csv(FILE_NAME)`` gives us a data frame created from imported data
* Attention! The working directory should be set correctly.
___

### Handling Missing Data in R with ``ifelse()``-function
``ifelse()`` returns a value with the same shape as test which is filled with elements selected from either yes or no depending on whether the element of test is TRUE or FALSE.
  
``ifelse(test, yes, no)``
#### Arguments
* ``test``    	
an object which can be coerced to logical mode.

* ``yes``	
return values for true elements of test.

* ``no``  	
return values for false elements of test.
___

### Encoding Categorial Data - with ``factor()``
Some data like the "Purchased" and "Country" are Categories (NOT NUMERICAL), we should assign to any of them a number, for example to the "Yes" or "No" for Purchase we can give 1 or 0 and to the countries also some numbers.

### Splitting dataset into training set and test set
Here we are going to use a package called "caTools" 

#### ``sample.split() - {caTools}``
Split data from vector Y into two sets in predefined ratio while preserving relative ratios of different labels in Y. Used to split the data used during classification into train and test subsets.

Usage
 sample.split( Y, SplitRatio = 2/3, group = NULL )  
Arguments  
Y	  
Vector of data labels. If there are only a few labels (as is expected) than relative ratio of data in both subsets will be the same.  

SplitRatio: This sets how much of data should go to the TRAIN-set

#### Return of sample.split()
This functions returns a list of TRUE and FALSE, where TRUE means that the Observation in that row goes to the TRAIN-set and the FALSE the Observation in that row goes to the TEST-set
___

## 1.10. Data PreProcessing in R
### 1.10.1. Dataset Description
___

### 1.10.2. R - Importing Dataset
Instead of ``setwd()`` to set the Working Directory we can use in the bottom right window: 

1. Browser to the folder of your project

2. in RStudio the ``More`` button and choose

3. Choose ``Set as working directory``.

### R is 1-Base!
ATTENTION: In contrary to a lot of programming languages, R is a 1-Base language this means the indexes in Data Structures and Datasets begin from 1 and NOT zero!
___

### 1.10.3. R - Taking care of Missing Data
Same explanations like in Python part.

We are going to replace the missing data with the average of their columns.

#### ifelse(condition, valIfTrue, valIfFalse)
This function works almost like a normal if-else statement, the first argument is the condition which should be evaluated.   
One of the second and the third arguments depends on the result of the first argument will be the output of this function. 

### is.na(VAR)
This function returns TRUE if the VAR is null or nothing. Otherwise FALSE.

### ave() vs. mean()
```r
v1 <- c(1,6,5)
mean(v1) # returns 4
ave(v1) # returns 4, 4, 4
``` 
___

### 1.10.4. R - Encoding Categorical Data
#### factor()
This function turns your categorical data (like Country and Purchased Features) into Numbers, so that your ML Model can use them in Training.

### One Hot Encoding in R 
Source: https://www.analytics-link.com/post/2017/08/25/how-to-r-one-hot-encoding  
```r
for(unique_value in unique(mydata$nationality)){
 
mydata[paste("nationality", unique_value, sep = ".")] <- ifelse(mydata$nationality == unique_value, 1, 0)
}
```
___

### 1.10.5. R - Splitting the dataset into the Training set and Test set
#### sample.split()
To split the dataset in R using the function ``sample.split()`` we need only to set the Y as the 1st argument and the second argument should be a number between 0 and 1 to specify the share of the Training Set.

By the way this function returns a vector of Trues and Falses for Observations, so if it returns for an Observation "True", this means, this Observation is chosen for the Training Set.
___

### 1.10.6. R - Feature Scaling
Same explanation like in the Feature Scaling Python Part above.

### scale()
Here we can not simply write so :
``training_set <- scale(training_set)`` Because the columns Country and Purchased are not Numeric yet!!! BECAUSE THE ``factor()`` IN R IS NOT A NUMERIC FUNCTION! 

Solution: We are going to exclude the Country and Purchases here!
___

### 1.10.7. R - Data PreProcessing Template
___
___
___

# 2. Regression
Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

In this part, you will understand and learn how to implement the following Machine Learning Regression models:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Support Vector for Regression (SVR)
* Decision Tree Classification
* Random Forest Classification

## 2.1. Simple Linear Regression
### 2.1.1. Dataset + Business Problem Description
### 2.1.2. Simple Linear Regression Intuition - Step 1
### 2.1.3. Simple Linear Regression Intuition - Step 2
### 2.1.4. Simple Linear Regression in Python - Step 1
### 2.1.5. Simple Linear Regression in Python - Step 2
### 2.1.6. Simple Linear Regression in Python - Step 3
### 2.1.7. Simple Linear Regression in Python - Step 4
### 2.1.8. Simple Linear Regression in R - Step 1
### 2.1.9. Simple Linear Regression in R - Step 2
### 2.1.10. Simple Linear Regression in R - Step 3
### 2.1.11. Simple Linear Regression in R - Step 4
___


## 2.2. Multiple Linear Regression
...
___

## 2.3. Polynomial Regression
...
___

## 2.4. Support Vector Regression (SVR)
...
___

## 2.5. Decision Tree Regression
...
___

## 2.6. Random Forest Regression
...
___

## 2.7. Evaluating Regression Models Performance
...
___








