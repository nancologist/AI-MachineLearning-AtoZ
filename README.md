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
### 2.1.1. Simple Linear Regression Intuition - Step 1
The linear equation and the name of its components:
![linear_equation](./images/linear_regression_01.png)

So we are going to look at an Example for Linear Regression: In the following we want to know homw the ``salary`` of employees in a company depends on their ``experience``.

![linear_regression_example](./images/linear_regression_02.png)

#### 2.1.1.1. Explaining The Above Diagram 
So now let's look at the simple in your regression because it's the easiest one to discuss. It's very pretty straightforward you can visualize it quite well.

So here we got the y and x axis. Let's look at that specific example where we have EXPERIENCE and SALARY. So experience is going to be our horizontal axis. Salary is all vertical axis and we want to understand how people's salary depends on their experience.

WELL WHAT WE DO IN REGRESSION IS WE DON'T JUST COME UP WITH A THEORY WE LOOK AT THE EVIDENCE WE'LL LOOK AT THE LIVE HARD FACTS SO HERE ARE SOME OBSERVATIONS WE'VE HAD.

So in a certain company this is how salaries are distributed among people who have different levels of experience and what a regression does.

So that's a formula for aggression. In our case it'll change to salary equals be zero plus ``b_1`` Times EXPERIENCE.

AND WHAT THAT ESSENTIALLY IT MEANS IS JUST PUTTING A LINE THROUGH YOUR CHART THAT BEST FITS THIS DATA and we'll talk about best fitting in the next tutorial when we're talking about ordinary squares.

But for now this is the chart. This is the line that best fits as Darren even looks like it right.
___

#### 2.1.1.2. Coefficient ``b_0``
For now let's focus on the coefficients and the caffeine and the constant.

So what does the constant mean here. Well the that actually means the point where the line crosses the vertical axis and let's say it's $30000.

What does that mean. Well it means that when when experience is zero. So when as you say on the horizontal axis when experience is at zero in the formula on the right you can see that the second part ``b_1`` Times experience becomes zero so salary equals zero.

That means that salary will equal to $30000 when a person has no experience so soon somebody is know fresh from University and joins this company. Most likely they will have a salary about $30000.
___

#### 2.1.1.3. Coefficient ``b_1``
Now what is ``b_1``, ``b_1`` IS THE SLOPE OF THE LINE. 

AND SO THE STEEPER THE LINE THE MORE YOU GET MORE MONEY YOU GET PER EXTRA YEAR OF EXPERIENCE.

Let's look at this. In this particular example let's say somebody went from I don't know maybe four to five years of ``experience``. So then to understand how his salary increase you have to project this onto the line and then project that onto the salary access and you can see that here for one of your experience the person will get AN EXTRA TEN THOUSAND DOLLARS ON TOP OF HIS SALARY.

So if the coefficient ``b_1`` is less, then the slope will be less and that means the salary increase will be less per every year of experience. If the slope is greater then that means the experience will yield more increase in salary and that's pretty much it.
___

That's how a simple your regression works. So the core goal here is that we're not just drawing a line theoretically that we can we came up with

SOME HOW WE'RE ACTUALLY USING OBSERVATIONS THAT WE HAVE TO FIND THE BEST FITTING LINE AND WHAT BEST FITTING LINE IS WE'LL TALK ABOUT THAT IN THE NEXT TUTORIAL.
___

### 2.1.2. Simple Linear Regression Intuition - Step 2

![linear_regression_HOW_works](./images/linear_regression_03.png)

#### 2.1.2.1 How Find Best Fitting Line
HOW THE LINEAR REGRESSION BEING A TREND LINE THAT BEST FITS YOUR DATA.

Today we'll find out how to find the best fitting light or in fact how the simple linear regression finds that line for you.

So here's our simple your aggression. The same chart salary versus experience. We've got these red dots which represent the actual observations that we have in our data and we've got the TREND LINE WHICH REPRESENTS THE BEST FITTING LINE OR THE SIMPLE LINEAR REGRESSION MODEL.

So now let's draw some vertical lines from the actual observations to the model. And let's look at one of the specific examples to understand what we're talking about here. 

So here you can see that the Red Cross is where that person is sitting at in terms of salary so let's say this person with 10 years of experience is earning $100000.

__-Interpretation-__  
WELL THE MODEL LINE (THE BLACK LINE), IT ACTUALLY TELLS US WHERE THAT PERSON SHOULD BE SITTING ACCORDING TO THE MODEL IN TERMS OF SALARY AND ACCORDING TO MODELS SHOULD BE A BIT LOWER. It should be somewhere without green crosses which is about maybe let's say thousand.

##### 2.1.2.2. Green And Red Pluses in Diagram:
So now the Red Cross is called ``y_i``. And that is the ACTUAL DURATION.

The Green Cross is called ``y_î`` (Y_i-hat)  is THE MODEL THE OBSERVATIONAL. or THE MODELED VALUE.

So basically with those that level of experience where would he be. Where does the model predict that he would be earning.

And so the green line therefore is the difference between what he's actually earning and what he should be earning.

So it should be what he's modeled to be earning. So therefore the green line will be the same regardless of what dependent variable you have whether it's salary or with it's grade school whatever. So it's the difference between the observed and the modeled for that level of independent variable.

#### 2.1.2.3. How Linear Regression Works!
Now to get this best fitting line what is done is you take the sum you take each one of those green lines are those distances (``y_i - y-î``) you square them and then you take some of those squares.

Once you have the sum of the squares for you got to find the MINIMUM of this ``SUM``!

So basically what a simple linear regression does is it draws lots and lots and lots of these lines. These trend lines all this is like a simplistic way of imagining the linear regression draws all these all possible trend trend lines (_Mori: Trend Lines are those vertical green lines!_) and counts the sum of those squares every single time.

And it store these SUMs somewhere in a temporary you know file or something like that and then 

IT FINDS THE MINIMUM ONE SO IT LOOKS FOR THE MINIMUM SUM OF SQUARES AND FINDS A LINE WHICH HAS THE SMALLEST SUM OF SQUARES POSSIBLE.

and that line will be the best fitting line and that is called the ordinary least squares method.

So that's how the simple linear regression works and look for you on the next tutorial.

___

### 2.1.3. Simple Linear Regression in Python - Importing and Splitting Data
#### 2.1.3.1. Data
* Every Row is corresponding to an EMPLOYEE (Observation).
* For each Employee we have two data : Years of Experience and Salary
___

#### 2.1.3.2. Goal
Building a simply linear regression model which will be trained to understand the correlation between the years of experience and the salary.
___

### 2.1.4. Simple Linear Regression in Python - Train Model
ERROR: ``X = dataset.iloc[:, 1].values`` had created a 1D array but for the ``linear_regression.fit()`` the ``X`` should be a 2D array. so i changed it to: ``X = dataset.iloc[:, 0:1].values``
___

### 2.1.5. Simple Linear Regression in Python - Predict Salaries
``y_pred`` is the vector of PREDICTIONS of the Dependent Variable. Here in our example this is the PREDICTED SALARY for all Observations in our TEST-SET.

Now we should compare the Prediction (``y_pred``) with ``y_test`` (y_test is the real data)
___

### 2.1.6. Simple Linear Regression in Python - Step 4
### 2.1.7. Simple Linear Regression in R - Step 1
### 2.1.8. Simple Linear Regression in R - Step 2
### 2.1.9. Simple Linear Regression in R - Step 3
### 2.1.10. Simple Linear Regression in R - Step 4
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








