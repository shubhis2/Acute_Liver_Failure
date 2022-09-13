# Acute-Liver-Failure

1. Introduction
This deliverable consists of the steps undertaken for the project. The steps completed so far after the corrections and updates of previous deliverables are selection of a project topic, description of dataset or project characteristics, descriptive analysis, summary statistics, graphs, analysis planning, importing qualitative variables as factors and implementing various classification algorithms mentioned in the textbook.
2. Project Proposal
2.1 Proposed Project Topic: ACUTE LIVER FAILURE
Description: This dataset was collected by JPAC Center for Health Diagnosis and Control, by conducting direct interviews, examinations, and blood samples. The dataset consists of selected information from 8786 adults of 20 years of age or older taken from 2008-2009 and 2014-2015 surveys. It gives the overview of the patient’s information like weight, height, obesity, BMI, blood pressure, cholesterol etc. As per the project criteria the following dataset has minimum missing values and mostly involves the numeric or binary values. The goal of the outcome variable is to predict the presence or absence of the liver disease.
Dataset location: https://www.kaggle.com/rahul121/acute-liver-failure
a)Number of Observations (n): 8786
Number of Predictor Variables (p): 29
Response variable and its type: The response variable in this dataset is ALF (Acute liver failure) and its type is categorical value. If the value is 0 then the patient does not have the liver disease. If the value is 1 the patient is diagnosed with the disease.
b) Predictor variables name and type:
   Predictor Variables
Age
Gender
Weight
Body Mass Index
Obesity
Maximum Blood Pressure Minimum Blood Pressure Good Cholesterol
Bad Cholesterol Dyslipidemia
PVD
Physical Activity Alcohol Consumption Hypertension
Type
Integer Character Numeric Numeric Integer Integer Integer Integer Integer Integer Integer Integer Integer Integer
                              3
  Diabetes Hepatitis
Region
Height
Waist
Total Cholesterol Education Unmarried Income
Source of Care Family Diabetes Family Diabetes Chronic Fatigue PoorVision
Integer Integer Character Numeric Numeric Integer Integer Integer Integer Character Integer Integer Integer Integer
                            Importance and Impact of Project: When a patient is diagnosed with cirrhosis, scar tissues slow the flow of blood through the liver. Over time, the liver cannot work the way it should. In severe cases, the liver gets so badly damaged that it stops working. Literature on liver cirrhosis incidence and prevalence is scarce, but statistics indicate that around 0.1% of Europe’s population is affected. Liver is a vital organ of the body part and thus, there is a need to identify the causes and risk factors of the liver failure to predict the disease before it becomes too severe. Liver is a rare disease with a high mortality rate and therefore people should be made aware about it because if diagnosed with the chronic liver disease, liver transplantation is the only lifesaving therapy.
Also, the project meets the criteria of the deliverable D0. The dataset has minimal missing values. The dataset involves categorical, numeric and binary values which makes it easier to analyze the dataset and the quality of the data is not compromised.
2.2. Other Considered Topics
IBM Employee Attrition and Performance
Description: The following data set shows the factors affecting employee’s attrition and performance in a company. Some of the key factors to be considered are Age, Hourly rate, Overtime, Distance from home, Years since last promotion and many more. These factors can be considered to predict whether a particular employee is likely to retire or resign.
Dataset location: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
 4

Number of Observations (n): 1471 Number of Predictor Variables (p): 7
Response variable and its type: The response variable in this dataset is attrition and its type is categorical.
Predictor variables name and type: Predictor Variables
Age
DistanceFromHome HourlyRate JobSatisfaction WorkLifeBalance PercentageSalaryHike YearsSinceLastPromotion
Type
Numeric Numeric Numeric Categorical Categorical Numeric Numeric
                  Importance of the Project: As the COVID-19 keeps unleashing its havoc, the world continues to get pushed into the crisis of the great economic recession, more and more companies start to cut down their underperforming employees. Companies firing hundreds and thousands of Employees is a typical headline today. Cutting down employees or reducing an employee salary is a tough decision to take. We can predict employee attrition using the given data set. It helps us to predict when an employee resigns/retires and is not replaced. This topic will be a challenging one to beat as there are many predictor variables and with different data types also it has many different columns which also can affect our outcome.
As we decided to consider health as a domain for this project, we are eliminating this topic.
5

Deaths due to Air Pollution
Description: Air pollution has reached an alarming rate in the past couple of years. It is one of the main causes of climate change and various diseases. This topic focuses on the following 5 rather fatal diseases caused by exposure to air pollution and the number of deaths in both males and females – 1) Respiratory infections, 2) Trachea, bronchus, lung cancer, 3) Ischemic heart disease, 4) Stroke and 5) Chronic obstructive pulmonary disease. This data is useful for predicting the percentage of deaths caused by a particular disease in a specified gender. It can also show what kind of disease is more persistent. Climate activists, government officials, organizations can refer to this data to make necessary actions forward for the betterment of the condition.
Dataset location: https://apps.who.int/gho/data/node.main.BODAMBIENTAIRDTHS?lang=en
Number of Observations (n): 1101
Number of Predictor Variables (p): 5
Response variable and its type: Percentage of deaths is the response variable and its type is numeric.
Predictor variables names and types:
   Predictor Variables
Country Cause Male Female Both sexes
Type
Nominal Nominal Numeric Numeric Numeric
            Importance of the Project/Goal: As it is more important now than ever to consider climate change as a global issue and the loss of lives caused by it, this project shines a light on the major contributor to the said issue – Air Pollution. Knowing which country has the highest rate/percentage of deaths caused by air pollution, can help the officials and the public take necessary actions and precautions if present in that country.
6

Though the data set passes some of the criteria mentioned in D0, it is not considered because data is not very clear. This might create inaccuracy in the outcome.
Early Stage Diabetes Risk Prediction
Description: The following dataset has been collected by doing the questionnaires from the patients who had diabetes at Sylhet hospital. The dataset consists of the patient’s information like age, sex, weight loss, itching, genital thrush, visual blurring etc. This information assists in predicting whether a person with certain attributes out of these is at risk of diabetes or not.
Dataset location : http://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.#
Number of Observations (n): 520
Number of Predictor Variables (p): 7
Response variable and its type: The response variable in this dataset is Class and its type is categorical. If the value is positive, then the patient has diabetes and if the value is negative then the patient is diabetic free.
Predictor variables name and type:
   Predictor Variables
Age
Sex
sudden weight loss partial paresis muscle stiffness Alopecia
Obesity
Type
Numeric Categorical Categorical Categorical Categorical Categorical Categorical
                7

Importance and Impact of dataset: The number of people having diabetes is steadily increasing day by day. It is one of the most chronic diseases in children. There are millions of people who have prediabetes and many live with diabetes undiagnosed. Some sets of signs and symptoms given in the dataset can be used to determine and predict the disease. This dataset can help in predictive analysis of this disease at an early stage.
Although this data set fills all criteria from D0, there are not many observations to work on. So, the prediction will not be the precise prediction that we are looking for hence we are eliminating this topic.
3. Project Characteristics and Descriptive Analysis
3.1 Description:
This section talks about the number of observations, parameters, response variables and predictor variables, summary statistics, histograms and bar plots in the acute liver failure dataset.
N (Number of Observations): 8785 p (Number of Parameters): 30
Response Variable: 1 i.e. ALF (Acute liver failure). If the value is 0 then the patient does not have liver disease. If the value is 1 the patient is diagnosed with the disease.
Predictor Variables: 29 i.e. Age, Gender, Weight, Body Mass Index, Obesity, Maximum blood pressure, Minimum blood pressure, Good cholesterol, Bad cholesterol, Dyslipidemia, PVD, Physical activity, Alcohol consumption, Hypertension, Diabetes, Hepatitis, Chronic fatigue, Region, Family HyperTension, PoorVision, Education, Unmarried, Income, Height, Waist, Source of Care, Family Diabetes, Hepatitis, Family Hepatitis.
8

3.2 Descriptive Analysis:
Summary statistics obtained from R for each variable:
Summary() function has been used to compute the minimum, first quartile, median, mean, third quartile and maximum values of the variables in the dataset.
 After cleaning and handling missing values:
 9

3.3 Histograms and Bar Plots:
Histograms for quantitative variables and bar charts for the qualitative variables all produced in R:
The function hist() computes the histogram and function barplot() computes the bar plot of the given data:
Ex: hist(newdata$Weight,prob = T,las =1, main = "Histogram of Weight", xlab = "Weight", col = "violet")
barplot(table(newdata$Obesity),las =1,main = "Bar plot of Obesity",xlab = "Obesity",col = "violet")
This dataset consists of 11 qualitative variable and 7 quantitative variables. Below are the figures for the same.
  10

   11

   12

  13

   14

   15

 3.4 Analysis Plan:
The above data is the result of data cleaning, data pre-processing and data preparation steps. We will be using classification for predictive modeling. Classification models like Logistic Regression, Naïve Bayes and K Nearest Neighbors will be used. These models will be compared based on results, accuracy and will be chosen accordingly. These models are implemented on the test and training data. We will analyze the various attributes of the dataset to answer the following research questions: Which attributes contribute the most to decide whether the person will be diagnosed with acute liver failure? Do some attributes relate to a higher chance than others of being at the risk of this disease? Such analysis will be useful in selecting the right predictive model for the dataset.
The response variable is ALF(Acute Liver Failure) and the predictor variables are mentioned above. Some of the variables are removed from the dataset. The variables that do not contribute or are irrelevant to the dataset have been removed. For example, region, height, waist, education, unmarried, source of care, poor vision of the patient that do not help in determining whether the patient might have acute liver failure or not. The other columns that have been removed from the dataset are Total cholesterol, family hypertension, family diabetes etc. These columns are eliminated as there are already other columns like Good/Bad cholesterol which gives us ample information in computing our result.
16

4. Predictive Modelling Exploration 4.1 Correlation Matrix:
The command ggcorrplot() gives visual analog of the correlation matrix of the data passed. The input to this command is the correlation matrix of the data set which is cor(norm1). It seems that Obesity, Body Mass Index and Weight have positive correlation amongst each other. Also, HyperTension is positively correlated to Maximum Blood Pressure which can help us in predicting our outcome. Diabetes and HyperTension shows good correlation which might also help in predicting Acute Liver Failure.
  17

4.2 Pair Plot:
  Implementation
Depending on the type of outcome variable, we have used Logistic Regression, Naïve Bayes and K- Nearest Neighbor classifier algorithms. We have experimented these models with different number of predictors. Performance of these models on both training and test data sets are documented.
4.3 Logistic Regression:
Logistic regression model is uses the predict() function to predict the ALF of the patient using different predictor variables. Over here, we have used the predictor variables Age, Alcohol Consumption, Diabetes and Chronic Fatigue. The glm() function is used to create the regression model and gets it summary for the analysis. In order to run the logistic regression in R, the argument family=binomial is passed which is not similar to other linear generalized models.
18

 The predict() function can be used to predict the probability that the patient has the acute liver failure, given values of the predictors. The type = “response” option tells R to output probabilities of the form P(Y = 1|X), as opposed to other information such as the logit. There is no data set supplied to the predict() function and therefore, the probabilities are computed for the training data that is used to fit the logistic regression model.
Further the confusion matrix is used to determine how many observations were correctly or incorrectly classified. The diagonal elements of the confusion matrix indicate the correct predictions while the off diagonals indicate the incorrect predictions. Hence the model has correctly classified that 4826 patients does not have acute liver failure whereas 42 patients have acute liver failure.
  19

After implementing the confusion matrix and predicting the correctly and the incorrectly classified elements we have predicted the accuracy of the model. We got the accuracy of the model as 92.35%
Below, we have taken combination of 5 and 7 various predictor variables and implanted logistic regression on them which gives us the accuracy of 92.4% and 92.1% accuracy that is almost like the previous considered predictor variables.
  20

 4.4 Naïve Bayes Classifier
First, the e1071 package is installed as it contains a function named naiveBayes() which is helpful in performing the Bayes classification. The function receives the categorical data and contingency table as input. It returns the object of class “naiveBayes”. The object is passed to predict outcomes of unlabeled subjects. In our dataset, a training model is created by using naiveBayes() function. The model is used to predict if the patient has ALF or not. Further, the a-priori probabilities tell how frequently each level of class occur in the training dataset. Conditional probabilities are calculated for each of the variables that we have taken.
21

 Further the test data is run through the model and confusion matrix is used to determine the correctly and incorrectly classified elements. The model states that the correctly classified elements are 3238 whereas the incorrectly classified elements are 276.
Next, we have calculated the accuracy of the model using the confusion matrix. Our dataset according to the model is 92.15% accurate.
 22

 Below, we have taken combination of 5 and 7 various predictor variables and implanted Naïve Bayes Classifier on them which gives us the accuracy of 91.7% that is almost like the previous considered predictor variables.
 23

 4.5 K-Nearest Neighbor
The class package is used for the kNN algorithm. The knn algorithm is standard for the k 5, 7 and 9. We have taken three predictor variables for the k nearest neighbour. The three predictor variables taken are PVD, hypertension and hepatitis. The knn is repeated for 3 times in order to get the best accuracy for the dataset. The first knn values are for k 5, 7 and 9 and among these the best accurate result was found for the k=9
  24

Similarly, we have repeated the knn for 2nd time with tune length as 15 where there will be different 15 k values be returned.
To create a 10-fold cross-validation based search of k, repeated 3 times we have to use the function trainControl
  25

 The confusion matrix is used to determine the accuracy of the model. The k-nn model is accurate with 92.09%.
 26

Below we have further combined the 5 more different predictor variables and compared the k values. After performing the model on the dataset with the different predictor variables the accuracy was 91.97%.
  27

  28

5 Resampling
Resampling can be performed through cross validation or Bootstrapping. Data is split into training set and test set initially in a certain ratio. This training data is taken into consideration for resampling. These resampling methods are performed on the three classification algorithms by using caret package of R.
5.1 Entire dataset as the training data.
The entire dataset is taken as the training data and applied on the naive bayes classifier.
  29

  30

  31

  32

 From the above output as the number of the predictor variables increase the accuracy of the model decreases. Therefore, when the whole dataset is taken as the training data the accuracy measure decreases.
33

5.2 Validation Set Approach
 The observations in the validation set approach are divided into two parts that is a training set and a validation set. On the training set, the model is then trained and the fitted model is use to predict the validation set. The validation set approach is splitting the dataset randomly. The entire data set is splitted into 98% training data and get the values of the RootMean Squared Error(RSME), Mean Absolute Error(MAE) and R2 Error.
5.3 K-Fold Cross Validation
k-fold cross validation
In this approach, the entire dataset is split into k equal sized parts. The model is then fit for (k- 1) Other parts and the prediction are obtained for the kth part. This way, each portion serves as the validation set for once. This records k errors and is averaged out to obtain cross validation error
We have applied this approach to the classification algorithms that we have used to build our model. As seen in the code snippets below, we have applied 5-fold and 10-fold cross validation to the training dataset. Firstly, train control is defined by specifying a method as ‘cv’ for cross validation and folds are defined using number=5 or number=10. The method trainControl(...) is offered by the Caret package of R.
train_control <- trainControl(method="cv", number=5)
The method argument can be used to identify the resampling method - cv, loocv,
boot etc whereas the number parameter gives either the number of folds or number
of the resampling iterations. After this, the model is trained using either of the three classification methods that we used for the predictive modelling. The model is
trained using train(...) method and summary(..) has been used to summarise the
results.
34

model_nb <- train(ALF~., data=newdata, trControl=train_control, method="nb")
Naïve Bayes
For k=10
  35

The naive bayes for the 10-fold cross validation is applied on the five predictor variables from the dataset and the above plot shows the accuracy measure of the naive bayes model.
K Nearest Neighbor
For K=10
  36

The above plot shows the values that are plotted for the k value that is for 5, 7 and 9. From the plot it can be seen that the highest accuracy is for the value k =7 and therefore the final values used for the model is k = 7
5.4 Leave One Out Cross Validation
Leave one out cross validation approach is the same as setting k as n, yielding n-folds cross validation. This means that only one observation is reserved as the validation set and the model is trained for the rest of the data. It takes higher execution time as cross validation is repeated n times.
In this code, the trainControl takes only the resampling method name as parameter i.e. LOOCV. After which the applied process is similar to above. The model is trained using the train(..) function with method names specified as ‘nb’ for naive bayes and ‘knn’ for k- nearest neighbours algorithms respectively:
model_nb <- train(ALF~., data=newdata, trControl=train_control, method="nb")
Naïve Bayes
  37

 38

K Nearest Neighbor
5.5 Bootstrap
Bootstrap is a resampling method to estimate the sampling distribution of your regression coefficients and therefore calculate the standard errors/confidence intervals of your regression
  39

coefficients. Repeatedly and arbitrarily, the nonparametric bootstrap resamples the observations with substitution that is some of the observations are drawn only once, while others are drawn many times and some never at all. All this is done in the logistic regression model and it is estimated and the coefficients are stored. They are repeated for n number of times. The function boot in R puts the bias in the which means that it is the difference between the regression coefficients in the single model and the mean of the bootstrap samples.
The bootstrap output shows the original regression coefficients and their bias, which is the distinction between the original and the bootstrapped coefficients. The standard errors are also given. Also, from the output the bootstrapped standard errors are larger than the original standard errors.
  40

 The above plot shows the histogram and the normal quantile comparison for all the predictive variables. The broken vertical line in the histogram shows the location of the regression coefficients for the model fit to the original sample.
41

6: Model Selection
6.1 Forward Selection:
  42

  43

  44

  45

  46

  47

 We have implemented forward selection using logistic regression. There are total 17 candidate terms which are our predictor variables. The process followed for 10 forward selection steps after that, no more variables were added. In those 10 steps, Model Summary, Parameter Estimates and ANOVA were calculated. There are total 10 selected variables Age, Minimum Blood Pressure, Hepatitis, PVD, HyperTension, Chronic Fatigue, Physical Activity, Good Cholesterol, Maximum Blood Pressure and Diabetes. Their Root Mean Square Error, Mean Absolute Error, Adj, R squared and many other details were calculated.
48

6.2 Backward Selection:
  49

  50

  51

  52

  53

  In backward selection, there are 17 candidate terms out of which 4 variables are eliminated they are Bad Cholesterol, Obesity, Alcohol Consumption and Dyslipidemia. Backward selection followed for 4 steps of the above mentioned 4 variables. Their Model summary, ANOVA, Parameter Estimates were calculated which includes root mean square error, adj. R-squared, mean absolute error and many more.
54

6.3 Ridge Regression
Ridge Regression tries to shrink the coefficients but keeps all variables in the model. The final values for the model were alpha =0 and lambda = 0.0001.
  55

The figure shows Log Lambda on X axis and Coefficients on Y axis. When log lambda is about 5, all the coefficients are 0 and then as we relax lambda, the coefficients starts to grow. So, increasing lambda helps to reduce the size of the coefficients.
 This is a variable importance plot. It shows that Physical Activity is the most important variable followed by Chronic Fatigue and the least important are at the bottom i.e. Bad Cholesterol, Minimum Blood Pressure so they may have coefficients very low.
56

6.4 Lasso Regression
 Lasso Regression does both shrinkage and feature selection. The final values used for the model were alpha = 1 and lambda = 0.001.
 The figure shows Log Lambda on X axis and Coefficients on Y axis. As we can see that coefficients 57

number 14 is growing much rapidly as the lambda is reduced compared to the positive ones.
This is the variable importance plot. It shows that Hypertension, Physical Activity, Chronic Fatigue are at the top and Maximum Blood Pressure, Bad Cholesterol, Alcohol Consumption contributes the least.
6.5 Generalized Additive Model
  Generalized Additive Models(GAMs) are simply a class of statistical models in which several non-linear
 smooth functions replace the linear relationship between the response and predictors to model and
 capture the non-linearities in the data. GAM helps to fit the linear models which can be either linearly or
 non-linearly dependent on several predictors with the help of the spline techniques. The generalized
 additive model follows the approach that the least significant columns drop from the dataset, refit the
 model and repeat this process until all terms are significant. From our dataset, it is seen that the numeric
 variables have the degree of freedom greater than zero meaning there is no need to refit the model.
58

 Simply, a linear model is summarized in order to compare between the linear model and the generalized additive model.
 59

 From the above result the output is separated into parametric and smooth or nonparametric parts. In this case, the only parametric component is the intercept and the smooth component of the model age, maximum blood pressure and minimum blood pressure suggests that it is statistically significant. Below, are the plots for the same.
 60

  61

7: Conclusion:
In this deliverable, we came to know that forward selection is a stepwise regression which begins with an empty model and adds in variable one by one. In our data set, we have 17 candidates and by implementing forward selection, the selected variables are 10 of them along with their statistics summary. Unlike forward selection, backward selection contains all the predictive variables and then iteratively removes the least useful predictor one at a time. In our case, there are 4 variables which are eliminated along with its parameter estimates and model summary. Ridge Regression tries to shrink the coefficients but keeps all variables in the model. The final values for the model were alpha =0 and lambda = 0.0001. We also came to know which variable contributes the most and which variable the least by plotting a variable importance graph. Lasso Regression does both shrinkage and feature selection. The final values used for the model were alpha = 1 and lambda = 0.001. As lasso does both shrinkage and feature selection, we will proceed with lasso which gives better picture, better accuracy as well as clear variable importance graph.
