# Importing libraries for computing Imputation
library(mice)

# Getting the location of the directory
getwd()

# Importing the data/Read the csv file
liver <- read.csv("ALFdata.csv")
class(liver)

# Displaying few values from our data set
head(liver)

# Displaying all of the column names before dropping unnecessary columns
colnames(liver)

# Dropping unnecessary columns
df = subset(liver, select = -c(Region, Height, Waist, Total.Cholesterol, Education, Unmarried, Income, Source.of.Care, PoorVision, Family.HyperTension, Family.Diabetes, Family.Hepatitis))

# Displaying all of the column names after dropping unnecessary columns
colnames(df)

# Checking null values in our data set
sum(is.na(df$Age))
sum(is.na(df$Gender))
sum(is.na(df$Weight))
sum(is.na(df$Body.Mass.Index))
sum(is.na(df$Obesity))
sum(is.na(df$Maximum.Blood.Pressure))
sum(is.na(df$Minimum.Blood.Pressure))
sum(is.na(df$Good.Cholesterol))
sum(is.na(df$Bad.Cholesterol))
sum(is.na(df$Dyslipidemia))
sum(is.na(df$PVD))
sum(is.na(df$Physical.Activity))
sum(is.na(df$Hepatitis))
sum(is.na(df$Alcohol.Consumption))
sum(is.na(df$HyperTension))
sum(is.na(df$Diabetes))
sum(is.na(df$Chronic.Fatigue))
sum(is.na(df$ALF))

# Determining the Summary statistics for each variable (Mean, Median, Quartile)
summary(df)

#Calculating the percentage of missing data
p <- function(x) {sum(is.na(x))/length(x)*100}
apply(df, 2, p)

#Replacing the null values with Mean does not always give accurate outcome. So, we have used imputation to handle missing values.

#Impute. Using imputation to handle missing values 
impute <- mice(df[,1:18], m=3, method=c("","","pmm","pmm","logreg","pmm","pmm","pmm","pmm","","","pmm","","logreg","logreg","logreg","logreg","logreg"),seed = 123)
print(impute)

#Printing calculated imputation for Body Mass Index for our reference.
impute$imp$Body.Mass.Index

#Complete data. Replacing NA values with the computed iteration
newdata <- complete(impute, 2)

# Cleaned Data after replacing NA values with the computed once
summary(newdata)

# Converting "M" and "F" to 0 and 1 for Gender column.
require(dplyr)
newdata <- newdata %>%
  mutate(Gender = ifelse(Gender == "M",0,1))


# Histogram for Quantitative variables and Barplot for Qualitative variables
hist(newdata$Age,prob = T,las =1,main = "Histogram of Age",xlab = "Age",col = "violet")
barplot(table(newdata$Gender),las =1,main = "Bar plot of Gender",xlab = "Gender",col = "violet")
hist(newdata$Weight,prob = T,las =1, main = "Histogram of Weight", xlab = "Weight", col = "violet")
hist(newdata$Body.Mass.Index,prob = T,las =1, main = "Histogram of Body Mass Index", xlab = "Body Mass Index", col = "violet")
barplot(table(newdata$Obesity),las =1,main = "Bar plot of Obesity",xlab = "Obesity",col = "violet")
hist(newdata$Maximum.Blood.Pressure, prob = T,las =1,main = "Histogram of Maximum Blood Pressure", xlab = "Blood Pressure", col = "violet")
hist(newdata$Minimum.Blood.Pressure, prob = T,las =1,main = "Histogram of Minimum Blood Pressure", xlab = "Blood Pressure", col = "violet")
hist(newdata$Good.Cholesterol, prob = T,las =1,main = "Histogram of Good Cholesterol", xlab = "Cholesterol", col = "violet")
hist(newdata$Bad.Cholesterol, prob = T,las =1,main = "Histogram of Bad Cholesterol", xlab = "Cholesterol", col = "violet")
barplot(table(newdata$Dyslipidemia),las =1,main = "Bar plot of Dyslipidemia",xlab = "Dyslipidemia",col = "violet")
barplot(table(newdata$PVD),las =1,main = "Bar plot of PVD",xlab = "PVD",col = "violet")
barplot(table(newdata$Physical.Activity),las =1,main = "Bar plot of Physical Activity",xlab = "Physical Activity",col = "violet")
barplot(table(newdata$Alcohol.Consumption),las =1,main = "Bar plot of Alcohol Consumption",xlab = "Alcohol Consumption",col = "violet")
barplot(table(newdata$HyperTension),las =1,main = "Bar plot of HyperTension",xlab = "HyperTension",col = "violet")
barplot(table(newdata$Diabetes),las =1,main = "Bar plot of Diabetes",xlab = "Diabetes",col = "violet")
barplot(table(newdata$Hepatitis),las =1,main = "Bar plot of Hepatitis",xlab = "Hepatitis",col = "violet")
barplot(table(newdata$Chronic.Fatigue),las =1,main = "Bar plot of Chronic Fatigue",xlab = "Chronic Fatigue",col = "violet")
barplot(table(newdata$ALF),las =1,main = "Bar plot of ALF",xlab = "ALF",col = "violet")

#Normalizing the data
#library(caret)
#preproc1 <- preProcess(newdata[,c(1,3:4,6:9)], method=c("center", "scale"))
#norm1 <- predict(preproc1, newdata[,c(1,3:4,6:9)])
#summary(norm1)

data_norm <- function(x) {
  return ((x-min(x))/(max(x)-min(x)))
}
norm1<-as.data.frame(lapply(newdata[,1:18],data_norm))
str(newdata)
head(norm1)

# Importing all Qualitative variables as Factors
newdata[c('Gender', 'Obesity' ,'Dyslipidemia', 'PVD', 'Physical.Activity', 'Alcohol.Consumption', 'HyperTension', 'Diabetes', 'Hepatitis', 'Chronic.Fatigue', 'ALF')] <- lapply(newdata[c('Gender', 'Obesity' ,'Dyslipidemia', 'PVD', 'Physical.Activity', 'Alcohol.Consumption', 'HyperTension', 'Diabetes', 'Hepatitis', 'Chronic.Fatigue', 'ALF')],factor)
str(newdata)

# Relationship between predictors
#install.packages('ggcorrplot')
library(ggplot2)
library(ggcorrplot)
# library(corrplot)

#Correlation matrix
ggcorrplot(cor(norm1))

#Pair plot
pairs(norm1)
library(caret)
set.seed(1000)


#Partitioned the data into training and test dataset.Training dataset as 60% and testing as 40%
train_ind <- createDataPartition(newdata$ALF, p=0.6, list = F)
train_health <- newdata[train_ind, ]
test_health <- newdata[-train_ind, ]

train.new<-newdata[train_ind,18]
test.new <-newdata[train_ind,18]


#Logistic Regression

# logistic regression is carried out on the training set. For that we use the glm function, where glm stands for general linear model. We have to specify the name of 
#the family = binomial, since the distribution of the dependent variable is binomial.
model <- glm(ALF ~ ï..Age + Alcohol.Consumption + Diabetes + Chronic.Fatigue, data = train_health, family=binomial)



#summarizes the model
summary(model)

#Running the test data through the model using the predict function
res <- predict(model, test_health, type = "response")
res <- predict(model, train_health, type = "response")

#Validating the model using confusion matrix
confmatrix <- table(Actual_value=train_health$ALF, Predicted_value= res>0.5)
confmatrix

#Calculating the accuracy of model 
(confmatrix[[1,1]] + confmatrix[[2,2]])/ sum(confmatrix)

#Performing logistic regression model with different variables
model <- glm(ALF ~ ï..Age + Maximum.Blood.Pressure + Physical.Activity + Dyslipidemia + Weight + Body.Mass.Index + Diabetes, data = train_health, family=binomial)
summary(model)
res <- predict(model, test_health, type = "response")
res <- predict(model, train_health, type = "response")
confmatrix <- table(Actual_value=train_health$ALF, Predicted_value= res>0.5)
confmatrix
(confmatrix[[1,1]] + confmatrix[[2,2]])/ sum(confmatrix)

#Naive Bayes
#This package holds Naive Bayes function
library(e1071)

#It uses to predict the outcome of debate in test data set.
classifier_cl <- naiveBayes(ALF ~ Obesity + Minimum.Blood.Pressure + Physical.Activity + Bad.Cholesterol + PVD + Chronic.Fatigue, data = train_health) 

#Calculates the Conditional probability for each feature or variable and is created by model separately
classifier_cl 

#Running the test data through the model
y_pred <- predict(classifier_cl, newdata = test_health)

#Validating the model using confusion matrix
cm <- table(test_health$ALF, y_pred) 
cm 

#Calculating the accuracy of model
confusionMatrix(cm)

#Implementing Naive Bayes with different predictor variables
classifier_cl <- naiveBayes(ALF ~ Obesity + ï..Age + Weight + Body.Mass.Index, data = train_health)
classifier_cl 
y_pred <- predict(classifier_cl, newdata = test_health)
cm <- table(test_health$ALF, y_pred) 
cm 
confusionMatrix(cm)

#Class is used for the knn algorithm
library(class)

#Claculates the accuracy of the model for k =5,7,9
kNNFit <- train(ALF ~ PVD + Hepatitis + Diabetes, 
                data = train_health,
                method = "knn",
                preProc = c("center", "scale"))
print(kNNFit)


kNNFit1 <- train(ALF ~  PVD + Hepatitis + Diabetes, 
                 data = train_health,
                 method = "knn",
                 tuneLength = 15,
                 preProc = c("center", "scale"))
print(kNNFit1)

ctrl <- trainControl(method = "repeatedcv", repeats = 3)
kNNFit2 <- train(ALF ~  PVD + Hepatitis + Diabetes, 
                 data = train_health,
                 method = "knn",
                 tuneLength = 25,
                 trControl = ctrl,
                 preProc = c("center", "scale"))
print(kNNFit2)

knnPredict <- predict(kNNFit2,newdata = test_health )

#Calculating the accuracy of the model
confusionMatrix(knnPredict, test_health$ALF)

#Performing KNN algorithm with different predictor variables

kNNFit <- train(ALF ~ ï..Age + Maximum.Blood.Pressure + Gender + Physical.Activity + Dyslipidemia, 
                data = train_health,
                method = "knn",
                preProc = c("center", "scale"))
print(kNNFit)


kNNFit1 <- train(ALF ~  ï..Age + Maximum.Blood.Pressure + Gender + Physical.Activity + Dyslipidemia, 
                 data = train_health,
                 method = "knn",
                 tuneLength = 15,
                 preProc = c("center", "scale"))
print(kNNFit1)

ctrl <- trainControl(method = "repeatedcv", repeats = 3)
kNNFit2 <- train(ALF ~  Age + Maximum.Blood.Pressure + Gender + Physical.Activity + Dyslipidemia, 
                 data = train_health,
                 method = "knn",
                 tuneLength = 25,
                 trControl = ctrl,
                 preProc = c("center", "scale"))
print(kNNFit2)

knnPredict <- predict(kNNFit2,newdata = test_health )
confusionMatrix(knnPredict, test_health$ALF)



### Entire dataset as training data
split_data=sample(1:nrow(newdata),1 *nrow(newdata))
train_data = newdata[c(1:18)][split_data,]
nrow(train_data)

set.seed(17)
cv.error.10=rep(0,10)
for(i in 1:10){
   glm.fit<-glm(poly(ï..Age, i),data = newdata,family = "binomial")
   cv.error.10[i]<-cv.glm(newdata,glm.fit,K=10)$delta[1]
    }

#It uses to predict the outcome of debate in test data set.
classifier <- naiveBayes(ALF ~ Obesity + Minimum.Blood.Pressure + Physical.Activity + Bad.Cholesterol, data = train_data) 

#Calculates the Conditional probability for each feature or variable and is created by model separately
classifier 

#Running the test data through the model
y_prediction <- predict(classifier, train_data)

#Validating the model using confusion matrix
cm <- table(train_data$ALF, y_prediction) 
cm 

#Calculating the accuracy of model
confusionMatrix(cm)

#Implementing Naive Bayes with different predictor variables
classifier <- naiveBayes(ALF ~ Obesity + ï..Age + Weight + Body.Mass.Index + Good.Cholesterol + Physical.Activity, data = train_data)
classifier 
y_prediction <- predict(classifier, train_data)
cm <- table(train_data$ALF, y_prediction) 
cm 
confusionMatrix(cm)

classifier <- naiveBayes(ALF ~ Gender + Weight + HyperTension + Diabetes + Hepatitis + Maximum.Blood.Pressure + PVD, data = train_data)
classifier 
y_prediction <- predict(classifier, train_data)
cm <- table(train_data$ALF, y_prediction) 
cm 
confusionMatrix(cm)

### Validation Set Approach ###
library(caret)
set.seed(123)
# creating training data as 98% of the dataset 
random_sample <- createDataPartition(norm1 $ALF,  
                                     p = 0.80, list = FALSE) 

# generating training dataset 
training_dataset  <- norm1[random_sample, ] 
# generating testing dataset 
testing_dataset <- norm1[-random_sample, ] 

# Building the model 
model <- lm(ALF ~., data = training_dataset) 

# predicting the target variable 
predictions <- predict(model, testing_dataset) 

# computing model performance metrics 
data.frame( R2 = R2(predictions, testing_dataset $ ALF), 
            RMSE = RMSE(predictions, testing_dataset $ ALF), 
            MAE = MAE(predictions, testing_dataset $ ALF))

##leave one out cross validation(LOOCV)
# define training control
train_control <- trainControl(method="LOOCV")
   # train the model and summarize results
#For the naive bayes classifier
model_nb <- train(ALF~ ï..Age + Weight + Gender + Body.Mass.Index +Obesity, data=newdata, trControl=train_control, method="nb")
print(model_nb)
plot(model_nb)

#For the Knn Algorithm
model_knn <- train(ALF~., data=newdata, trControl=train_control, method = 'knn')
print(model_knn)
plot(model_knn)


##10-fold cross validation
# define training control
train_control <- trainControl(method="cv", number=10)

# train the model and summarize results
#For the naive bayes model
model_nb <- train(ALF~ ï..Age + Weight + Gender + Body.Mass.Index +Obesity, data=newdata, trControl=train_control, method="nb")
print(model_nb)
plot(model_nb)

#glm method helps in implementing the logistic regression model
model_glm <- train(ALF~., data=newdata, trControl=train_control, method = 'glm')
print(model_glm)
plot(model_glm)

#For the knn model
model_knn <- train(ALF~ Age + Gender + PVD + Diabetes, data=newdata, trControl=train_control, method = 'knn')
print(model_knn)
plot(model_knn)


##Bootstrap
library(boot)
logit.bootstrap <- function(data, indices) {
   
   newdata <- data[indices, ]
   fit <- glm(ALF~. ,data=newdata, family = "binomial")
   
   return(coef(fit))
}

set.seed(3456)
logit.boot <- boot(data=newdata, statistic=logit.bootstrap, R=8785)
logit.boot
glm.fit=glm(ALF~. ,data=newdata   ,family=binomial )
summary (glm.fit)
plot(logit.boot)


### Forward Selection ###
install.packages("olsrr")
library(olsrr)
model_glm <- glm(ALF~ ï..Age + Maximum.Blood.Pressure + PVD + Body.Mass.Index + Gender + Physical.Activity + Alcohol.Consumption + Diabetes,
                 data=norm1)
FWDfit.p<-ols_step_forward_p(model,penter=.05,details=TRUE)
#This gives you the short summary of the models at each step
FWDfit.p


### Backward Selection ###
model_glm <- glm(ALF~ ï..Age + Maximum.Blood.Pressure + PVD + Body.Mass.Index + Gender + Physical.Activity + Alcohol.Consumption + Diabetes,
                 data=norm1)
BWDfit.p<-ols_step_backward_p(model,prem=.05,details=TRUE)
BWDfit.p

### Ridge Regression ###
library(caret)
library(glmnet)
library(mlbench)
library(psych)

set.seed(1234)
model_glm <- train(ALF~., data=newdata, trControl=train_control, method = 'glm')

set.seed(1234)
ridge <- train(ALF~., data=newdata, method ='glmnet', tuneGrid = expand.grid(alpha = 0, lambda = seq(0.0001, 1, length=5)), trControl=train_control)

plot(ridge)
ridge
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(varImp(ridge, scale=F)) 

### Lasso Regression ###
set.seed(1234)
lasso <- train(ALF~., data=newdata, method = 'glmnet',
               tuneGrid = expand.grid(alpha=1, lambda =seq(0.001, 1, length = 5)),
               trControl = train_control)
plot(lasso)
lasso
plot(lasso$finalModel, xvar = "lambda", label = T)
plot(varImp(lasso, scale=F)) 

### Generalized Additive Models ###
install.packages("gam")

require(gam)
library(gam)
library(mgcv)
str(newdata)

mod_lm2 <- gam(ALF ~ Age + Maximum.Blood.Pressure + Minimum.Blood.Pressure + Gender + Obesity + Chronic.Fatigue + Diabetes + HyperTension ,data=newdata)
summary(mod_lm2)

AM1 <- gam(ALF~  s(Age) + s(Maximum.Blood.Pressure) + s(Minimum.Blood.Pressure) + Gender + Obesity + Chronic.Fatigue + Diabetes + HyperTension, data = newdata)
anova(AM1)
summary(AM1)
par(mfrow=c(1,8))
plot(AM1,se=TRUE)

