
library(readr)
library(stringr)
source("Gini.R")
source("preprocess.R")
library(e1071)

train = fill_missing(read.csv('train.csv'))
for(name in colnames(train)){
  if(str_detect(name, "cat")){
    train[[name]] <- as.factor(train[[name]])}}


train.data = stratified.sample(train,10000)
train.data$id = NULL

#------------------------------------------
#POLY
svm.model.poly = svm(target~.,data=train.data,kernel='polynomial',degree=3,cost=1,na.action = na.exclude)

summary(svm.model.poly)

test = read.csv('test.csv')
for(name in colnames(test)){
  if(str_detect(name, "cat")){
    test[[name]] <- as.factor(test[[name]])}}
preds = predict(svm.model.poly,test[2:58])

#------------------------------------------
#RAD
svm.model.rad = svm(target~.,data=train.data,kernel='radial',gamma=1,cost=1)
