
library(readr)
library(stringr)
source("Gini.R")
source("preprocess.R")
library(e1071)

train = fill_missing(read.csv('train.csv'))
for(name in colnames(train)){
  if(str_detect(name, "cat")){
    train[[name]] <- as.factor(train[[name]])}}

# Due to performance constraints, we are limited to the validation set approach
indices <- sample(1:nrow(train), size = as.integer(0.25 * nrow(train)))
train.data <- train[indices, -1]
test.target <- train[-indices, 2]
test.data <- train[-indices, -c(1,2)]
rm(indices)

svm.model.poly = svm(target~.,data=train.data,kernel='polynomial',degree=3,cost=1)

predict(svm.model.poly,test.data)

svm.model.rad = svm(target~.,data=train.data,kernel='radial',gamma=1,cost=1)


