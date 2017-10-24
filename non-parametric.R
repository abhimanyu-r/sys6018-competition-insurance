# Non-parametric 

library(readr)
library(stringr)
library(randomForest)

train <- as.data.frame(read_csv("clean_train.csv"))

for(name in colnames(train)){
  if(str_detect(name, "cat")){
    if(length(levels(as.factor(train[[name]]))) > 53){
      train[[name]] <- NULL
    }else{
      train[[name]] <- as.factor(train[[name]]) 
    }
  }
}

train$target <- as.factor(train$target)
rf <- randomForest(target ~ . - id, train, nodesize = 100, ntree = 1)

trainx <- train[, -c(1,2)]
trainy <- train[, 2]

model1 <- rfcv(trainx = trainx, trainy = trainy)

