# Non-parametric 

library(readr)
library(stringr)
library(randomForest)
library(caret)

# https://stackoverflow.com/questions/7830255/suggestions-for-speeding-up-random-forests
library("foreach")
library("doSNOW")
registerDoSNOW(makeCluster(4, type="SOCK"))

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
rf <- foreach(ntree = rep(1, 4), .combine = combine, .packages = "randomForest") %dopar% 
  randomForest(x = train[, -c(1, 2)], y = as.factor(train[, 2]), ntree = ntree, nodesize = 1000)

trainx <- train[, -c(1,2)]
trainy <- train[, 2]

model1 <- rfcv(trainx = trainx, trainy = trainy)

