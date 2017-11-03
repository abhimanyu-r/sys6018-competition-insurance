
library(readr)
library(stringr)
source("Gini.R")
source("preprocess.R")
library(e1071)

# Load in the Train Data
train = fill_missing(read.csv('train.csv'))
for(name in colnames(train)){
  if(str_detect(name, "cat")){
    train[[name]] <- as.factor(train[[name]])}}

# Stratify the data into a smaller sample
train.data = stratified.sample(train,50000)

# Load in the test data
test = fill_missing(read.csv('test.csv'))
for(name in colnames(test)){
  if(str_detect(name, "cat")){
    test[[name]] <- as.factor(test[[name]])}}

#------------------------------------------
#POLYNOMIAL KERNEL
set.seed(1)

# generate the model
svm.model.poly = svm(target~.-id,data=train.data,kernel='polynomial',degree=3,cost=1)
summary(svm.model.poly)

# generate predictions
set.seed(1)
preds = predict(svm.model.poly,test, type = 'response')
preds[preds<0] = 0
preds[preds>1] = 1

# cross validate
tune.svm(target~.,data=train.data,kernel='polynomial')

write.csv(data.frame(id=as.integer(test$id),target=preds),'svm_submissions.csv', row.names = FALSE)

#------------------------------------------------------------------------
#RADIAL KERNEL
svm.model.rad = svm(target~.,data=train.data,kernel='radial',gamma=1,cost=1)

tune.svm(target~.,data=train.data,kernel='radial',gamma=1,cost=1)
# the polynonmial performed better