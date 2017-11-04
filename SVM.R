
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
# ran with 10,000 when using tune for CV

# Load in the test data
test = fill_missing(read.csv('test.csv'))
for(name in colnames(test)){
  if(str_detect(name, "cat")){
    test[[name]] <- as.factor(test[[name]])}}

#------------------------------------------------------------------------
#RADIAL KERNEL
# set seed
set.seed(1)

# generate model
svm.model.rad = svm(target~.,data=train.data,kernel='radial',gamma=1,cost=1)
summary(svm.model.rad)

#generate predictions
preds.rad = predict(svm.model.rad,test, type = 'response')

# output prediction
write.csv(data.frame(id=as.integer(test$id),target=preds.rad),'svm_submissions.csv', row.names = FALSE)

# cross validate (used 10,000 sample)
tune.svm(target~.,data=train.data,kernel='radial',gamma=1,cost=1)

# the radial performed better
#------------------------------------------
#POLYNOMIAL KERNEL
set.seed(1)

# generate the model
svm.model.poly = svm(target~.-id,data=train.data,kernel='polynomial',degree=3,cost=1)
summary(svm.model.poly)

# generate predictions
preds.poly = predict(svm.model.poly,test, type = 'response')
# model not performing well likely the cause of values beyond the domain [0,1]
# morph outside values
preds.poly[preds.poly<0] = 0
preds.poly[preds.poly>1] = 1

# write.csv(data.frame(id=as.integer(test$id),target=preds.poly),'svm_submissions.csv', row.names = FALSE)

# cross validate (used 10,000)
tune.svm(target~.,data=train.data,kernel='polynomial')
