library(tidyverse)
library(caret)
library(stringr)
#Loads the cleaned data. The pre-process script was used for cleaning
train3=read.csv('train3.csv')
#Specifies the categorical variables
for(name in colnames(train3)){
  if(str_detect(name, "cat")){
    train3[[name]] <- as.factor(train3[[name]])
  }
}
#Partitions to train and test portions 0.65 was used for partitioning after 
#trial and error and recalculating the Gini Index based on the results.
 inTrain<-createDataPartition(y=train3$target, p=0.65, list=FALSE)
 traindata<-train3[inTrain,]
 testdata0<-train3[-inTrain,]
#Builds the parametric model 
para_model<-glm(target~.-id,family="binomial",data=traindata)
#Findsthe significant predictors    
which(summary(para_model)$coef[,'Pr(>|z|)']<0.05)
#Rebuilds the model
para_model3<-glm(target~ps_ind_01+ps_ind_02_cat+ps_ind_03+ps_ind_04_cat+ps_ind_05_cat+ps_ind_07_bin+ps_ind_08_bin+ps_ind_15+ps_ind_17_bin+ps_reg_01+ps_reg_02+ps_car_01_cat+ps_car_04_cat+ps_car_06_cat+ps_car_09_cat+ps_car_11_cat+ps_car_12+ps_car_15,data=traindata)
summary(para_model3)

#Crossvalidation
predictions<-0
predictions<-predict.glm(para_model3,testdata0,type="response")
library(boot)
cv.glm(testdata0,para_model3, K=10)
#Delta=0.035392

#Load the clean test data using the preprocess script
test<-read.csv('clean_test.csv')
for(name in colnames(test)){
  if(str_detect(name, "cat")){
    test[[name]] <- as.factor(test[[name]])
  }
}
#Generate the predictions
testpredictions<-0
testpredictions<-predict.glm(para_model3,test,type="response")
output<-as.data.frame(as.character(test$id))
#Write out the CSV file
colnames(output)<-c('id','target')
write.csv(output,'output.csv',row.names=FALSE)
