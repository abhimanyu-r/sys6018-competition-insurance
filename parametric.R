# Parametric script

library(readr)
library(stringr)

train <- read_csv("clean_train.csv")
for(name in colnames(train)){
  if(str_detect(name, "cat")){
    train[[name]] <- as.factor(train[[name]])
  }
}

# Due to performance constraints, we are limited to the validation set approach
indices <- sample(1:nrow(train), size = as.integer(0.8 * nrow(train)))
train.data <- train[indices, -1]
test.target <- train[-indices, 2]
test.data <- train[-indices, -c(1,2)]
rm(indices)

trim <- function(fit){
  fit$data <- NULL
  fit$y <- NULL
  fit$linear.predictors <- NULL
  fit$weights <- NULL
  fit$fitted.values <- NULL
  fit$model <- NULL
  fit$prior.weights <- NULL
  fit$residuals <- NULL
  fit$effects <- NULL
  fit$qr$qr <- NULL
  return(fit)
}
# Run a logistic regression model

model1 <- glm(target ~ ., train.data, family = "binomial"(link = "logit"))

# Extract only signficicant coefficients
data.frame(summary(model1)$coef[summary(model1)$coef[,4] <= .05, 4])

# Remove old model, make new one with only significant coefficients
rm(model1)
model2 <- glm(target ~ ps_ind_01 + ps_ind_02_cat + ps_ind_03 + 
                ps_ind_04_cat + ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin +
                ps_ind_15 + ps_ind_16_bin + ps_ind_17_bin + ps_reg_01 + ps_reg_02 +
                ps_car_02_cat + ps_car_08_cat + ps_car_09_cat + ps_car_11 + ps_car_12 + 
                ps_car_13 + ps_car_15, data = train.data, family = "binomial"(link = "logit"))

summary(model2)

# All variables significant, trim the model

model2 <- trim(model2)

# Predict using new data set
preds <- predict.glm(model2, newdata = test.data, type = "response")

sum(preds > 0.1)
