# Parametric script

library(readr)
library(stringr)
source("Gini.R")
source("preprocess.R")

train <- read_csv("clean_train.csv")
for(name in colnames(train)){
  if(str_detect(name, "cat")){
    train[[name]] <- as.factor(train[[name]])
  }
}

# Due to performance constraints, we are limited to the validation set approach
indices <- sample(1:nrow(train), size = as.integer(0.5 * nrow(train)))
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

model1 <- glm(target ~ ., train.data, family = "binomial")

# Extract only signficicant coefficients
data.frame(summary(model1)$coef[summary(model1)$coef[,4] <= .05, 4])

# Remove old model, make new one with only significant coefficients
rm(model1)
model2 <- glm(target ~ ps_ind_01 + ps_ind_02_cat + ps_ind_03 + 
                ps_ind_04_cat + ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin +
                ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_02 +
                ps_car_01_cat + ps_car_04_cat + ps_car_06_cat + ps_car_08_cat + 
                ps_car_09_cat + ps_car_11_cat + ps_car_12 + 
                ps_car_15, data = train.data, family = "binomial")

summary(model2)

# Extract only signficicant coefficients
data.frame(summary(model2)$coef[summary(model2)$coef[,4] <= .05, 4])

# All variables significant, trim the model

model2 <- trim(model2)

# Predict using new data set
preds <- predict.glm(model2, newdata = test.data, type = "response")

# Test the Gini on our predictor
normalized.gini.index(test.target$target, preds)
#  0.2445368

# Clean up the environment
rm(test.data, test.target, train, train.data, preds)

# Build the model on the full dataset
train <- read_csv("clean_train.csv")
for(name in colnames(train)){
  if(str_detect(name, "cat")){
    train[[name]] <- as.factor(train[[name]])
  }
}

# Preprocess test in the same way as train
test <- read_csv("test.csv")
test <- fill_missing(test, mean = F)
for(name in colnames(test)){
  if(str_detect(name, "cat")){
    test[[name]] <- as.factor(test[[name]])
  }
}

# Use sample to get our ids
sample <- read_csv("sample_submission.csv")

# Get the id for our sample data
id <- sample$id

# Clean up sample
rm(sample)

# Build full model
model.full <- glm(target ~ ps_ind_01 + ps_ind_02_cat + ps_ind_03 + 
                ps_ind_04_cat + ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin +
                ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_02 +
                ps_car_01_cat + ps_car_04_cat + ps_car_06_cat + ps_car_08_cat + 
                ps_car_09_cat + ps_car_11_cat + ps_car_12 + 
                ps_car_15, data = train, family = "binomial")

# Trim the model
model.full <- trim(model.full)

# Make the predictions
preds <- predict(model.full, newdata = test, type = "response")

# Create the results
results <- data.frame(id = id, target = preds)

# Write the results
write_csv(x = results, path = "parametric.csv")

# Clean up environment
rm(results, test, train, corrected_preds, id)
