# Preprocess script

library(readr) # For reading in the data
library(stringr) # For string contains operator
library(car) # For VIF functions


# Helper Functions --------------------------------------------------------

# Function to calculate the mode
# From: https://www.tutorialspoint.com/r/r_mean_median_mode.htm
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Fills in the missing values for the dataset
# If the column in categorical or binary, the mode is used
# Else, the option of median and mean is used, with mean as default
fill_missing <- function(dataset, mean = T){
  for(name in colnames(dataset)){
    if(str_detect(name, "cat") || str_detect(name, "bin")){
      dataset[dataset[[name]] == -1, name] <- getmode(dataset[[name]])
    }else{
      if(mean){
        dataset[dataset[[name]] == -1, name] <- mean(dataset[[name]])  
      }else{
        dataset[dataset[[name]] == -1, name] <- median(dataset[[name]])
      }
    }
  }
  
  return(dataset)
}


# Exploratory -------------------------------------------------------------

exploratory <- function(){
  # Read in the train data
  train <- read_csv("train.csv")
  
  # See how many missing variables have per category
  num_nas <- apply(train, 2, function(x){return(sum(x == -1))})
  num_nas 
  
  # Potentially drop ps_reg_03, ps_car_03_cat, ps_car_05_cat
  
  # Drop high density of missing values, more than 10%
  train2 <- train[, num_nas < as.integer(nrow(train) * 0.1)]
  
  # Fill in missing values
  train3 <- fill_missing(train2, mean = F)
  
  # See how many missing variables have per category
  num_nas2 <- apply(train3, 2, function(x){return(sum(x == -1))})
  # Now we have no NA's in our dataset
  
  # Clean up intermediate data sets
  rm(train)
  rm(train2)
  
  # Write the current clean file to csv for quicker reads in the future
  write_csv(train3, "clean_train.csv")
  
  # Read in the clean data
  train3 <- read_csv("clean_train.csv")
  
  # Create initial linear model to try and find the predictors with
  # multicollinearity
  mod1 <- lm(target ~ . - id, train3, singular.ok = F)
  # We have perfect multicollinearity in the data
  
  # Use the alias function to figure out the problems
  alias(lm(target ~ . - id, train3))
  # Perfect multicollinearity in ps_ind_09_bin and ps_ind_14
  
  # Drop those two columns
  train3$ps_ind_09_bin <- NULL
  train3$ps_ind_14 <- NULL
  
  # Make a new model without pure multicollinearity
  mod2 <- lm(target ~ . - id, train3, singular.ok = F)
  
  # Remove columns with high VIF
  vif(mod2)
  
  # Remove those variables
  train3$ps_ind_16_bin <- NULL
  
  # Try again
  mod2 <- lm(target ~ . - id, train3, singular.ok = F)
  
  # Remove columns with high VIF
  vif(mod2)
  
  train3$ps_car_13 <- NULL
  
  # One more time
  # Try again
  mod2 <- lm(target ~ . - id, train3, singular.ok = F)
  
  # Remove columns with high VIF
  vif(mod2)
  
  
  # Write the current clean file to csv for quicker reads in the future
  write_csv(train3, "clean_train.csv")
  rm(train3)
}

# Gets a stratified sample of the data roughly maintaining class distribution
stratified.sample <- function(data, size, seed = 0){
  set.seed(seed)
  
  # Get class proportion
  prop <- sum(data$target == 1)/nrow(data)
  
  # Split the data into positive and negative data
  positived_data <- data[data$target == 1, ]
  negative_data <- data[data$target == 0, ]
  neg_size <- as.integer(size * (1-prop))
  pos_size <- size - neg_size
  
  # Get random samples
  pos_sample <- sample(1:nrow(positived_data), size = pos_size)
  neg_sample <- sample(1:nrow(negative_data), size = neg_size)
  
  return(rbind(positived_data[pos_sample, ], negative_data[neg_sample, ]))
}

l <-  stratified.sample(train2, 10000)
l
