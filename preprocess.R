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

preprocess <- function(data, mean = F, drop_na = T, threshold = 0.1){

  # See how many missing variables have per category
  num_nas <- apply(data, 2, function(x){return(sum(x == -1))})
  num_nas 
  
  # Potentially drop ps_reg_03, ps_car_03_cat, ps_car_05_cat
  
  # Drop high density of missing values, more than 10%
  if(drop_na){
    train2 <- data[, num_nas < as.integer(nrow(data) * threshold)]
  }
  
  # Fill in missing values
  train3 <- fill_missing(train2, mean = mean)
  
  # Remove aliased columns
  train3$ps_ind_09_bin <- NULL
  train3$ps_ind_14 <- NULL
  
  # Clean up intermediate data set
  rm(train2)

  return(train3)
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
