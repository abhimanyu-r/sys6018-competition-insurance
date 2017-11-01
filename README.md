# Porto Seguro's Safe Driving Prediction
## Team

All team members participated in all parts of the project, but the primary responsibilities were distributed as follows:

* Abhimanyu Roy - Github coordinator, logistic regression approach

* Ben Greenawald - Data preprocessing, random forest approach

* Gregory Wert - Exploration and implementation of other approaches (e.g SVM)

## Repository Structure

The important files in this repository are the following:

 * preprocess.R: Script that contains some of the preprocessing done in R, especially for the parametric approach. Contains functionality to fill in missing values, drop columns that are perfectly multicolinear, and create stratified samples of the data.
 
 * Non Parametric.ipynb: Contains the random forest analysis done in python (the reason for python is detailed in the next section)
 
 * parametric.R: file that contains code use for our paramtetric approach.

## A preface on performance

Before we dive into our approach, we need to address an issue that drove much of our analysis, computational performance. This was the first Kaggle competition where the size of the dataset made performance a huge issue. For example, a simple logistic regression model would take a couple of minutes to fit. This means that the statistically valid method of variable selection, stepwise elimination, was out of the question. Instead, variables had to be eliminated in a batch fashion, where all insignificant variables were removed to get from one model to the next. This idea of sacrificing "statistical validity" in favor of more computational friendly approaches is one that appears throughout our approach to this competition.

One additional note, when dealing with random forests, R proved too slow to even be useful. The creation of a forest of 10 trees was taking a matter of minutes, meaning creating a forest of an actually useful size was intractable. For this reason, python was used for this particular implementation. Python provides a speedup just via the nature of the language but also provides more natural support for parallel computation.

## Preprocessing

Overall, this was a clean dataset that required minimal preprocessing. For random forest especially, the data could be used out of the box. For parametric (logistic regression), a little but of work was needed. Particularly, there were two columns that were perfectly multicolinear (aliased) to other columns and thus has to be removed for logistic regression to even work. 
Further, there were a few columns that had high densities of NA's which were dropped for parametric, and the rest of the missing values were imputed with the median (mean was found to be problematic because so many variables were categorical). This cleaned data was used for much of logistic regression. The cleaned data was not primarily used for random forest, but after we found a good forest model, we went back and used the clean data to see if performance improved.

## Parametric

The primary parametric method used was logistic regression since it has shown to work well for classification. Validation set approach with a 50/50 split was used to help address the performance constraints. Using the cleaned data described above, we first attempted a batch removal of variables with multicolinearity removed. Basically, we took a train set, fit a logistic regression model, then removed all insignificant variables at once, refit the model and repeat until all variables are significant. While this is not the statistically "good" approach for variable selection, it was the only method that got us results in a reasonable time. We used our test set and saw we had a decent Gini. We then fit this model on the full train set and got predictions using the test set. We then did a similar approach with a few tweaks. We went back and fit a model on the full train set without removal of multicollinear variables, since multicollinearity shouldn't have a huge impact on predictive power. We again did batch removal and got new predictions. A final attempt was to downsample the data in order to run a more statistically valid approach. Using a stratified 10,000 observation sample, backwards elimination was done to come up with a model.This model performed very poorly and was not used.

The secondary paremetic method used was Support Vector Machines (SVM). Due to performance constraints, the training data was made into a a stratified sample of 10,000 observations. This helped boost the performance enough to rapidly construct models and made cross validation feasible. Two kernels were investigated in constructing the model. The first was a polynomial kernel given a degree of 3 (a standard recommended in the ISLR textbook). Cross-validating the model using a 10-fold CV we got an error estimation of 0.03550885.
We also ran an SVM model with a radial kernel with a gamma set to 1. Upon completing 10 fold cross validation we attained an error estimation of 0.03528487. Thus, we were able to see that both kernel's were performing comparably, but with the polymetric one slightly outperforming. 

## Non Parametric

For non parametric, we went with the random forest approach. Bagging was computationally infeasible and boosting we just did not have time to try out. For the random forest, we started with the full, untouched data set. This is because random forest should be better at looking past noise in the data, thus preprocessing data which is already pretty clean shouldn't have any real affect. We first used OOB estimation to try and find a good number of features to consider for each tree (this is called mtry in R).
