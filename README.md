# Porto Seguro's Safe Driving Prediction
## Team

All team members participated in all parts of the project, but the primary responsibilities were distributed as follows:

* Abhimanyu Roy - Github coordinator, logistic regression approach

* Ben Greenawald - Data preprocessing, random forest approach

* Gregory Wert - Exploration and implementation of other approaches (e.g SVM)

## Results

The best results for the required approaches are shown below.
 * Non-parametric (random forest): **0.259**
 * Parametric (logistic regression): 0.253

## Repository Structure

The important files in this repository are the following:

 * preprocess.R: Script that contains some of the preprocessing done in R, especially for the parametric approach. Contains functionality to fill in missing values, drop columns that are perfectly multicolinear, and create stratified samples of the data.
 
 * Non Parametric.ipynb: Contains the random forest analysis done in python (the reason for python is detailed in the next section). This file is also included as both an HTML and markdown file for easy viewing without having to run jupyter.
 
 * parametric.R: file that contains code use for our paramtetric approach (The file logistic.R contains some exploratory analysis on the logistic, but the final approach is contained in parametric.R).

## A preface on performance

Before we dive into our approach, we need to address an issue that drove much of our analysis, computational performance. This was the first Kaggle competition where the size of the dataset made performance a huge issue. For example, a simple logistic regression model would take a couple of minutes to fit. This means that the statistically valid method of variable selection, stepwise elimination, was out of the question. Instead, variables had to be eliminated in a batch fashion, where all insignificant variables were removed to get from one model to the next. This idea of sacrificing "statistical validity" in favor of more computational friendly approaches is one that appears throughout our approach to this competition.

One additional note, when dealing with random forests, R proved too slow to even be useful. The creation of a forest of 10 trees was taking a matter of minutes, meaning creating a forest of an actually useful size was intractable. For this reason, python was used for this particular implementation. Python provides a speedup just via the nature of the language but also provides more natural support for parallel computation.

## Preprocessing

Overall, this was a clean dataset that required minimal preprocessing. For random forest especially, the data could be used out of the box. For parametric (logistic regression), a little but of work was needed. Particularly, there were two columns that were perfectly multicolinear (aliased) to other columns and thus has to be removed for logistic regression to even work. 
Further, there were a few columns that had high densities of NA's which were dropped for parametric, and the rest of the missing values were imputed with the median (mean was found to be problematic because so many variables were categorical). This cleaned data was used for much of logistic regression. The cleaned data was not primarily used for random forest, but after we found a good forest model, we went back and used the clean data to see if performance improved.

## Parametric

The primary parametric method used was logistic regression since it has shown to work well for classification. Validation set approach with a 50/50 split was used to help address the performance constraints. Using the cleaned data described above, we first attempted a batch removal of variables with multicolinearity removed. Basically, we took a train set, fit a logistic regression model, then removed all insignificant variables at once, refit the model and repeat until all variables are significant. While this is not the statistically "good" approach for variable selection, it was the only method that got us results in a reasonable time. We used our test set and saw we had a decent Gini. We then fit this model on the full train set and got predictions using the test set. We then did a similar approach with a few tweaks. We went back and fit a model on the full train set without removal of multicollinear variables, since multicollinearity shouldn't have a huge impact on predictive power. We again did batch removal and got new predictions. This model ended up being the best of the approaches we tried. A final attempt was to downsample the data in order to run a more statistically valid approach. Using a stratified 10,000 observation sample, backwards elimination was done to come up with a model. This model performed very poorly and was not used.

The secondary paremetic method used was Support Vector Machines (SVM). Due to performance constraints, the training data was made into a a stratified sample of 10,000 observations. This helped boost the performance enough to rapidly construct models and made cross validation feasible. Two kernels were investigated in constructing the model. The first was a polynomial kernel given a degree of 3 (a standard recommended in the ISLR textbook). Cross-validating the model using a 10-fold CV we got an error estimation of 0.03550885.
We also ran an SVM model with a radial kernel with a gamma set to 1. Upon completing 10 fold cross validation we attained an error estimation of 0.03528487. Thus, we were able to see that both kernel's were performing comparably, but with the polymetric one slightly outperforming. 

## Non Parametric

For non parametric, we went with the random forest approach. Bagging was computationally infeasible and boosting we just did not have time to try out. For the random forest, we started with the full, untouched data set. This is because random forest should be better at looking past noise in the data, thus preprocessing data which is already pretty clean shouldn't have any real affect. We first used OOB estimation to try and find a good number of features to consider for each tree (this is called mtry in R). 500 trees were used for each test, and using test values of mtry = 1, 10, 20, and 30 (anything above 30 took extremely long to run), we found practically no difference in the performance based on the value of mtry, thus we proceeded forward with the sklearn default, which is log(p). For the sake of exploration, we went and looked at what variables were most important in the model, but the results ended up not being used.
We then needed to optmized the other levels that could be pulled, namely, number of trees and depth of trees. For these, we ended up using cross validation. This is because using cross validation let us use gini as our metric instead of raw accuracy, which OOB in sklearn uses. We chose 3-fold CV because it provided the most reasonable computation time. Using this approach, we tested number of trees range from 100 to 1000 and found that the performance improved as a function of trees until about 500 when it tapered off. Similarly, we explored depth values from 1-12 and found that it kept improving up until 11, where it tapered off (this is why we stopped at 12. Also, anything beyond 12 took forever to run). We then built a model on the full train using these parameter values and got our best score of 0.259. We went back and refit this model on the clean data set from logistic regression, and found that performance dropped, so we kept just using raw data.

## Conclusions

In the end, the random forest model did prove to be the best in our analysis. There are a variety of ways we could go about improving the results, the primary way being obtaining more compute power. This was the first competition where performance drove the way we approached our analysis. Pushing analysis off to AWS or Rivanna could allow us to explore more ensemble models, lik bagging or boosting (boosting we may be able to do locally, we ran out of time to try it). More computing power could also have allowed us to do better variable selection on the logistic regression, such as stepwise elimination. In the end, this competition ended up being a balancing act of using the most data to that we could to get the most accurate results without crashing our poor laptops.
