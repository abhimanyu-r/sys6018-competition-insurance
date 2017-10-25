# Porto Seguro's Safe Driving Prediction
## Team

All team members participated in all parts of the project, but the primary responsibilities were distributed as follows:

* Abhimanyu Roy - Github coordinator, logistic regression approach

* Ben Greenawald - Data preprocessing, random forest approach

* Gregory Wert - Exploration and implementation of other approaches (e.g SVM)

  ​

## A preface on performance

Before we dive into our approach, we need to address an issue that drove much of our analysis, computational performance. This was the first Kaggle competition where the size of the dataset made performance a huge issue. For example, a simple logistic regression model would take a couple of minutes to fit. This means that the statistically valid method of variable selection, stepwise elimination, was out of the question. Instead, variables had to be eliminated in a batch fashion, where all insignificant variables were removed to get from one model to the next. This idea of sacrificing "statistical validity" in favor of more computational friendly approaches is one that appears throughout our approach to this competition.

One additional note, when dealing with random forests, R proved too slow to even be useful. The creation of a forest of 10 trees was taking a matter of minutes, meaning creating a forest of an actually useful size was intractable. For this reason, python was used for this particular implementation. Python provides a speedup just via the nature of the language but also provides more natural support for parallel computation.

