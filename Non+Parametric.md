
# Non-Parametric Approach (Random Forests)

Here we explore a python implementation of random forest for this competition. The reason for the transition to python was a computational one. R was proving to be far too slow to create even trivially sized forests. Python gives an increase in speed by nature, but also has more natural integration of parallel tree creation, allowing for the creation of larger forests. That being said, computation is still an issue and will come into play as we go through.


```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
```

## Gini Coefficient

Code, adapted from the code on collab, to compute the gini index


```python
def unnormalized_gini_index(g, predicted_probabilities):
  
    if len(g) != len(predicted_probabilities):
        print("Actual and Predicted need to be equal lengths!")
        return

    # arrange data into table with columns of index, predicted values, and actual values
    d = {"truth": g, "pred": predicted_probabilities}
    gini_table = pd.DataFrame(data = d, index = range(1,len(g) + 1))

    # sort rows in decreasing order of the predicted values, breaking ties according to the index
    # gini_table = gini.table[order(-gini.table$predicted.probabilities, gini.table$index), ]
    gini_table = gini_table.sort_values("pred", ascending = False)

    # get the per-row increment for positives accumulated by the model 
    num_ground_truth_positivies = sum(gini_table["truth"])
    model_percentage_positives_accumulated = gini_table["truth"] / num_ground_truth_positivies

    # get the per-row increment for positives accumulated by a random guess
    random_guess_percentage_positives_accumulated = 1 / len(gini_table["truth"])

    # calculate gini index
    gini_sum = np.cumsum(model_percentage_positives_accumulated - random_guess_percentage_positives_accumulated)
    gini_index = sum(gini_sum) / len(gini_table["truth"]) 
    return(gini_index)


#' Calculates normalized Gini index from ground truth and predicted probabilities.
#' @param ground.truth Ground-truth scalar values (e.g., 0 and 1)
#' @param predicted.probabilities Predicted probabilities for the items listed in ground.truth
#' @return Normalized Gini index, accounting for theoretical optimal.
def normalized_gini_index(g, predicted_probabilities):
    model_gini_index = unnormalized_gini_index(g, predicted_probabilities)
    optimal_gini_index = unnormalized_gini_index(g, g)
    return(model_gini_index / optimal_gini_index)

```

## Baseline

First we are going to just try and fit a random forest to the raw train data and get a baseline for the gini index.


```python
# Read in the raw train set
train = pd.read_csv("train.csv")
```


```python
# Split into response and predictors
y = train["target"]
x = train.drop(["id", "target"], axis = 1)
```


```python
x.shape
```




    (595212, 57)




```python
# Create a baseline model
clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0, n_jobs=-1, max_features=1, oob_score=True)
```


```python
model = clf.fit(x, y)
```


```python
# Read in the test set
test = pd.read_csv("test.csv")
ids = test["id"]
test = test.drop(["id"], axis = 1)
```


```python
# Extract the probabilities
probs = clf.predict_proba(test)
probs_final = [x[1] for x in probs]
```


```python
# Create and write the results
result1 = {'id':ids, 'target':probs_final}
result1_df = pd.DataFrame(data = result1)

result1_df.to_csv("predictionsRF10-24-17.csv", index=False)
```

Gave a baseline score of  0.232

## Number of features

We now use OOB to look at the effect of the number of predictors considered for each tree.


```python
# Run OOB on number of features

for i in [1, 10, 20, 30]:
    
    # Use max depth of 11 from cross validation done below
    cl = RandomForestClassifier(n_estimators=500, max_depth=11, random_state=0, n_jobs=-1, max_features=i, oob_score=True)
    cl = clf.fit(x, y)
    print(str(i) + " " + str(model.oob_score_))
```

OOB Score for various sizes of features

1: 0.96355248214081701

10: 0.963542401699

20: 0.963552482141

30: 0.963545761846

Number of features does not seem to make a huge difference, so we will stick with the baseline of log(n)

## Variable Importance

Next we get a sense of some of the important predictors. This is more or less exploratory. We will run a 3 fold cv and find what variables show up in in the top 20 most important predictors for each fold.


```python
# Separate the x and the y
y_kfold = np.array(train["target"])
x_kfold = np.array(train.drop(["id", "target"], axis = 1))

# Split into three folds
kf = StratifiedKFold(n_splits=3)
kf.get_n_splits(x_kfold, y_kfold)

# Create a random forest that also store importance
forest = ExtraTreesClassifier(n_estimators=500, max_depth=2, random_state=0, n_jobs=-1, max_features="log2")

ginis = []

# List to store how many times a variable shows up in the top 20 most important variables
ratio = [0]*57

# Iterate over the 3 folds 
for train_index, test_index in kf.split(x_kfold, y_kfold):
    X_train, X_test = x_kfold[train_index], x_kfold[test_index]
    y_train, y_test = y_kfold[train_index], y_kfold[test_index]
    
    # Fit a model
    model = forest.fit(X_train, y_train)
    probs = forest.predict_proba(X_test)
    probs_final = [x[1] for x in probs]
    
    # Extract the importances
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # If a variable is in the top 20 importances, increase its value in the ratio list
    index = 0
    for f in range(X_train.shape[1]):
        # print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        if(index < 20):
            ratio[indices[f]] += 1
        index += 1

    ginis.append(normalized_gini_index(g=y_test, predicted_probabilities=probs_final))
    print(len(ginis))
```

    1
    2
    3



```python
# Print out the cross validation Gini
print(ginis)

# Find which variables top level of important in all folds
interest = [x for x in range(0,57) if ratio[x] > 2]
interest
```

    [0.21582272536451991, 0.22396413793473299, 0.22061223708988106]





    [3, 4, 5, 6, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 34]



Gives Gini scores of [0.21581968897515907, 0.22396513332529083, 0.22062593969640684]

and important columns of [3, 4, 5, 6, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 34]

## Cross Validation

We now dive into our actual cross validation. We will stick to 3-fold cross validation as opposed to OOB error just because this was the approach we took before learning about OOB and the results should be roughly the same. Plus, this will give us an idea of gini as opposed to accuracy. Further, we choose 3 fold as it showed to be the most computationally feasible. We will cross validate on both tree size and max depth of tree.


```python
# Split x and y
y_kfold = np.array(train["target"])
x_kfold = np.array(train.drop(["id", "target"], axis = 1))

# Split into folds
kf = StratifiedKFold(n_splits=3)
kf.get_n_splits(x_kfold, y_kfold)
forest = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=0, n_jobs=4, max_features="log2")

ginis = []
index = 1

# Iterate over folds to cross validate on various parameters
for train_index, test_index in kf.split(x_kfold, y_kfold):
    X_train, X_test = x_kfold[train_index], x_kfold[test_index]
    y_train, y_test = y_kfold[train_index], y_kfold[test_index]
    
    model = forest.fit(X_train, y_train)
    probs = forest.predict_proba(X_test)
    probs_final = [x[1] for x in probs]
    ginis.append(normalized_gini_index(g=y_test, predicted_probabilities=probs_final))
    print(index)
    index += 1
    
ginis
```

    1
    2
    3





    [0.25916813682529266, 0.2643633991281692, 0.2615973871654434]



Ginis for various tree sizes, 3 fold, max depth 2:

n = 100  : [0.22831513135331596, 0.23864099086790785, 0.23103020976139921]

n = 500  : [0.23399624333700408, 0.2411106344163601, 0.23516488700519048]

n = 750  : [0.23386079491144549, 0.24087016368279662, 0.23520949462470858]

n = 1000 : [0.23388456236128888, 0.24129153652280616, 0.23561118600698075]

Tree size seems to have little effects, beyond 500, so we will use that and cross validate on the maximum depth of the tree.

max depth = 3 : [0.24062733291355087, 0.24570173158111713, 0.24159062096266626]

max depth = 4 : [0.2426543130398667, 0.25000749888491419, 0.24464726289094912]

max depth = 5 : [0.24782642403706451, 0.25376128306085599, 0.2492086061125024]

max depth = 6 : [0.25131812425267502, 0.25640949938610264, 0.25279046253967674]

max depth = 7 : [0.2551048402902254, 0.26008715643937819, 0.25576034299523143]

max depth = 8 : [0.25678261394094259, 0.2616982944395666, 0.25828227944175075]

max depth = 9 : [0.25785094996229607, 0.26305865767777453, 0.25982247787905172]

max depth = 10 : [0.26043552198432962, 0.26494932536341631, 0.26163287418190628]

max depth = 11 : [0.26024302733938087, 0.266616577059546, 0.26264853956131246]

max depth = 12 : [0.25916813682529266, 0.2643633991281692, 0.2615973871654434]

Increasing depth does seem to have an effect, but it tapers off after 11.

## Actual Model on Raw Data (BEST MODEL)

Using what we have found, make a prediction tree using the raw, uncleaned data.


```python
# Read in train data
train = pd.read_csv("train.csv")
```


```python
# Separate predictors and response
y = train["target"]
x = train.drop(["id", "target"], axis = 1)
```

Using the best paramaters found using CV.


```python
# Create and fit the tree using best parameters
clf_test = RandomForestClassifier(n_estimators=500, max_depth=11, random_state=0, n_jobs=-1, max_features="log2")
clf_test.fit(x, y)
```


```python
# Predict using the test set
test = pd.read_csv("test.csv")
ids = test["id"]
test = test.drop(["id"], axis = 1)
probs = clf_test.predict_proba(test)
probs_final = [x[1] for x in probs]
```


```python
# Write the results
result1 = {'id':ids, 'target':probs_final}
result1_df = pd.DataFrame(data = result1)

result1_df.to_csv("predictionsRF11-1-17.csv", index=False)
```

**KAGGLE SCORE: 0.259**

## Rerun on clean dataset

Now we try again but use the cleaned dataset with NA's imputed with the median.


```python
# Read in the train data
train = pd.read_csv("clean_data_rf.csv")
```


```python
# Split the predictor and response
y = train["target"]
x = train.drop(["id", "target"], axis = 1)
```


```python
# Create and fit the random forest using best parameters from above
clf_test = RandomForestClassifier(n_estimators=500, max_depth=11, random_state=0, n_jobs=-1, max_features="log2")
clf_test.fit(x, y)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=11, max_features='log2', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=500, n_jobs=-1, oob_score=False, random_state=0,
                verbose=0, warm_start=False)




```python
# Read in the test data and predict
test = pd.read_csv("test.csv")
ids = test["id"]
test = test.loc[:, train.columns]
test = test.drop(["id", "target"], axis = 1)
probs = clf_test.predict_proba(test)
probs_final = [x[1] for x in probs]
```


```python
# Write the results
result1 = {'id':ids, 'target':probs_final}
result1_df = pd.DataFrame(data = result1)

result1_df.to_csv("predictionsRF11-1-17.csv", index=False)
```

**KAGGLE SCORE: 0.250**
