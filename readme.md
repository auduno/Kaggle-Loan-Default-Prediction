# Kaggle Loan Default Prediction

This is code to generate my best submission to the [Kaggle Loan Default Prediction competition](http://www.kaggle.com/c/loan-default-prediction). This submission managed to give me a 4th place in the competition (under the alias [auduno](http://www.kaggle.com/users/48400/auduno)).

The challenge of this competition was to determine which loans in a portfolio of loans would default, as well as the size of the loss incurred for those who defaulted. The given dataset was an anonymized history of transactions for each loan. The challenge was sponsored by researchers at the Imperial College of London.

My prediction was done in two steps:

1. Creating a classifier to determine whether a loan would default or not. This classifier was based on a weighted blend of several classifiers, namely:
 * random forest classification
 * gradient boosting classification
 * generalized linear modelling with a cauchy link
2. Creating a regression model to estimate the size of the loss for those loans who did default. I created two regression models on different training datasets : the first model was trained on only data where the loss was positive (i.e. only loans who defaulted), and the second model was trained on data where loss was positive *as well* as some training data where loss was zero, more specifically loans where my classifier had problem classifying the loan correctly as a "defaulter". This ensured that the second regression model would be more robust to false positives, and would be able to effectively distinguish losses that should be zero. The final regression model was a 50/50 blend of these two models. Both regression models were a similarly weighted blend of several regression models, namely:
 * linear regression on cloglog-transformed losses
 * random forest regression on cloglog-transformed losses
 * random forest regression on inverse-transformed losses
 * gradient boosting regression on cloglog-transformed losses
 * gradient boosting regression on inverse-transformed losses

The features I used for training the classifier and regressor was a slightly modified subset of all features, as well as some new features created by ordering the dataset by features *f277* and *f521*.

## To recreate the submission

* download the competion data *test_v2.csv* and *train_v2.csv* and place it in the folder *data*
* run *CreateAddLines.R* (for instance by `Rscript CreateAddLines.R`) to create some new needed features for training and test set
* run *classify.py* to create the final submission as *submission.csv*