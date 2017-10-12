
# Scikit-learn hyperparameter search wrapper

Iaroslav Shcherbatyi, Tim Head and Gilles Louppe. June 2017.

## Introduction

This example assumes basic familiarity with [scikit-learn](http://scikit-learn.org/stable/index.html). 

Search for parameters of machine learning models that result in best cross-validation performance is necessary in almost all practical cases to get a model with best generalization estimate. A standard approach in scikit-learn is using `GridSearchCV` class, which takes a set of values for every parameter to try, and simply enumerates all combinations of parameter values. The complexity of such search grows exponentially with addition of new parameters. A more scalable approach is using `RandomizedSearchCV`, which however does not take advantage of the structure of a search space.

Scikit-optimize provides a drop in replacement for `GridSearchCV`, which utilizes Bayesian Optimization where a predictive model reffered to as "surrogate" is used to model the search space and utilized in order to arrive at good parameter values combination as soon as possible.

Note: for a manual hyperparameter optimization example, see "Hyperparameter Optimization" notebook.

## Minimal example
 
A minimal example of optimizing hyperparameters of SVC (Support Vector machine Classifier) is given below.



```python
from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),  
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=32
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
```

    /home/ubuntu/miniconda3/envs/testenv/lib/python3.6/site-packages/sklearn/model_selection/_split.py:2010: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)


    val. score: 0.991833704529
    test score: 0.993333333333


## Advanced example 

In practice, one wants to enumerate over multiple predictive model classes, with different search spaces and number of evaluations per class. An example of such search over parameters of Linear SVM, Kernel SVM and decision trees is given below. 


```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

# used to try different model classes
pipe = Pipeline([
    ('model', SVC())
])

# single categorical value of 'model' parameter is used  to set the model class
lin_search = {
    'model': Categorical([LinearSVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
}

dtc_search = {
    'model': Categorical([DecisionTreeClassifier()]),
    'model__max_depth': Integer(1,32),
    'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
}

svc_search = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    [(lin_search, 16), (dtc_search, 24), (svc_search, 32)], # (parameter space, # of evaluations)
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
```

    /home/ubuntu/miniconda3/envs/testenv/lib/python3.6/site-packages/sklearn/model_selection/_split.py:2010: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)


    val. score: 0.985152190052
    test score: 0.982222222222


## Iterative search utilizing `step` function

A single call to `step` function of the `BayesSearchCV` instance allows to perform partial search for at most `n_jobs` points in parallel. This allows to use custom stopping criterions and to pickle a parameter search class instance to the file for recovery in case of failures of any sort or in case computations need to be transported to a different machine. 

An example usage is shown below. 


```python
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

opt = BayesSearchCV(SVC())

# add instance of a search space 
opt.add_spaces('space_1', {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
})

for i in range(16):
    opt.step(X_train, y_train, 'space_1')
    # save the model or use custom stopping criterion here
    # model is updated after every step
    # ...
    score = opt.score(X_test, y_test)
    print(i, score)
```

    /home/ubuntu/miniconda3/envs/testenv/lib/python3.6/site-packages/sklearn/model_selection/_split.py:2010: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)


    0 0.0844444444444
    1 0.973333333333
    2 0.973333333333
    3 0.973333333333
    4 0.973333333333
    5 0.973333333333
    6 0.973333333333
    7 0.973333333333
    8 0.973333333333
    9 0.973333333333
    10 0.984444444444
    11 0.993333333333
    12 0.993333333333
    13 0.993333333333
    14 0.993333333333
    15 0.993333333333

