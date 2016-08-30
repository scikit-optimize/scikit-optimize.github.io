
# Tuning a scikit-learn estimator with `skopt`

Gilles Louppe, July 2016 <br />
Katie Malone, August 2016


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)
```

## Problem statement

Tuning the hyper-parameters of a machine learning model is often carried out using an exhaustive exploration of (a subset of) the space all hyper-parameter configurations (e.g., using `sklearn.model_selection.GridSearchCV`), which often results in a very time consuming operation. 

In this notebook, we illustrate how `skopt` can be used to tune hyper-parameters using sequential model-based optimisation, hopefully resulting in equivalent or better solutions, but within less evaluations.

## Objective 

The first step is to define the objective function we want to minimize, in this case the cross-validation mean absolute error of a gradient boosting regressor over the Boston dataset, as a function of its hyper-parameters:


```python
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

boston = load_boston()
X, y = boston.data, boston.target
reg = GradientBoostingRegressor(n_estimators=50, random_state=0)

def objective(params):
    max_depth, learning_rate, max_features, min_samples_split, min_samples_leaf = params

    reg.set_params(max_depth=max_depth,
                   learning_rate=learning_rate,
                   max_features=max_features,
                   min_samples_split=min_samples_split, 
                   min_samples_leaf=min_samples_leaf)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1, scoring="mean_absolute_error"))
```

Next, we need to define the bounds of the dimensions of the search space we want to explore, and (optionally) the starting point:


```python
space  = [(1, 5),                           # max_depth
          (10**-5, 10**-1, "log-uniform"),  # learning_rate
          (1, X.shape[1]),                  # max_features
          (2, 30),                          # min_samples_split
          (1, 30)]                          # min_samples_leaf

x0 = [3, 0.01, 6, 2, 1]
```

## Optimize all the things!

With these two pieces, we are now ready for sequential model-based optimisation. Here we compare gaussian process-based optimisation versus forest-based optimisation.


```python
from skopt import gp_minimize
res_gp = gp_minimize(objective, space, x0=x0, n_calls=50, random_state=0)

"Best score=%.4f" % res_gp.fun
```




    'Best score=2.8595'




```python
print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], 
                            res_gp.x[2], res_gp.x[3], 
                            res_gp.x[4]))
```

    Best parameters:
    - max_depth=5
    - learning_rate=0.090178
    - max_features=7
    - min_samples_split=24
    - min_samples_leaf=2



```python
from skopt import forest_minimize
res_forest = forest_minimize(objective, space, x0=x0, n_calls=50, random_state=0)

"Best score=%.4f" % res_forest.fun
```




    'Best score=2.9195'




```python
print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_forest.x[0], res_forest.x[1], 
                            res_forest.x[2], res_forest.x[3], 
                            res_forest.x[4]))
```

    Best parameters:
    - max_depth=4
    - learning_rate=0.089097
    - max_features=8
    - min_samples_split=6
    - min_samples_leaf=3


As a baseline, let us also compare with random search in the space of hyper-parameters, which is equivalent to `sklearn.model_selection.RandomizedSearchCV`.


```python
from skopt import dummy_minimize
res_dummy = dummy_minimize(objective, space, x0=x0, n_calls=50, random_state=0)

"Best score=%.4f" % res_dummy.fun
```




    'Best score=3.0592'




```python
print("""Best parameters:
- max_depth=%d
- learning_rate=%.4f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_dummy.x[0], res_dummy.x[1], 
                            res_dummy.x[2], res_dummy.x[3], 
                            res_dummy.x[4]))
```

    Best parameters:
    - max_depth=5
    - learning_rate=0.0596
    - max_features=10
    - min_samples_split=23
    - min_samples_leaf=1


## Convergence plot


```python
from skopt.plots import plot_convergence
plot_convergence(("gp_optimize", res_gp),
                 ("forest_optimize", res_forest),
                 ("dummy_optimize", res_dummy))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f4364b1c0f0>




![png](hyperparameter-optimization_files/hyperparameter-optimization_18_1.png)


## Part 2: Tuning a scikit-learn pipeline with `skopt`

### Introduction

Scikit-learn objects (transformers, estimators) are often not used singly, but instead chained together into a <a href="http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">pipeline</a>.  When that happens, there can be several different sets of hyperparameters to examine, one for each object.  In the same way that `GridSearchCV` can be applied to a pipeline to tune the hyperparameters of several objects at once, we can do a more efficient search (this example uses GPs) over more than one scikit-learn object.  

We'll do that here now.  A common method for reducing dimensionality is to preprocess with <a href="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">principal components analysis</a>, or PCA.  A free parameter of PCA is how many components to use; now we'll optimize over this parameter as well as the parameters of the gradient boosted classifier.  It's not guaranteed that using PCA will give us better results than the classifier all by itself, and there are other problems where PCA might make more sense than it does here, but a reasonable person might want to try it out anyway.  Here's how you can do that.


```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca = PCA()
reg_pipe = GradientBoostingRegressor(n_estimators=50, random_state=0)
pipe = Pipeline([('pca', pca), ('reg', reg_pipe)])
```

### Defining the search space

Our parameter space has dimensions for the n_components parameter of PCA, as well as several parameters of the decision tree.
Optionally, we can also define a starting point for the search.


```python
pipe_space  = [(1,13),                           # n_components of PCA
          (1, 5),                           # max_depth of GBR
          (10**-5, 10**-1, "log-uniform"),  # learning_rate of GBR
          (1, 20),                          # max_features of GBR
          (2, 30),                          # min_samples_split of GBR
          (1, 30)]                          # min_samples_leaf of GBR


pipe_x0 = [13].extend(x0)  # optional starting point
```

The updated objective function (which I've named `objective_pipe`) is very similar to the objective function that we've had before, except now there are two places where parameters get set--we're changing the parameters of both `pca` and `reg` at the same time.


```python
def objective_pipe(params):
    n_components, max_depth,\
    learning_rate, max_features,\
    min_samples_split, min_samples_leaf = params
    
    # set PCA n_components parameter
    pipe.set_params(pca__n_components=n_components)

    # set decision tree classifier parameters
    pipe.set_params(reg__max_depth=max_depth,
                   reg__learning_rate=learning_rate,
                   reg__max_features=n_components,
                   reg__min_samples_split=min_samples_split,
                   reg__min_samples_leaf=min_samples_leaf)

    error = -np.mean(cross_val_score(pipe, X, y, cv=5, n_jobs=-1, scoring="mean_absolute_error"))
    return error
```

### Optimize all the things, again!

Again, things are analagous with the example above, except we're running with the pipeline-specific objective function and parameter options.


```python
import warnings
warnings.filterwarnings("ignore") # this minimize call issues a lot of warnings--quiet them
                                  # associated scikit-learn issue #6746

pipe_res_gp = gp_minimize(objective_pipe, pipe_space, x0=pipe_x0, n_calls=50, random_state=0)
print("Best score=%.4f" % pipe_res_gp.fun)
print("""Best parameters:
    - n_components=%d
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (pipe_res_gp.x[0], pipe_res_gp.x[1],
                                pipe_res_gp.x[2], pipe_res_gp.x[3],
                                pipe_res_gp.x[4], pipe_res_gp.x[5]))
```

    Best score=3.8677
    Best parameters:
        - n_components=11
        - max_depth=3
        - learning_rate=0.099074
        - max_features=14
        - min_samples_split=6
        - min_samples_leaf=24


We've started with a gaussian process algorithm; now add dummy and forest minimization functions to get a survey of the field.


```python
pipe_res_dummy = dummy_minimize(objective_pipe, pipe_space, x0=pipe_x0, n_calls=50, random_state=0)
print("Best score=%.4f" % pipe_res_dummy.fun)
print("""Best parameters:
    - n_components=%d
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (pipe_res_dummy.x[0], pipe_res_dummy.x[1],
                                pipe_res_dummy.x[2], pipe_res_dummy.x[3],
                                pipe_res_dummy.x[4], pipe_res_dummy.x[5]))
```

    Best score=4.3505
    Best parameters:
        - n_components=6
        - max_depth=4
        - learning_rate=0.077229
        - max_features=17
        - min_samples_split=20
        - min_samples_leaf=13



```python
from skopt import forest_minimize
pipe_res_forest = forest_minimize(objective_pipe, pipe_space, x0=pipe_x0, n_calls=50, random_state=0)
print("Best score=%.4f" % pipe_res_forest.fun)
print("""Best parameters:
    - n_components=%d
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (pipe_res_forest.x[0], pipe_res_forest.x[1],
                                pipe_res_forest.x[2], pipe_res_forest.x[3],
                                pipe_res_forest.x[4], pipe_res_forest.x[5]))
```

    Best score=3.8386
    Best parameters:
        - n_components=11
        - max_depth=3
        - learning_rate=0.098992
        - max_features=13
        - min_samples_split=5
        - min_samples_leaf=2



```python
plot_convergence(("gp_optimize", res_gp),
                 ("forest_optimize", res_forest),
                 ("dummy_optimize", res_dummy),
                 ("gp_optimize_w_pca", pipe_res_gp),
                 ("forest_optimize_w_pca", pipe_res_forest),
                 ("dummy_optimize_w_pca", pipe_res_dummy))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f4364a5be10>




![png](hyperparameter-optimization_files/hyperparameter-optimization_31_1.png)


So, interestingly, our PCA pipelines seem to do much worse than the classifiers do all by themselves.  As we said above, there are other problems where PCA might make more sense than it does here.  But now, instead of that being an argument that you'd make based on theoretical reasons, you've actually tried it both ways and found empirically that PCA is probably not something you want to use here.
