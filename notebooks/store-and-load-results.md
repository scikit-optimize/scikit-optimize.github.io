
# Store and load `skopt` optimization results

Mikhail Pak, October 2016.


```python
import numpy as np
np.random.seed(777)
```

## Problem statement

We often want to store optimization results in a file. This can be useful, for example,

* if you want to share your results with colleagues;
* if you want to archive and/or document your work;
* or if you want to postprocess your results in a different Python instance or on an another computer.

The process of converting an object into a byte stream that can be stored in a file is called _serialization_.
Conversely, _deserialization_ means loading an object from a byte stream.

**Warning:** Deserialization is not secure against malicious or erroneous code. Never load serialized data from untrusted or unauthenticated sources!

## Simple example

We will use the same optimization problem as in the [`bayesian-optimization.ipynb`](https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/bayesian-optimization.ipynb) notebook:


```python
from skopt import gp_minimize

noise_level = 0.1

def obj_fun(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level

res = gp_minimize(obj_fun,            # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  x0=[0.],            # the starting point
                  acq_func="LCB",     # the acquisition function (optional)
                  n_calls=15,         # the number of evaluations of f including at x0
                  n_random_starts=0,  # the number of random initialization points
                  random_state=777)
```

    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "
    /home/ubuntu/scikit-optimize/skopt/optimizer/optimizer.py:399: UserWarning: The objective has been evaluated at this point before.
      warnings.warn("The objective has been evaluated "


As long as your Python session is active, you can access all the optimization results via the `res` object.

So how can you store this data in a file? `skopt` conveniently provides functions `skopt.dump()` and `skopt.load()` that handle this for you. These functions are essentially thin wrappers around the [`joblib`](http://pythonhosted.org/joblib) module's `dump()` and `load()`.

We will now show how to use `skopt.dump()` and `skopt.load()` for storing and loading results.

## Using `skopt.dump()` and `skopt.load()`

For storing optimization results into a file, call the `skopt.dump()` function:


```python
from skopt import dump, load

dump(res, 'result.pkl')
```

And load from file using `skopt.load()`:


```python
res_loaded = load('result.pkl')

res_loaded.fun
```




    -0.17487957729512074



You can fine-tune the serialization and deserialization process by calling `skopt.dump()` and `skopt.load()` with additional keyword arguments. See the `joblib` documentation ([dump](https://pythonhosted.org/joblib/generated/joblib.dump.html) and [load](https://pythonhosted.org/joblib/generated/joblib.load.html)) for the additional parameters.

For instance, you can specify the compression algorithm and compression level (highest in this case):


```python
dump(res, 'result.gz', compress=9)

from os.path import getsize
print('Without compression: {} bytes'.format(getsize('result.pkl')))
print('Compressed with gz:  {} bytes'.format(getsize('result.gz')))
```

    Without compression: 83237 bytes
    Compressed with gz:  22641 bytes


### Unserializable objective functions

Notice that if your objective function is non-trivial (e.g. it calls MATLAB engine from Python), it might be not serializable and `skopt.dump()` will raise an exception when you try to store the optimization results.
In this case you should disable storing the objective function by calling `skopt.dump()` with the keyword argument `store_objective=False`:


```python
dump(res, 'result_without_objective.pkl', store_objective=False)
```

Notice that the entry `'func'` is absent in the loaded object but is still present in the local variable:


```python
res_loaded_without_objective = load('result_without_objective.pkl')

print('Loaded object: ', res_loaded_without_objective.specs['args'].keys())
print('Local variable:', res.specs['args'].keys())
```

    Loaded object:  dict_keys(['dimensions', 'base_estimator', 'n_calls', 'n_random_starts', 'acq_func', 'acq_optimizer', 'x0', 'y0', 'random_state', 'verbose', 'callback', 'n_points', 'n_restarts_optimizer', 'xi', 'kappa', 'n_jobs'])
    Local variable: dict_keys(['func', 'dimensions', 'base_estimator', 'n_calls', 'n_random_starts', 'acq_func', 'acq_optimizer', 'x0', 'y0', 'random_state', 'verbose', 'callback', 'n_points', 'n_restarts_optimizer', 'xi', 'kappa', 'n_jobs'])


## Possible problems

* __Python versions incompatibility:__ In general, objects serialized in Python 2 cannot be deserialized in Python 3 and vice versa.
* __Security issues:__ Once again, do not load any files from untrusted sources.
* __Extremely large results objects:__ If your optimization results object is extremely large, calling `skopt.dump()` with `store_objective=False` might cause performance issues. This is due to creation of a deep copy without the objective function. If the objective function it is not critical to you, you can simply delete it before calling `skopt.dump()`. In this case, no deep copy is created:


```python
del res.specs['args']['func']

dump(res, 'result_without_objective_2.pkl')
```
