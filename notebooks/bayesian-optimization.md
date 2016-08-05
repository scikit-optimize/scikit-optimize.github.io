
# Bayesian optimization with `skopt`

Gilles Louppe, July 2016.


```python
import numpy as np
np.random.seed(777)

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)
```

    /home/travis/miniconda3/envs/testenv/lib/python3.5/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')
    /home/travis/miniconda3/envs/testenv/lib/python3.5/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')


## Problem statement

We are interested in solving $$x^* = \arg \min_x f(x)$$ under the constraints that

- $f$ is a black box for which no closed form is known (nor its gradients);
- $f$ is expensive to evaluate;
- evaluations $y = f(x)$ of may be noisy.

**Disclaimer.** If you do not have these constraints, then there is certainly a better optimization algorithm than Bayesian optimization.

## Bayesian optimization loop

For $t=1:T$:

1. Given observations $(x_i, y_i=f(x_i))$ for $i=1:t$, build a probabilistic model for the objective $f$. Integrate out all possible true functions, using Gaussian process regression.
   
2. optimize a cheap acquisition/utility function $u$ based on the posterior distribution for sampling the next point.
   $$x_{t+1} = \arg \min_x u(x)$$
   Exploit uncertainty to balance exploration against exploitation.
    
3. Sample the next observation $y_{t+1}$ at $x_{t+1}$.

## Acquisition functions

Acquisition functions $\text{u}(x)$ specify which sample $x$ should be tried next:

- Lower confidence bound: $\text{LCB}(x) = \mu_{GP}(x) + \kappa \sigma_{GP}(x)$;
- Probability of improvement: $-\text{PI}(x) = -P(f(x) \geq f(x_t^+) + \kappa) $;
- Expected improvement: $-\text{EI}(x) = -\mathbb{E} [f(x) - f(x_t^+)] $;

where $x_t^+$ is the best point observed so far.

In most cases, acquisition functions provide knobs (e.g., $\kappa$) for
controlling the exploration-exploitation trade-off.
- Search in regions where $\mu_{GP}(x)$ is high (exploitation)
- Probe regions where uncertainty $\sigma_{GP}(x)$ is high (exploration)

## Toy example

Let assume the following noisy function $f$:


```python
noise_level = 0.1

def f(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level
```

**Note.** In `skopt`, functions $f$ are assumed to take as input a 1D vector $x$ represented as an array-like and to return a scalar $f(x)$.


```python
# Plot f(x) + contours
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in x]
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx], 
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")
plt.legend()
plt.grid()
plt.show()
```


![png](bayesian-optimization_files/bayesian-optimization_8_0.png)


Bayesian optimization based on gaussian process regression is implemented in `skopt.gp_minimize` and can be carried out as follows:


```python
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

gp = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed"), 
                              alpha=noise_level**2, random_state=0)

res = gp_minimize(f,                  # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  x0=[0.],            # the starting point
                  acq="LCB",          # the acquisition function (optional)
                  base_estimator=gp,  # a GP estimator (optional)
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=0,  # the number of random initialization points
                  random_state=777)
```

Accordingly, the approximated minimum is found to be:


```python
"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
```




    'x^*=-0.3217, f(x^*)=-0.9259'



For further inspection of the results, attributes of the `res` named tuple provide the following information:

- `x` [float]: location of the minimum.
- `fun` [float]: function value at the minimum.
- `models`: surrogate models used for each iteration.
- `x_iters` [array]: location of function evaluation for each
   iteration.
- `func_vals` [array]: function value for each iteration.
- `space` [Space]: the optimization space.


```python
for key, value in sorted(res.items()):
    print(key, "=", value)
    print()
```

    fun = -0.925941066255
    
    func_vals = [ 0.22468304  0.05499527 -0.09826131 -0.2306291   0.16016027  0.04066236
      0.12350637  0.08315042  0.10237308 -0.39991553  0.128577   -0.64048322
     -0.89628578 -0.92594107 -0.83922154]
    
    models = [GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0)]
    
    random_state = <mtrand.RandomState object at 0x7f9cfc84f438>
    
    space = <skopt.space.Space object at 0x7f9cff0d5208>
    
    specs = {'args': {'kappa': 1.96, 'x0': [0.0], 'n_restarts_optimizer': 5, 'search': 'auto', 'alpha': 1e-09, 'xi': 0.01, 'n_points': 500, 'random_state': 777, 'dimensions': [(-2.0, 2.0)], 'base_estimator': GaussianProcessRegressor(alpha=0.010000000000000002, copy_X_train=True,
                 kernel=Matern(length_scale=1, nu=1.5), n_restarts_optimizer=0,
                 normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=0), 'y0': None, 'func': <function f at 0x7f9cff0c2bf8>, 'n_random_starts': 0, 'n_calls': 15, 'acq': 'LCB'}, 'function': 'gp_minimize'}
    
    x = [-0.32167462923223483]
    
    x_iters = [[0.0], [-2.0], [2.0], [1.0783056681658953], [-1.0607209567449765], [1.5140003732987382], [0.62375407407250694], [1.2243917203159205], [-1.5528926690953295], [-0.53675657487932271], [-0.67998050022698353], [-0.36427313079405427], [-0.34695632908030555], [-0.32167462923223483], [-0.30663049322803215]]
    


Together these attributes can be used to visually inspect the results of the minimization, such as the convergence trace or the acquisition function at the last iteration:


```python
from skopt.plots import plot_convergence
plot_convergence(res)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9cf25d5550>




![png](bayesian-optimization_files/bayesian-optimization_16_1.png)



```python
from skopt.acquisition import gaussian_lcb

# Plot f(x) + contours
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in x]
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx], 
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")

# Plot GP(x) + concours
gp = res.models[-1]
y_pred, sigma = gp.predict(x, return_std=True)

plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma, 
                         (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.2, fc="g", ec="None")

# Plot sampled points
plt.plot(res.x_iters, 
         res.func_vals, 
         "r.", markersize=15, label="Observations")

# Plot LCB(x) + next query point
acq = gaussian_lcb(x, gp)
plt.plot(x, gaussian_lcb(x, gp), "b", label="LCB(x)")
next_x = np.argmin(acq)
plt.plot([x[next_x]], [acq[next_x]], "b.", markersize=15, label="Next query point")

plt.title(r"$x^* = %.4f, f(x^*) = %.4f$" % (res.x[0], res.fun))
plt.legend(loc="best")
plt.grid()

plt.show()
```


![png](bayesian-optimization_files/bayesian-optimization_17_0.png)

