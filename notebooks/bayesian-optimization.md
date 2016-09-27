
# Bayesian optimization with `skopt`

Gilles Louppe, Manoj Kumar July 2016.


```python
import numpy as np
np.random.seed(777)

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)
```

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
from skopt.acquisition import gaussian_lcb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

res = gp_minimize(f,                  # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  x0=[0.],            # the starting point
                  acq_func="LCB",     # the acquisition function (optional)
                  n_calls=15,         # the number of evaluations of f including at x0
                  n_random_starts=0,  # the number of random initialization points
                  random_state=777)
```

Accordingly, the approximated minimum is found to be:


```python
"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
```




    'x^*=-0.4770, f(x^*)=-0.5625'



For further inspection of the results, attributes of the `res` named tuple provide the following information:

- `x` [float]: location of the minimum.
- `fun` [float]: function value at the minimum.
- `models`: surrogate models used for each iteration.
- `x_iters` [array]: location of function evaluation for each
   iteration.
- `func_vals` [array]: function value for each iteration.
- `space` [Space]: the optimization space.
- `specs` [dict]: parameters passed to the function.


```python
for key, value in sorted(res.items()):
    print(key, "=", value)
    print()
```

    fun = -0.562540232051
    
    func_vals = [ 0.22468304  0.05499527 -0.09338278 -0.09002946 -0.04237854  0.03806005
     -0.20223996 -0.13259817  0.26546289 -0.42315714 -0.35121192  0.14205274
     -0.36350223 -0.56254023  0.06602537]
    
    models = [GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777)]
    
    random_state = <mtrand.RandomState object at 0x7f9cafadeb88>
    
    space = Space([Real(low=-2.0, high=2.0, prior=uniform)])
    
    specs = {'args': {'dimensions': [(-2.0, 2.0)], 'func': <function f at 0x7f9cbc7d1d08>, 'kappa': 1.96, 'verbose': False, 'random_state': 777, 'n_calls': 15, 'acq_func': 'LCB', 'n_restarts_optimizer': 5, 'xi': 0.01, 'n_points': 10000, 'callback': None, 'y0': None, 'base_estimator': GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5),
                 n_restarts_optimizer=0, noise='gaussian', normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=777), 'x0': [0.0], 'acq_optimizer': 'auto', 'n_random_starts': 0}, 'function': 'base_minimize'}
    
    x = [-0.47701228275212976]
    
    x_iters = [[0.0], [-2.0], [1.7079881909995787], [1.7706505677188598], [1.1789622049849457], [1.5429127927759865], [0.9167323209730952], [0.71963658431051836], [-1.0705399564442766], [0.85181693110420054], [-0.53221537323817647], [-0.6101469022520023], [0.87490702813533872], [-0.47701228275212976], [2.0]]
    


Together these attributes can be used to visually inspect the results of the minimization, such as the convergence trace or the acquisition function at the last iteration:


```python
from skopt.plots import plot_convergence
plot_convergence(res)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9cafaf1c18>




![png](bayesian-optimization_files/bayesian-optimization_16_1.png)


Let us visually examine

1. The approximation of the fit gp model to the original function.
2. The acquistion values (The lower confidence bound) that determine the next point to be queried.

At the points closer to the points previously evaluated at, the variance dips to zero.

The first column shows the following:
1. The true function.
2. The approximation to the original function by the gaussian process model
3. How sure the GP is about the function.

The second column shows the acquisition function values after every surrogate model is fit. It is possible that we do not choose the global minimum but a local minimum depending on the minimizer used to minimize the acquisition function.


```python
plt.rcParams["figure.figsize"] = (20, 20)

x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = np.array([f(x_i, noise_level=0.0) for x_i in x])

# Plot first five iterations.
for n_iter in range(5):
    gp = res.models[n_iter]
    curr_x_iters = res.x_iters[: n_iter+1]
    curr_func_vals = res.func_vals[: n_iter+1]

    # Plot true function.
    plt.subplot(5, 2, 2*n_iter+1)
    plt.plot(x, fx, "r--", label="True (unknown)")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([fx - 1.9600 * noise_level, fx[::-1] + 1.9600 * noise_level]),
             alpha=.2, fc="r", ec="None")

    # Plot GP(x) + contours
    y_pred, sigma = gp.predict(x, return_std=True)
    plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma, 
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.2, fc="g", ec="None")

    # Plot sampled points
    plt.plot(curr_x_iters, curr_func_vals,
             "r.", markersize=15, label="Observations")
    plt.title(r"$x_{%d} = %.4f, f(x_{%d}) = %.4f$" % (
              n_iter, res.x_iters[n_iter][0], n_iter, res.func_vals[n_iter]))
    plt.grid()

    if n_iter == 0:
        plt.legend(loc="best", prop={'size': 8}, numpoints=1)

    plt.subplot(5, 2, 2*n_iter+2)
    acq = gaussian_lcb(x, gp)
    plt.plot(x, acq, "b", label="LCB(x)")
    plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

    next_x = np.asarray(res.x_iters[n_iter + 1])
    next_acq = gaussian_lcb(next_x.reshape(-1, 1), gp)
    plt.plot(next_x[0], next_acq, "bo", markersize=10, label="Next query point")
    plt.grid()
    
    if n_iter == 0:
        plt.legend(loc="best", prop={'size': 12}, numpoints=1)

plt.suptitle("Sequential model-based minimization using gp_minimize.", fontsize=20)
plt.show()
```


![png](bayesian-optimization_files/bayesian-optimization_18_0.png)


Finally, as we increase the number of points, the GP model approaches the actual function. The final few points are clustered around the minimum because the GP does not gain anything more by further exploration.


```python
# Plot f(x) + contours
plt.rcParams["figure.figsize"] = (10, 6)
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


![png](bayesian-optimization_files/bayesian-optimization_20_0.png)

