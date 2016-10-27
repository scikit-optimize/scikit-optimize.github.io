
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
                  random_state=777,
                  noise=0.1**2)
```

Accordingly, the approximated minimum is found to be:


```python
"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
```




    'x^*=-0.3390, f(x^*)=-0.9066'



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

    fun = -0.906643745289
    
    func_vals = [ 0.22468304  0.05499527 -0.09338278 -0.09172426  0.00150194  0.05658501
      0.12220788  0.85973216  0.09839041  0.25529044 -0.00599222 -0.61164474
     -0.90664375 -0.84707683 -0.84074289]
    
    models = [GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=0.01),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777)]
    
    random_state = <mtrand.RandomState object at 0x7f5c59aab3a8>
    
    space = Space([Real(low=-2.0, high=2.0, prior=uniform, transform=normalize)])
    
    specs = {'function': 'base_minimize', 'args': {'dimensions': [Real(low=-2.0, high=2.0, prior=uniform, transform=normalize)], 'xi': 0.01, 'y0': None, 'n_random_starts': 0, 'acq_optimizer': 'auto', 'n_points': 10000, 'n_calls': 15, 'kappa': 1.96, 'n_restarts_optimizer': 5, 'acq_func': 'LCB', 'x0': [0.0], 'base_estimator': GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                 kernel=1**2 * Matern(length_scale=1, nu=2.5),
                 n_restarts_optimizer=0, noise=0.010000000000000002,
                 normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=777), 'callback': None, 'verbose': False, 'func': <function f at 0x7f5c6679dea0>, 'random_state': 777}}
    
    x = [-0.33900924076046079]
    
    x_iters = [[0.0], [-2.0], [1.7079881909995787], [1.8517089381176524], [2.0], [-1.1915279339683462], [1.2957241148400129], [0.40149974265149657], [-1.5085932154601376], [-0.81650922365031198], [-1.2334758759137854], [-0.20192500033697414], [-0.33900924076046079], [-0.20433782410853585], [-0.30271204061909707]]
    


Together these attributes can be used to visually inspect the results of the minimization, such as the convergence trace or the acquisition function at the last iteration:


```python
from skopt.plots import plot_convergence
plot_convergence(res)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5c59ac5cc0>




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
x_gp = res.space.transform(x.tolist())

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
    y_pred, sigma = gp.predict(x_gp, return_std=True)
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
    acq = gaussian_lcb(x_gp, gp)
    plt.plot(x, acq, "b", label="LCB(x)")
    plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

    next_x = res.x_iters[n_iter + 1]
    next_acq = gaussian_lcb(res.space.transform([next_x]), gp)
    plt.plot(next_x, next_acq, "bo", markersize=10, label="Next query point")
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
x_gp = res.space.transform(x.tolist())

fx = [f(x_i, noise_level=0.0) for x_i in x]
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx], 
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")

# Plot GP(x) + concours
gp = res.models[-1]


y_pred, sigma = gp.predict(x_gp, return_std=True)

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
acq = gaussian_lcb(x_gp, gp)
plt.plot(x, acq, "b", label="LCB(x)")
next_x = np.argmin(acq)
plt.plot([x[next_x]], [acq[next_x]], "b.", markersize=15, label="Next query point")

plt.title(r"$x^* = %.4f, f(x^*) = %.4f$" % (res.x[0], res.fun))
plt.legend(loc="best")
plt.grid()

plt.show()
```


![png](bayesian-optimization_files/bayesian-optimization_20_0.png)

