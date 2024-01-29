import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

BostonHousing = pd.read_csv('BostonHousing.csv')
x_1 = BostonHousing['rm'].values
x_2 = BostonHousing['crim'].values
x_3 = BostonHousing['indus'].values
y = BostonHousing['medv'].values
X = np.column_stack((x_1,x_2,x_3))
X_mean = X.mean(axis=0, keepdims=True)
#pentru a ne face o idee asupra mediilor si dev. standard:
print("Medii si deviatii standard:")
print(X_mean)   
print(y.mean())
print(X.std(axis=0, keepdims=True))
print(y.std())

with pm.Model() as model_mlr:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=3)

        eps = pm.HalfCauchy('ϵ', 5000)
        ν = pm.Exponential('ν', 1/30)
        
        X_shared = pm.MutableData('x_shared',X)
        miu = pm.Deterministic('miu',alpha + pm.math.dot(X_shared, beta))

        medv = pm.Normal('medv', mu=miu, sigma=eps, observed=y)

        idata_mlr = pm.sample(1250, return_inferencedata=True)


az.plot_forest(idata_mlr,hdi_prob=0.95,var_names=['beta'])
plt.show()
print(az.summary(idata_mlr,hdi_prob=0.95,var_names=['beta']))

# din figura se observa ca x2 si x3 au o mica influenta asupra modelului (crim si indus)->beta1 si beta2 sunt mici in comparatie cu beta0; cea mai mare influenta o are x1 (rm)

# setam valori pentru x_shared conform cu mediile datelor din csv
pm.set_data({"x_shared":[[x_1.mean(), x_1.mean(), x_3.mean()]]}, model=model_mlr)

# simulam extrageri din distributia predictiva posterioara
ppc = pm.sample_posterior_predictive(idata_mlr, model=model_mlr)

# selectam valoarea locuintelor
y_ppc = ppc.posterior_predictive['medv'].stack(sample=("chain", "draw")).values

# cautam intervalul de predictie de 50% HDI pentru valoarea locuintelor
az.plot_posterior(y_ppc,hdi_prob=0.5)
plt.show()

#EX2

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#var1
def posterior_grid(grid_points=50, heads_obtinut_la_aruncarea=20):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)

    likelihood = stats.geom.pmf(5 * heads_obtinut_la_aruncarea, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

points = 10
grid, posterior = posterior_grid(points, 20)

plt.plot(grid, posterior, "o-")
plt.yticks([])
plt.xlabel("θ")
plt.show()

MAP_estimate = grid[np.argmax(posterior)]
print("theta maxim: ", MAP_estimate)

#var2
for a in range(1, 5):
        numar_aruncari_pana_la_stema = 5

        h = 1
        t = numar_aruncari_pana_la_stema - h

        x = np.linspace(0, 1, 10)

        true_posterior = stats.geom.pmf(numar_aruncari_pana_la_stema - 1, x)
        plt.plot(x, true_posterior, label="True posterior (geometric)")

        mean_q = {"p": (h + 1) / (h + t + 2)}
        std_q = 1 / np.sqrt(h + t + 2)
        plt.plot(x, stats.norm.pdf(x, mean_q["p"], std_q), label="Quadratic approximation")
        
        theta = sp.symbols("theta")
        posterior = theta * (1 - theta)  (h + t)
        derivative = sp.diff(posterior, theta)
        maxima_theta = sp.solve(derivative, theta)

        print(
            f"Valoarea lui theta care maximizează probabilitatea a posteriori: {maxima_theta}"
        )

        plt.legend(loc=0, fontsize=13)
        plt.title(f"Prima apariție a unei steme: heads = {h}, tails = {t}")
        plt.xlabel("θ", fontsize=14)
        plt.yticks([])
        plt.show()