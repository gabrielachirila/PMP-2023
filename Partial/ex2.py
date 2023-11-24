import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np



np.random.seed(42)
timp_mediu_asteptare = np.random.normal(loc=10, scale=2, size=100)

with pm.Model() as model:
    miu = pm.Normal('miu', mu=10)
    sigma = pm.HalfNormal('sigma', sigma=2)

    likelihood = pm.Normal('likelihood', mu=miu, sigma=sigma, observed=timp_mediu_asteptare)

    trace = pm.sample(100, tune=100)

az.plot_posterior(trace, var_names=['miu'])

plt.title('Distribuția a posteriori pentru miu')
plt.xlabel('Miu')
plt.ylabel('Densitatea de probabilitate')
plt.show()

