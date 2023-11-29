import pymc3 as pm
import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\gabri\\Desktop\\uni\\PMP\\PMP-2023\\Lab08\\Prices.csv')

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])

    y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])

    trace = pm.sample(10000, tune=1000, cores=1, step=pm.Metropolis(vars=[alpha, beta1, beta2, sigma]))

beta_hdi = pm.summary(trace, hdi_prob=0.95, var_names=['beta1', 'beta2'])
print("Estimări de 95% HDI ale parametrilor beta1 și beta2:")
print(beta_hdi)

utility_analysis = {'beta1': trace['beta1'], 'beta2': trace['beta2']}
predictors_utility = {key: "Include zero" if value[0] > 0 and value[1] < 0 else "Exclude zero" for key, value in utility_analysis.items()}
print("\nAnaliza utilității predictorilor:")
print(predictors_utility)

new_data = {'Speed': [33], 'HardDrive': [540]}
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=5000, var_names=['y'], data=new_data)


price_hdi = pm.stats.hpd(post_pred['y'], hdi_prob=0.9)
print("\nIntervalul de 90% HDI pentru prețul așteptat:")
print(price_hdi)

with model:
    post_pred_all = pm.sample_posterior_predictive(trace, samples=5000, vars=[y])

prediction_hdi = pm.stats.hpd(post_pred_all['y'], hdi_prob=0.9)
print("\nIntervalul de 90% HDI pentru intervalul de predicție:")
print(prediction_hdi)

premium_effect = pm.summary(trace, hdi_prob=0.95, var_names=['alpha'])['hdi_95%'].values
premium_effect_analysis = "Include zero" if premium_effect[0] > 0 and premium_effect[1] < 0 else "Exclude zero"
print("\nBonus: Analiza efectului producătorului premium:")
print(f"Intervalul de 95% HDI pentru alpha: {premium_effect}, Rezultat: {premium_effect_analysis}")
