import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

data = pd.read_csv('Lab09/Admission.csv')

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)

    logit_p = beta0 + beta1 * data['GRE'] + beta2 * data['GPA']

    admit = pm.Bernoulli('admit', logit_p=logit_p, observed=data['Admission'])

with logistic_model:
    trace = pm.sample(2000, tune=1000, cores=1)

pm.plot_posterior(trace, var_names=['beta0', 'beta1', 'beta2'])
plt.show()
