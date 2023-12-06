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

    logit_p = pm.math.sigmoid(beta0 + beta1 * data['GRE'] + beta2 * data['GPA'])

    admit = pm.Bernoulli('admit', logit_p=logit_p, observed=data['Admission'])

    trace = pm.sample(2000, tune=1000, cores=1, target_accept=0.95)

pm.plot_posterior(trace, var_names=['beta0', 'beta1', 'beta2'])
plt.show()

with logistic_model:
    ppc = pm.sample_posterior_predictive(trace, samples=500, model=logistic_model)
    y_pred = ppc["admit"].mean(axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(data['GRE'], data['GPA'], c=data['Admission'], cmap='viridis', alpha=0.8)
plt.contour(data['GRE'], data['GPA'], y_pred.reshape(data['GRE'].shape), levels=[0.5], colors='red')
az.plot_hdi(data['GRE'], y_pred, color='red', fill_kwargs={'alpha': 0.2})
plt.xlabel('GRE')
plt.ylabel('GPA')
plt.title('Decision Boundary and HDI')
plt.show()

new_student_data = pd.DataFrame({'GRE': [550], 'GPA': [3.5]})
logit_p_new_student = trace['beta0'] + trace['beta1'] * new_student_data['GRE'] + trace['beta2'] * new_student_data['GPA']
admit_trace_new_student = pm.sample_posterior_predictive(trace, samples=500, model=logistic_model, var_names=['admit'])
prob_admit_new_student = pm.hdi(admit_trace_new_student["admit"], hdi_prob=0.9)

print(f"HDI for the admission probability for a student with GRE 550 and GPA 3.5: {prob_admit_new_student}")

new_student_data_2 = pd.DataFrame({'GRE': [500], 'GPA': [3.2]})
logit_p_new_student_2 = trace['beta0'] + trace['beta1'] * new_student_data_2['GRE'] + trace['beta2'] * new_student_data_2['GPA']
admit_trace_new_student_2 = pm.sample_posterior_predictive(trace, samples=500, model=logistic_model, var_names=['admit'])
prob_admit_new_student_2 = pm.hdi(admit_trace_new_student_2["admit"], hdi_prob=0.9)

print(f"HDI for the admission probability for a student with GRE 500 and GPA 3.2: {prob_admit_new_student_2}")
