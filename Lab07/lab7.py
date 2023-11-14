import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
from scipy.stats import linregress

df = pd.read_csv("C:/Users/gabri/Desktop/uni/PMP/PMP-2023/Lab07/auto-mpg.csv")

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(inplace=True)

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.5)
plt.title('Relationship between Horsepower and MPG')
plt.xlabel('Horsepower (HP)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.grid(True)
plt.show()

slope, intercept, r_value, p_value, std_err = linregress(df['horsepower'], df['mpg'])

x_values = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
y_values = intercept + slope * x_values

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.5, label='Observed Data')
plt.plot(x_values, y_values, color='blue', label='Linear Regression Line')
plt.title('Linear Regression Line')
plt.xlabel('Horsepower (HP)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.show()

print("Linear Regression Coefficients:")
print("Intercept:", intercept)
print("Slope:", slope)

with pm.Model() as linear_model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = pm.Deterministic('mu', alpha + beta * pm.floatX(df['horsepower'].values))

    sigma = pm.Gamma('sigma', alpha=1, beta=1)
    
    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg'])

    trace = pm.sample(1000, tune=1000, random_seed=42)

print(pm.summary(trace))

pm.plot_posterior(trace, var_names=['alpha', 'beta', 'sigma'], figsize=(12, 6))
plt.show()

posterior_predictive = pm.sample_posterior_predictive(trace, samples=1000, model=linear_model)
hdi_95_ppd = pm.hpd(posterior_predictive['mpg'], hdi_prob=0.95)

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.5, label='Observed Data')
plt.fill_between(df['horsepower'].values, hdi_95_ppd[:, 0], hdi_95_ppd[:, 1], color='orange', alpha=0.3, label='95% HDI PPD')
plt.title('95% HDI for Posterior Predictive Distribution')
plt.xlabel('Horsepower (HP)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.show()

# Modelul de regresie liniară Bayesiană sugerează o relație între caii putere și consumul de combustibil la mașini. 
# Coeficienții estimati pentru interceptare și caii putere oferă o idee despre valoarea inițială și modificarea consumului de combustibil. 