import numpy as np
import pandas as pd
import pymc3 as pm

data = pd.read_csv('C:/Users/gabri/Desktop/uni/PMP/PMP-2023/Lab05/trafic.csv', usecols=['minut', 'nr. masini'])

schimbari_medie = [7 * 60, 8 * 60, 16 * 60, 19 * 60] 

with pm.Model() as model:
    lambda_prior = pm.Gamma("lambda", alpha=1, beta=0.1)

    intervale = []
    valori_lambda = []

    for i in range(1, len(schimbari_medie)):
        start = schimbari_medie[i - 1]
        end = schimbari_medie[i]
        valoare_lambda = pm.Deterministic(f"lambda_{i}", lambda_prior)
        intervale.append((start, end))
        valori_lambda.append(valoare_lambda)

    observatii = []
    for i in range(1, len(intervale) + 1):
        obs = pm.Poisson(f"observatii_{i}", mu=valori_lambda[i - 1], observed=data['nr. masini'][(data['minut'] >= intervale[i - 1][0]) & (data['minut'] < intervale[i - 1][1])].values)
        observatii.append(obs)

    trace = pm.sample(100, cores=1)

intervale_probabile = []
valori_lambda_probabile = []

for i in range(1, len(schimbari_medie)):
    start = schimbari_medie[i - 1]
    end = schimbari_medie[i]
    lambda_samples = trace[f"lambda_{i}"]
    interval_probabil = (start, end)
    valoare_lambda_probabila = lambda_samples.mean()
    intervale_probabile.append(interval_probabil)
    valori_lambda_probabile.append(valoare_lambda_probabila)

print("Intervalele de timp probabile sunt:", intervale_probabile)
print("Valorile probabile ale lui lambda sunt:", valori_lambda_probabile)
