import numpy as np
import pandas as pd
import pymc as pm

data = pd.read_csv('C:\\Users\\gabri\\Desktop\\uni\\PMP\\PMP-2023\\Lab08\\Prices.csv')

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, tau=1/10**2)
    beta_speed = pm.Normal('beta_speed', mu=0, tau=1/10**2)
    beta_hard_drive = pm.Normal('beta_hard_drive', mu=0, tau=1/10**2)
    beta_ram = pm.Normal('beta_ram', mu=0, tau=1/10**2)
    beta_premium = pm.Normal('beta_premium', mu=0, tau=1/10**2)
    sigma = pm.HalfNormal('sigma', tau=1)

    mu = alpha + beta_speed * data['Speed'] + beta_hard_drive * data['HardDrive'] + beta_ram * data['Ram'] + beta_premium * (data['Premium'] == 'yes')

    y = pm.Normal('y', mu=mu, tau=1/sigma**2, observed=data['Price'])

if __name__ == '__main__':
    with model:
        trace = pm.sample(2000, tune=1000)
        
    print(pm.summary(trace))