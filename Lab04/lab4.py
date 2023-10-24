# Doreşti sa deschizi o nouă locaţie fast-food în oraş. Analizând volumul de trafic al locaţiei, aproximezi ca
# numărul de clienţi care ar intra în restaurant umează o distribuţie Poisson de parametru λ = 20 clienţi/oră.
# Timpul de plasare si plată a unei comenzi la o casă urmează o distribuţie normală cu media de 2 minute si
# deviatie standard de 0.5 minute. O staţie de gătit pregateste o comandă intr-un timp descris de o distribuţie
# exponenţială cu media de α minute.
# 1. Definiţi modelul probabilist care sa descrie contextul de mai sus.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

lambda1 = 20 # parametrul distributiei Poisson

medie_comanda = 2 # media timpului in minute
deviatie_comanda = 0.5  

alpha = 1 # media distributiei exponentiale a pregatirii comenzii in minute
 
num_samples = 10000  

num_clienti = stats.poisson.rvs(lambda1, size=num_samples)

timp_comanda = stats.norm.rvs(loc=medie_comanda, scale=deviatie_comanda, size=num_samples)

timp_pregatire = stats.expon.rvs(scale=1/alpha, size=num_samples)

timp_total = timp_comanda + timp_pregatire

timp_total_all_clients = timp_total.sum()

print("Media clientilor care intra in restaurant:", num_clienti.mean())
print("Media timpul total de servire pentru un client (minute):", timp_total.mean())
print("Timpul total de servire pentru clienti (minute):", timp_total_all_clients)

plt.hist(timp_total, bins=40, density=True, label='Timp servire')
plt.xlabel('Timp (minute)')
plt.ylabel('Probabilitate')
plt.legend()
plt.show()
