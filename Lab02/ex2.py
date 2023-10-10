import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parametrii distributiilor gamma
a = [4, 4, 5, 5]
b = [3, 2, 2, 3]

# distrubutia exponentiala a latenței
lambda_latenta = 4

# probab ca un client sa fie redirectionat catre un anumit server
P_server = [0.25, 0.25, 0.30, 0.20]

# vector pentru a stoca timpul total de servire X
X = np.zeros(1000)

for i in range(1000):
    # Alegem un server în funcție de probab date
    server = np.random.choice(4, p=P_server)
    
    # timpul de procesare pe serverul ales
    t_proces = stats.gamma(a[server], scale=1/b[server]).rvs()
    
    # latenta dintre client si server 
    latenta = stats.expon(scale=1/lambda_latenta).rvs()
    
    # timp total de servire 
    X[i] = t_proces + latenta

# probab ca timpul total de servire X să fie mai mare de 3 milisecunde
P_3 = np.mean(X > 3)

print("Probabilitatea ca X să fie mai mare de 3 milisecunde:", P_3)

# Grafic al densitatii distributiei lui X
plt.hist(X, bins=40)
plt.xlabel('Timp de servire')
plt.ylabel('Densitate')
plt.show()
