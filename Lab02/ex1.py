# EX 1
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# distribuția exponentiala pentru mecanici
lambda1 = 4  
lambda2 = 6  

# Probabilitatea de a fi servit de primul mecanic
P1 = 0.4

# vector pentru timpul de servire al fiecarui client
# numpy.zeros - Return a new array of given shape and type, filled with zeros.
X = np.zeros(1000)

# timpul de servire pentru fiecare client in functie de probabilitatea de a fi servit de primul mecanic
for i in range(1000):
    if np.random.rand() < P1:
        X[i] = stats.expon(scale=1/lambda1).rvs()
    else:
        X[i] = stats.expon(scale=1/lambda2).rvs()

# numpy.mean - Returns the average of the array elements
# numpy.std - Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
media = np.mean(X)
deviatia_std = np.std(X)

print("Media timpului de servire:", media)
print("Deviatia standard a timpului de servire:", deviatia_std)

# Grafic al densitatii distributiei lui X
plt.hist(X, bins=40)
plt.xlabel('Timp de servire')
plt.ylabel('Densitate')
plt.show()
