import random
import numpy as np
from scipy import stats
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


sanse_jucator0 = 0
sanse_jucator1 = 0

for i in range(10000):
    p0 = 0
    p1 = 0

    # se arunca o moneda pentru a decide care dintre p0 sau p1 incepe jocul
    moneda = random.random()
    if moneda < 0.5:
        p0 = 1
    else:
        p1 = 1
    
    # in prima runda, in functie de cine a fost ales sa inceapa jocul, se arunca o moneda cu probabilitatea corespunzatoare(stiind ca p1 foloseste o moneda masluita, iar p0 foloseste o moneda normala)
    if p1 == 1:
        stema_moneda1 = stats.binom.rvs(1,2/3)
    elif p0 == 1:
        stema_moneda1 = stats.binom.rvs(1,0.5)

    # numarul de steme obtinute in prima runda
    if stema_moneda1 == 1:
        n = 1
    else:
        n = 0


    # calculam cate steme sunt obtinute in a doua runda stiind ca celalalt jucator arunca moneda de n+1 ori, m este numarul de steme obtinute
    m = 0
    if p1 == 1:
        stema_moneda2 = stats.binom.rvs(1,2/3, size=n+1)
    elif p0 == 1:
        stema_moneda2 = stats.binom.rvs(1,0.5, size=n+1)

    for j in range(n + 1):
        if stema_moneda2[j] == 1:
            m += 1
    
    # daca n >= m, atunci jucatorul din prima runda castiga, altfel jucatorul din a doua runda castiga
    if n >= m:
        if p0 == 1:
            sanse_jucator0 += 1
        elif p1 == 1:
            sanse_jucator1 += 1
    else:
        if p1 == 0:
            sanse_jucator1 += 1
        elif p0 == 0:
            sanse_jucator0 += 1

            
print("Jucatorul 0 are sanse de ", sanse_jucator0 / 10000 * 100, "%")
print("Jucatorul 1 are sanse de ", sanse_jucator1 / 10000 * 100, "%")

# ex2
model = BayesianModel([('StartingPlayer', 'n'), ('n', 'm')])

cpd_starting_player = TabularCPD('StartingPlayer', 2, [[0.5], [0.5]])

cpd_n = TabularCPD('n', 2, [[2/3, 0.5], [1/3, 0.5]], evidence=['StartingPlayer'], evidence_card=[2])

cpd_m = TabularCPD('m', 2, [[2/3, 0.5], [1/3, 0.5]], evidence=['n'], evidence_card=[2])

model.add_cpds(cpd_starting_player, cpd_n, cpd_m)

model.check_model()

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

# ex3
infer = VariableElimination(model)

prob_jucator0_stiind_m = infer.query(variables=['StartingPlayer'], evidence={'m': 1})
print(prob_jucator0_stiind_m)


