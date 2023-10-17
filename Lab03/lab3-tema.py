from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

# Defining individual CPDs.
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]]) # C=0 NU are loc cutremur, C=1 are loc cutremur
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.03], [0.01, 0.97]], evidence=['C'], evidence_card=[2]) # I=0 NU are loc incendiu, I=1 are loc incendiu

# The CPD for A is defined using the conditional probabilities based on C and I
cpd_a = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.9999, 0.05, 0.98, 0.02], 
                           [0.0001, 0.95, 0.02, 0.98]],
                  evidence=['C', 'I'],
                  evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_c, cpd_i, cpd_a)

# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
infer = VariableElimination(model)

# prob avut loc un cutremur, dat fiind că alarma de incendiu s-a declanșat
result = infer.query(variables=['C'], evidence={'A': 1})
print(result)
prob_cutremur = result.values[1]
print("Probabilitatea ca a avut loc un cutremur, dat fiind ca alarma de incendiu s-a declansat:", prob_cutremur)

# prob a avut loc un incendiu, fara ca alarma de incendiu sa se activeze
result = infer.query(variables=['I'], evidence={'A': 0})
print(result)
prob_incendiu = result.values[1]
print("Probabilitatea ca a avut loc un incendiu, fara ca alarma de incendiu sa se fi activat:", prob_incendiu)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

