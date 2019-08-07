import Orange

base = Orange.data.Table('credit_data.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base,n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
print(len(base_treinamento))
print(len(base_teste))

classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento,base_teste,[classificador])
print(Orange.evaluation.CA(resultado))

# Base Line Classifier Contagem
from collections import Counter
print(Counter(str(d.get_class()) for d in base_teste))

# Base Line Classifier divisão zeros e uns por total de valores
zero=0
um=0
for d in base_teste: 
  if d[3] == 0: zero += 1 
  else: um += 1
if zero >= um: divisao = zero/len(base_teste)
else: divisao = um/len(base_teste)  
print("Zero =",zero,", Um =",um,", BLClassifier = ",divisao)

