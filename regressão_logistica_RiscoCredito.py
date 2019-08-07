import pandas as pd

base = pd.read_csv('risco_credito2.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
ctgm = 0
while ctgm < previsores.shape[1]:
    previsores[:,ctgm] = labelencoder.fit_transform(previsores[:,ctgm])
    ctgm += 1

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(solver='lbfgs')
classificador.fit(previsores, classe)
print(classificador.intercept_)
print(classificador.coef_)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
cA = [0,0,1,2]
cB = [3, 0, 0, 0]
print("Cliente A:",classificador.predict([cA]),classificador.predict_proba([cA]))
print("Cliente B:",classificador.predict([cB]),classificador.predict_proba([cB]))

del ctgm,cA,cB