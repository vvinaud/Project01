import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
matrizes = []
resultados = []
erro = []
for indice_treina,indice_teste in kfold.split(previsores,np.zeros(shape=(previsores.shape[0],1))):
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treina],classe[indice_treina])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste],previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste],previsoes))
    resultados.append(precisao)
for c in matrizes:
   erro.append(c[0,1]+c[1,0])
del c

matriz_final = np.mean(matrizes,axis=0)
resultados = np.asarray(resultados)    

print('Média:',resultados.mean())
print("Média MTRZ:",np.mean(matrizes,axis=0))
print("Desvio Padrão:",resultados.std())
print("Desvio P. MTRZ:",np.std(matrizes,axis=0))
print("MAIOR:",resultados.max())
print("MAIOR MTRZ:",np.max(matrizes,axis=0))
print("MENOR:",resultados.min())
print("MENOR MTRZ:",np.min(matrizes,axis=0))
print("MAIOR ERRO:",max(erro),"Index:",erro.index(max(erro)),"MATRIZ:",matrizes[erro.index(max(erro))])
print("MENOR ERRO:",min(erro),"Index:",erro.index(min(erro)),"MATRIZ:",matrizes[erro.index(min(erro))])
