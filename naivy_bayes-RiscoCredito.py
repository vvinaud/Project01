# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
          
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
print('Existem',classificador.class_count_[0],'classes de nome "',classificador.classes_[0].upper(),'" com probabilidade de',classificador.class_prior_[0])
print('Existem',classificador.class_count_[1],'classes de nome "',classificador.classes_[1].upper(),'" com probabilidade de',classificador.class_prior_[1])
print('Existem',classificador.class_count_[2],'classes de nome "',classificador.classes_[2].upper(),'" com probabilidade de',classificador.class_prior_[2])

d1 = input('Histórico: (Boa=0 Baixa=1 Desconhecida=2 Ruim=3):')
d2 = input('Dívida: (Alta=0 Baixa=1):')
d3 = input('Garantias: (Adequada=0 Nenhuma=1):')
d4 = input('Renda: (<15=0 15a35=1 >35=2):')
#teste boa, alta, nenhuma, >35 = 0,0,1,2 
resultado = classificador.predict([[int(d1),int(d2),int(d3),int(d4)]])
print('Resultado do risco do cliente "',resultado[0].upper(),'"')
