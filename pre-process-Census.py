# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:53:49 2019

@author: IV-NOTE
"""
# Chama arquivo
import pandas as pd
base = pd.read_csv('census.csv')
# Separa previsores e classe
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values
# LabelEncoder(coloca um número igual para classificar cada string)
# OneHotEncoder(faz o array 0 ou 1 para não prejudicar os cálculos e colocar as strings equânimes) 
# StandardScaler(faz o escalonamento utilizando a distância euclidiana)
# ColumnTrasformer (Altera somente os campos das colunas requeridas)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])
# Alterando a classe para 0 e 1
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)
# retirando as linhdas abaixo aumenta a propabilidade para 80%
# onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
# previsores = onehotencoder.fit_transform(previsores).toarray()
# Alterando os previsores para valores escalonados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
# Treinamento e Teste
from sklearn.model_selection import train_test_split
contador = 0.1
melhores = []
while contador < 0.3:
 previsores_training,previsores_test,classe_training,classe_test=train_test_split(previsores, classe, test_size = contador, random_state = 0)
 # Com a base de dados faz treinamento e teste
 from sklearn.naive_bayes import GaussianNB
 classificador = GaussianNB()
 classificador.fit(previsores_training, classe_training)
 preview = classificador.predict(previsores_test)
 # Faz a matriz de confusão 
 from sklearn.metrics import  confusion_matrix, accuracy_score
 if accuracy_score(classe_test, preview) >= 0.8:
     melhores.append([accuracy_score(classe_test, preview),contador,confusion_matrix(classe_test, preview)])
 contador += 0.01
print(max(melhores))
