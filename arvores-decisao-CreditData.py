import pandas as pd
base = pd.read_csv('credit_data.csv')
# Alterando valores incorretos para valores médios
base.loc[base.age < 0, 'age'] = base.loc[base.age > 0].mean().age
# Separando Previsores de Classes
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values
#Alterando campos vazios para valores médios
from sklearn.preprocessing import Imputer, StandardScaler
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
previsores[:,0:3] = imputer.fit_transform(previsores[:,0:3])
# Criando matrizes 0 e 1
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
# Realizando treinamento(database) e teste(aprendizado)
from sklearn.model_selection import train_test_split
previsores_training,previsores_test,classe_training,classe_test=train_test_split(previsores,classe,test_size=0.25,random_state=0)
# Fazendo Estatística dos valores
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy',random_state=0)
classificador.fit(previsores_training,classe_training)
prevision = classificador.predict(previsores_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_test, prevision)
matriz = confusion_matrix(classe_test, prevision)