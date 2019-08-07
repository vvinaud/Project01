'''
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    resultados = []
    for indice_treina,indice_teste in kfold.split(previsores,np.zeros(shape=(previsores.shape[0],1))):
        #classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        #classificador = RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=0)
        #classificador = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        #classificador = LogisticRegression(solver='lbfgs',random_state=0)
        #classificador = SVC(kernel='rbf',random_state=1,C=2.0,gamma='scale')
        #classificador = MLPClassifier(verbose=True,max_iter=1000,tol=0.00001,solver='adam',
        #                              hidden_layer_sizes=(100),activation='relu',batch_size=200,
        #                              learning_rate_init=0.001)            
        classificador.fit(previsores[indice_treina],classe[indice_treina])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste],previsoes)
'''        

#Código para REGRAS
import pandas as pd
import numpy as np
import Orange   
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
base = Orange.data.Table('credit_data.csv')
resultados30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    resultados = []
    for indice_treina,indice_teste in kfold.split(base,np.zeros(shape=(previsores.shape[0],1))):
        cn2_learner = Orange.classification.rules.CN2Learner()
        classificador = cn2_learner(base[indice_treina])
        previsoes = classificador(base[indice_teste])
        precisao = accuracy_score(base.Y[indice_teste],previsoes)

        resultados.append(precisao)
    resultados = np.asarray(resultados)
    media = resultados.mean()
    resultados30.append(media)
resultados30 = np.asarray(resultados30) 
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.',','))


