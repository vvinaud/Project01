import pandas as pd
import numpy as np

base = pd.read_csv('plano-saude.csv')
X = base.iloc[:,0].values
y = base.iloc[:,1].values
correlacao = np.corrcoef(X,y)
X = X.reshape(-1,1) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#B0 ponto inicial da reta
regressor.intercept_
#B1
regressor.coef_

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X,regressor.predict(X),color='red')
plt.title("Regressão linear simples")
plt.xlabel("Idade")
plt.ylabel("Custo")

from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()
