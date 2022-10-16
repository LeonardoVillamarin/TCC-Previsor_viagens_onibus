#Run cell
#%%
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

#%%
#Leitura da base de dados e conversão datetime

df = pd.read_csv("rota33642.csv")

df.data_partida = pd.to_datetime(df.data_partida)

print(df.head())


#%%
#Gráfico de dispersão e correlação

df_plot = df[["data_partida", "tempo_viagem"]]
df_plot = df_plot.set_index("data_partida")

plt.figure(figsize=(30, 15))

plt.subplot(2, 1, 1)
plt.scatter(df_plot.index, df_plot["tempo_viagem"])

plt.subplot(2, 1, 2)
plt.scatter(df_plot[(df_plot.index > "2019-07-07") & (df_plot.index < "2019-07-14")].index, df_plot.loc[(df_plot.index > "2019-07-07") & (df_plot.index < "2019-07-14"), "tempo_viagem"])

plt.show()

df.corr()


#%%
#Separação teste e treino

shape = int(df.shape[0] * 0.3)
df = df.sort_values(by=["data_partida"], ascending=False)
df_teste = df[:shape]
df_treino = df[shape:]
print(df_teste)
print(df_treino)


#%%
#Separação previsores da classes
"""data_partida foi desconsiderado pois era cobrado que a data fosse int
data_chegada e chegadaTimeStamp não fazem sentido pois nos dariam a "resposta"
linha e equipamento são apenas identificadores e não interferem nos dados
direcao apenas havia 1 valor (valor = 2)
turno_dia removido pois tem alta correlação com hora_dia"""

x_teste = df_teste[["partidaTimeStamp", "dia_semana", "tipo_dia", "hora_dia", "qtdDiasAno"]]
y_teste = df_teste["tempo_viagem"]
x_treino = df_treino[["partidaTimeStamp", "dia_semana", "tipo_dia", "hora_dia", "qtdDiasAno"]]
y_treino = df_treino["tempo_viagem"]

#Conferência do tamanho dos dataframes
print(df.shape)
print(len(x_treino))
print(len(x_teste))
print(len(y_teste))
print(len(y_treino))


#%%
#Escolha dos parâmetros

params_svr = {
         "C": [0.1, 1, 10, 100, 1000], #Regularization parameter. The strength of the regularization is inversely proportional to C
         "gamma": ["scale", "auto"],#Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        }

pprint(params_svr)


# %%
#Criação do modelo_svr utilizando RandomizedSearchCV

modelo_svr = SVR(kernel = 'rbf')
svr_rand_search = RandomizedSearchCV(modelo_svr, params_svr, scoring="neg_mean_squared_error", n_iter=10, verbose=True, cv=10, n_jobs=-1, random_state=123)
svr_rand_search.fit(x_treino, y_treino)
modelo_svr = svr_rand_search.best_estimator_

pprint(svr_rand_search.best_params_)


# %%
#Fit do modelo_svr
modelo_svr.fit(x_treino, y_treino)

# %%
#Plot do feature importance (duvidoso)

# results = permutation_importance(modelo_svr, x_treino, y_treino, scoring='neg_mean_squared_error')
# print(f'Result: {results}')
# importance = results.importances_mean
# print(f'Importance: {importance}')
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (x_treino.columns[i],v)) # Duvidoso, não sabemos a onde das colunas
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()

#%%
#Merge com predição
df_teste["predicao_svr"] = modelo_svr.predict(x_teste)

df = df.merge(df_teste[["predicao_svr"]], how="left", left_index=True, right_index=True)


# %%
#Cálculo das métricas
RRSE_svr = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_svr"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_svr = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_svr"], squared=False) 

print(f"RRSE_svr: {RRSE_svr}")
print(f"RMSE_svr: {RMSE_svr}")
# %%

