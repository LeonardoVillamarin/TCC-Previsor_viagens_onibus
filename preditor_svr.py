#Run cell
#%%
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from eli5 import show_weights


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

params = {
         "C": [0.1, 1, 10, 100, 1000], 
         "gamma": ["scale", "auto"],
         "kernel": ["rbf", "sigmoid", "poly"]
        }

pprint(params)


# %%
#Criação do modelo_svr utilizando RandomizedSearchCV

modelo_svr = SVR()
svr_rand_search = RandomizedSearchCV(modelo_svr, params, scoring="neg_mean_squared_error", n_iter=30, verbose=True, cv=10, n_jobs=-1, random_state=123)
svr_rand_search.fit(x_treino, y_treino)
modelo_svr = svr_rand_search.best_estimator_

pprint(svr_rand_search.best_params_)


# %%
#Fit do modelo_svr

modelo_svr.fit(x_treino, y_treino)


# %%
#Plot do feature importance

# names = list(x_treino.columns.values)
# print(names)

# imp = list(modelo_svr.coef_[0])

# # imp, names = zip(*sorted(zip(imp,names)))
# print(imp)
# print(names)
# plt.barh(names, imp)
# plt.yticks(range(len(names)), names)
# plt.show()


#%%
#Merge com predição
df_teste["predicao"] = modelo_svr.predict(x_teste)

df = df.merge(df_teste[["predicao"]], how="left", left_index=True, right_index=True)


# %%
#Cálculo das métricas
rrse = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
rmse = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao"], squared=False) 

print(f"RRSE: {rrse}")
print(f"RMSE: {rmse}")
# %%


