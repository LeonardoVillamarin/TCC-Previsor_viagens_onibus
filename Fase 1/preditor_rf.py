#Run cell
#%%
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


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

params_rf = {
         "n_estimators": list(range(100, 1100, 100)),#The number of trees in the forest.
         "bootstrap": [True, False],#Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
         "max_depth": list(range(2, 15)),#The maximum depth of the tree.
         "max_features": ["auto", "sqrt", "log2"],#The number of features to consider when looking for the best split:
         "min_samples_leaf": list(range(1, 11)),#The minimum number of samples required to split an internal node
         "min_samples_split": list(range(2, 11)),#The minimum number of samples required to be at a leaf node.
        }

pprint(params_rf)


# %%
#Criação do modelo_rf utilizando RandomizedSearchCV

modelo_rf = RandomForestRegressor()
rf_rand_search = RandomizedSearchCV(modelo_rf, params_rf, scoring="neg_mean_squared_error", n_iter=40, verbose=True, cv=10, n_jobs=-1, random_state=123)
rf_rand_search.fit(x_treino, y_treino)
modelo_rf = rf_rand_search.best_estimator_

pprint(rf_rand_search.best_params_)


# %%
#Fit do modelo_rf
modelo_rf.fit(x_treino, y_treino)


# %%
#Plot do feature importance
fi_rf = pd.DataFrame(data=modelo_rf.feature_importances_, index=x_treino.columns, columns=["importance"])
fi_rf.sort_values("importance").plot(kind="barh", title="Importância Dados")


#%%
#Merge com predição
df_teste["predicao_rf"] = modelo_rf.predict(x_teste)

df = df.merge(df_teste[["predicao_rf"]], how="left", left_index=True, right_index=True)


# %%
#Cálculo das métricas
RRSE_rf = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_rf"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_rf = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_rf"], squared=False) 

print(f"RRSE_rf: {RRSE_rf}")
print(f"RMSE_rf: {RMSE_rf}")
# %%


