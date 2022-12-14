#Run cell
#%%
import xgboost as xgb
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import RandomizedSearchCV


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

params_xgb = {
         "n_estimators": list(range(100, 1100, 100)), #Number of gradient boosted trees. Equivalent to number of boosting rounds
         "max_depth": list(range(2, 15)),#Maximum tree depth for base learners.
         "min_child_weight": list(range(1, 11)),#Minimum sum of instance weight(hessian) needed in a child.
         "learning_rate": [0.3, 0.2, 0.1, 0.05, 0.01, 0.005],#Boosting learning rate (xgb’s “eta”)
         "gamma": np.arange(0, 0.7, 0.1)#Minimum loss reduction required to make a further partition on a leaf node of the tree.
        }

pprint(params_xgb)


# %%
#Criação do modelo_xgb utilizando RandomizedSearchCV

modelo_xgb = xgb.XGBRegressor(early_stop_rounds = 100)
xgb_rand_search = RandomizedSearchCV(modelo_xgb, params_xgb, scoring="neg_mean_squared_error", n_iter=40, verbose=True, cv=10, n_jobs=-1, random_state=123)
xgb_rand_search.fit(x_treino, y_treino)
modelo_xgb = xgb_rand_search.best_estimator_

pprint(xgb_rand_search.best_params_)


# %%
#Fit do modelo_xgb
modelo_xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino)], verbose=True)


# %%
#Plot do feature importance
fi_xgb = pd.DataFrame(data=modelo_xgb.feature_importances_, index=modelo_xgb.feature_names_in_, columns=["importance"])
fi_xgb.sort_values("importance").plot(kind="barh", title="Importância Dados")


#%%
#Merge com predição
df_teste["predicao_xgb"] = modelo_xgb.predict(x_teste)

df = df.merge(df_teste[["predicao_xgb"]], how="left", left_index=True, right_index=True)


# %%
#Cálculo das métricas
RRSE_xgb = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_xgb"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_xgb = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_xgb"], squared=False) 

print(f"RRSE_xgb: {RRSE_xgb}")
print(f"RMSE_xgb: {RMSE_xgb}")
# %%


