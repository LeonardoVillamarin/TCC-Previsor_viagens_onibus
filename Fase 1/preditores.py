#Run cell
#%%
import xgboost as xgb
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
#Escolha dos parâmetros dos modelos

params_xgb = {
         "n_estimators": list(range(100, 1100, 100)), #Number of gradient boosted trees. Equivalent to number of boosting rounds
         "max_depth": list(range(2, 15)),#Maximum tree depth for base learners.
         "min_child_weight": list(range(1, 11)),#Minimum sum of instance weight(hessian) needed in a child.
         "learning_rate": [0.3, 0.2, 0.1, 0.05, 0.01, 0.005],#Boosting learning rate (xgb’s “eta”)
         "gamma": np.arange(0, 0.7, 0.1)#Minimum loss reduction required to make a further partition on a leaf node of the tree.
        }

pprint(params_xgb)

params_rf = {
         "n_estimators": list(range(100, 1100, 100)),#The number of trees in the forest.
         "bootstrap": [True, False],#Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
         "max_depth": list(range(2, 15)),#The maximum depth of the tree.
         "max_features": ["auto", "sqrt", "log2"],#The number of features to consider when looking for the best split:
         "min_samples_leaf": list(range(1, 11)),#The minimum number of samples required to split an internal node
         "min_samples_split": list(range(2, 11)),#The minimum number of samples required to be at a leaf node.
        }

pprint(params_rf)

params_svr = {
         "C": [0.1, 1, 10, 100, 1000], #Regularization parameter. The strength of the regularization is inversely proportional to C
         "gamma": ["scale", "auto"],#Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        }

pprint(params_svr)

# %%
#Criação dos modelos utilizando RandomizedSearchCV

modelo_xgb = xgb.XGBRegressor(early_stop_rounds = 100)
xgb_rand_search = RandomizedSearchCV(modelo_xgb, params_xgb, scoring="neg_mean_squared_error", n_iter=40, verbose=True, cv=10, n_jobs=-1, random_state=123)
xgb_rand_search.fit(x_treino, y_treino)
modelo_xgb = xgb_rand_search.best_estimator_

pprint(xgb_rand_search.best_params_)

modelo_rf = RandomForestRegressor()
rf_rand_search = RandomizedSearchCV(modelo_rf, params_rf, scoring="neg_mean_squared_error", n_iter=40, verbose=True, cv=10, n_jobs=-1, random_state=123)
rf_rand_search.fit(x_treino, y_treino)
modelo_rf = rf_rand_search.best_estimator_

pprint(rf_rand_search.best_params_)

modelo_svr = SVR(kernel = 'rbf')
svr_rand_search = RandomizedSearchCV(modelo_svr, params_svr, scoring="neg_mean_squared_error", n_iter=10, verbose=True, cv=10, n_jobs=-1, random_state=123)
svr_rand_search.fit(x_treino, y_treino)
modelo_svr = svr_rand_search.best_estimator_

pprint(svr_rand_search.best_params_)

modelo_lr = LinearRegression()


# %%
#Fit dos modelos
modelo_xgb.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino)], verbose=True)
modelo_lr.fit(x_treino, y_treino)
modelo_rf.fit(x_treino, y_treino)
modelo_svr.fit(x_treino, y_treino)

# %%
#Plot dos feature importance de cada modelo
plt.subplot(2,2,1)
fi_xgb = pd.DataFrame(data=modelo_xgb.feature_importances_, index=modelo_xgb.feature_names_in_, columns=["importance"])
fi_xgb.sort_values("importance").plot(kind="barh", title="Importância Dados XGB")

plt.subplot(2,2,2)
coefs_lr = pd.DataFrame(
   modelo_lr.coef_,
   columns=['Coeficiente'], 
   index = x_treino.columns
)
coefs_lr.Coeficiente = coefs_lr.Coeficiente.abs()
coefs_lr['DesvioPadrão'] = x_treino.std(axis=0)
coefs_lr['Importancia'] = (100*coefs_lr['DesvioPadrão']*coefs_lr['Coeficiente'])/coefs_lr['DesvioPadrão']*coefs_lr['Coeficiente'].max()

coefs_lr['Importancia'].plot(kind='barh', title = 'Importância Dados LR')


plt.subplot(2,2,3)
fi_rf = pd.DataFrame(data=modelo_rf.feature_importances_, index=x_treino.columns, columns=["importance"])
fi_rf.sort_values("importance").plot(kind="barh", title="Importância Dados RF")

plt.subplot(2,2,4)
#Feature importance do SVR, duvidoso e sem os nomes
# results = permutation_importance(modelo_svr, x_treino, y_treino, scoring='neg_mean_squared_error')
# print(f'Result: {results}')
# importance = results.importances_mean
# print(f'Importance: {importance}')
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (x_treino.columns[i],v)) # Duvidoso, não sabemos a onde das colunas
# plt.bar([x for x in range(len(importance))], importance)
plt.show()

#%%
#Merge das predições
df_teste["predicao_xgb"] = modelo_xgb.predict(x_teste)
df = df.merge(df_teste[["predicao_xgb"]], how="left", left_index=True, right_index=True)

df_teste["predicao_lr"] = modelo_lr.predict(x_teste)
df = df.merge(df_teste[["predicao_lr"]], how="left", left_index=True, right_index=True)

df_teste["predicao_rf"] = modelo_rf.predict(x_teste)
df = df.merge(df_teste[["predicao_rf"]], how="left", left_index=True, right_index=True)

df_teste["predicao_svr"] = modelo_svr.predict(x_teste)
df = df.merge(df_teste[["predicao_svr"]], how="left", left_index=True, right_index=True)

#df = df.merge(df_teste[["predicao_svr"],["predicao_rf"],["predicao_lr"],["predicao_xgb"]], how="left", left_index=True, right_index=True)

# %%
#Cálculo das métricas
RRSE_xgb = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_xgb"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_xgb = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_xgb"], squared=False) 
print(f"RRSE_xgb: {RRSE_xgb}")
print(f"RMSE_xgb: {RMSE_xgb}")

RRSE_lr = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_lr"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_lr = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_lr"], squared=False) 
print(f"RRSE_lr: {RRSE_lr}")
print(f"RMSE_lr: {RMSE_lr}")

RRSE_rf = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_rf"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_rf = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_rf"], squared=False) 
print(f"RRSE_rf: {RRSE_rf}")
print(f"RMSE_rf: {RMSE_rf}")

RRSE_svr = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao_svr"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
RMSE_svr = mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao_svr"], squared=False) 
print(f"RRSE_svr: {RRSE_svr}")
print(f"RMSE_svr: {RMSE_svr}")

# %%
