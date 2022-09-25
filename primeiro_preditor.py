#Run cell
#%%
from tabnanny import verbose
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as mtr
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("rota33642.csv")

df.drop(["direcao", "hora", "equipamento", "data_chegada", "chegadaTimeStamp", "linha"], axis=1, inplace=True)
df.data_partida = pd.to_datetime(df.data_partida)
df["mes"] = df.data_partida.dt.month

df_2 = df[["data_partida", "tempo_viagem"]]
df_2 = df_2.set_index("data_partida")

#%%
df_2 = df_2[(df_2.index > "2019-07-07") & (df_2.index < "2019-07-14")]
df_2.plot(style=".", figsize=(15, 5))

#%%
del df["data_partida"]
previsores = df[["partidaTimeStamp", "dia_semana",  "qtdDiasAno",  "tipo_dia",  "turno_dia",  "hora_dia",  "mes"]]
classe = df["tempo_viagem"]

# print(df.head())
# %%
x_treino, x_teste, y_treino, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
print(df.shape)
print(len(x_treino))
print(len(x_teste))
print(len(y_teste))
print(len(y_treino))

# %%
modelo = xgb.XGBRegressor(n_estimators=1000, random_state=123)
cross_val_score(modelo, x_treino, y_treino).mean()
#modelo.fit(x_treino, y_treino, eval_set=[(x_treino, y_treino), (x_teste, y_teste)], verbose=True)

# %%
modelo_2 = xgb.XGBRegressor(n_estimators=100, random_state=123, max_depth=3)
cross_val_score(modelo, x_treino, y_treino).mean()

# %%
modelo_2.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], verbose=True)
# %%

fi = pd.DataFrame(data=modelo_2.feature_importances_, index=modelo_2.feature_names_in_, columns=["importance"])
# %%
fi.sort_values("importance").plot(kind="barh", title="Importância Dados")

# %%