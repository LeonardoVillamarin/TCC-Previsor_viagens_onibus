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

df.drop(["direcao", "hora", "equipamento", "data_chegada", "chegadaTimeStamp", "qtdDiasAno", "linha", "turno_dia"], axis=1, inplace=True)
df.data_partida = pd.to_datetime(df.data_partida)

#%%
df_plot = df[["data_partida", "tempo_viagem"]]
df_plot = df_plot.set_index("data_partida")

df_plot = df_plot[(df_plot.index > "2019-07-07") & (df_plot.index < "2019-07-14")]
df_plot.plot(style=".", figsize=(15, 5))

#%%
shape = int(df.shape[0] * 0.3)
df = df.sort_values(by=["data_partida"], ascending=False)
df_teste = df[:shape]
df_treino = df[shape:]

#%%
del df["data_partida"]
x_teste = df_teste[["partidaTimeStamp", "dia_semana", "tipo_dia", "hora_dia"]]
y_teste = df_teste["tempo_viagem"]
x_treino = df_treino[["partidaTimeStamp", "dia_semana", "tipo_dia", "hora_dia"]]
y_treino = df_treino["tempo_viagem"]

print(df.shape)
print(len(x_treino))
print(len(x_teste))
print(len(y_teste))
print(len(y_treino))

# %%
modelo = xgb.XGBRegressor(n_estimators=1000, random_state=123)
cross_val_score(modelo, x_treino, y_treino).mean()

# %%
modelo_2 = xgb.XGBRegressor(n_estimators=100, random_state=123, max_depth=3)
cross_val_score(modelo, x_treino, y_treino).mean()

# %%
modelo_2.fit(x_treino, y_treino, eval_set = [(x_treino, y_treino), (x_teste, y_teste)], verbose=True)
# %%

fi = pd.DataFrame(data=modelo_2.feature_importances_, index=modelo_2.feature_names_in_, columns=["importance"])
fi.sort_values("importance").plot(kind="barh", title="Importância Dados")

#%%
df_teste["predicao"] = modelo_2.predict(x_teste)

df = df.merge(df_teste[["predicao"]], how="left", left_index=True, right_index=True)

# %%
rrse = np.sqrt(sum((df_teste["tempo_viagem"] - df_teste["predicao"]) ** 2) / sum((df_teste["tempo_viagem"] - np.mean(df_teste["tempo_viagem"])) ** 2))
print(rrse)
mtr.mean_squared_error(df_teste["tempo_viagem"], df_teste["predicao"], squared=False) 

# %%
