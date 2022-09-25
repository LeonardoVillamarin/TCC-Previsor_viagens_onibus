#Run cell
#%%
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as mtr

df = pd.read_csv("rota33642.csv")

df.drop(["direcao", "hora", "equipamento", "data_chegada", "partidaTimeStamp", "chegadaTimeStamp", "linha"], axis=1, inplace=True)
df.data_partida = pd.to_datetime(df.data_partida)
df["mes"] = df.data_partida.dt.month

df_2 = df[["data_partida", "tempo_viagem"]]
df_2 = df_2.set_index("data_partida")

#%%
df_2 = df_2[(df_2.index > "2019-07-07") & (df_2.index < "2019-07-14")]
df_2.plot(style=".", figsize=(15, 5))

# print(df.head())
# %%
