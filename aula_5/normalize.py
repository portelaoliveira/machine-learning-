import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Para reproducilidade
np.random.seed(13)

df = pd.DataFrame(
    {
        "beta": np.random.beta(4, 2, 900) * 60,  # beta
        "exponencial": np.random.exponential(9, 900),  # expo
        "normal_p": np.random.normal(10, 2, 900),  # platykurtic
        "normal_l": np.random.normal(10, 10, 900),  # leptokurtic
    }
)


col_names = list(df.columns)
# print(col_names)
# print(df.loc[0, "beta"])
# print(df["beta"])

plt.hist(df["beta"])
plt.savefig("imgs/hist_beta.png")
plt.close()

plt.hist(df["normal_l"])
plt.savefig("imgs/hist_exponencial.png")
plt.close()

plt.hist(df["normal_p"])
plt.savefig("imgs/hist_normal_p.png")
plt.close()

plt.hist(df["normal_l"])
plt.savefig("imgs/hist_normal_l.png")
plt.close()

# KDE plot
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title("Distribuições originais")

sns.kdeplot(df["beta"], ax=ax1)
sns.kdeplot(df["exponencial"], ax=ax1)
sns.kdeplot(df["normal_p"], ax=ax1)
sns.kdeplot(df["normal_l"], ax=ax1)
plt.savefig("imgs/dist_originais.png")

# Conhecendo as ferramentas de pré-processamento do scikit learn

# MinMaxScaler - Não usar distribuição normal por conta que mantem a escala

mm = preprocessing.MinMaxScaler()
df_mm = mm.fit_transform(df)

df_mm = pd.DataFrame(df_mm, columns=col_names)
fig, ax2 = plt.subplots(ncols=1, figsize=(10, 8))
ax2.set_title("Distribuições MinMaxScaler")
sns.kdeplot(df_mm["beta"], ax=ax2)
sns.kdeplot(df_mm["exponencial"], ax=ax2)
sns.kdeplot(df_mm["normal_p"], ax=ax2)
sns.kdeplot(df_mm["normal_l"], ax=ax2)
plt.savefig("imgs/MinMaxScaler.png")

# Robust Scaler

r_scaler = preprocessing.RobustScaler()
df_r = r_scaler.fit_transform(df)

df_r = pd.DataFrame(df_r, columns=col_names)

fig, (ax3) = plt.subplots(ncols=1, figsize=(10, 8))
ax3.set_title("Distribuições RobusScaler")

sns.kdeplot(df_r["beta"], ax=ax3)
sns.kdeplot(df_r["exponencial"], ax=ax3)
sns.kdeplot(df_r["normal_p"], ax=ax3)
sns.kdeplot(df_r["normal_l"], ax=ax3)
plt.savefig("imgs/RobusScaler.png")

# Standard Scaler

s_scaler = preprocessing.StandardScaler()
df_s = s_scaler.fit_transform(df)

df_s = pd.DataFrame(df_s, columns=col_names)

fig, (ax4) = plt.subplots(ncols=1, figsize=(10, 8))
ax4.set_title("Distribuições StandardScaler")

sns.kdeplot(df_s["beta"], ax=ax4)
sns.kdeplot(df_s["exponencial"], ax=ax4)
sns.kdeplot(df_s["normal_p"], ax=ax4)
sns.kdeplot(df_s["normal_l"], ax=ax4)
plt.savefig("imgs/StandardScaler.png")

# Normalizer

n_scaler = preprocessing.Normalizer()
df_n = n_scaler.fit_transform(df)

df_n = pd.DataFrame(df_n, columns=col_names)

fig, (ax5) = plt.subplots(ncols=1, figsize=(10, 8))
ax5.set_title("Distribuições Normalizer")

sns.kdeplot(df_n["beta"], ax=ax5)
sns.kdeplot(df_n["exponencial"], ax=ax5)
sns.kdeplot(df_n["normal_p"], ax=ax5)
sns.kdeplot(df_n["normal_l"], ax=ax5)
plt.savefig("imgs/Normalizer.png")
