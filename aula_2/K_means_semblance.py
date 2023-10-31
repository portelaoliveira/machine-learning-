# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans


# Aprendizado de máquina não supervisionada - KMeans (sensível a escala - deixar tudo na mesma escala) e DBSCAN

# Carregando e plotando o painel de semblance
dado = np.loadtxt("data/semblance.ascii")  # Carregamos o dado

nt = 101  # registros em tempo
nv = 120  # velocidades
dado2 = dado.reshape([nv, nt])  # Organizamos o dado em forma de matriz

dt = 0.02  # time shift (s)
dv = 75  # velocity shift (m/s)
fv = 4000  # primeira velocidade (m/s)

# extent=[coluna, coluna, linha, linha]
plt.imshow(dado2.T, aspect="auto", extent=[fv, fv + nv * dv, nt * dt, 0])
plt.xlabel("Velocidades (m/s)")
plt.ylabel("Tempo (s)")
plt.colorbar()
plt.savefig("imgs/semblance.png")
plt.close()

# Vamos mutar os primeiros segundos do painel, que não trazem informação relevante

dado_mute = dado2[:, 10:-1]


plt.imshow(dado_mute.T, aspect="auto", extent=[fv, fv + nv * dv, nt * dt, 11 * dt])
plt.xlabel("Velocidades (m/s)")
plt.ylabel("Tempo (s)")
plt.colorbar()
plt.savefig("imgs/semblance_mute.png")
plt.close()


# Pre-processamento
# Vamos normalizar/escalonar os dados para que as duas dimensões
# (velocidade e tempo) fiquem na mesma escala
def standardise(vector):
    # Padronizando os dados
    media = np.mean(vector)
    desvpad = np.std(vector)
    stand = (vector - media) / desvpad
    return stand


def normalise(vector):
    # Normalizando os dados
    media = np.mean(vector)
    maxi = np.max(vector)
    mini = np.min(vector)
    norma = (vector - media) / (maxi - mini)
    return norma


t = np.arange(11 * dt, nt * dt, dt)
v = np.arange(fv, fv + nv * dv, dv)

times, velocities = np.meshgrid(t, v)

timesN = normalise(times.flatten())
velocitiesN = normalise(velocities.flatten())
# print(times.shape, velocities.shape)

# Vamos selecionar apenas aquelas semblances maiores que um determinado threshold
filtro = dado_mute.flatten() > 0.9
dado_clean = dado_mute.flatten()
# dado_clean[filtro]=0

# print(times.shape, velocities.shape, filtro.shape)

X = np.zeros([len(dado_clean[filtro]), 3])
X[:, 0] = timesN[filtro]
X[:, 1] = velocitiesN[filtro]
X[:, 2] = 0 * normalise(dado_clean[filtro])


plt.scatter(X[:, 1], X[:, 0], c=dado_clean[filtro])
plt.colorbar()
plt.savefig("imgs/semblance_pre-processamento.png")
plt.close()

k = 4
ctrds = np.zeros([k, 3])
colors = cm.rainbow(np.linspace(0, 1, k))  # vou plotar cada centroide com uma cor
kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)
pred_y = kmeans.fit_predict(X)

plt.scatter(X[:, 1], X[:, 0], c=pred_y)
plt.scatter(
    kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c=colors
)
plt.savefig("imgs/semblance_clusters.png")
plt.close()

# E agora, vamos plotar os centroides sobre a imagem da semblance
dado_mute = dado2[:, 10:-1]


plt.imshow(
    dado_mute.T,
    aspect="auto",
    extent=[min(velocitiesN), max(velocitiesN), max(timesN), min(timesN)],
)
plt.xlabel("Velocidades (m/s)")
plt.ylabel("Tempo (s)")
plt.colorbar()

plt.scatter(X[:, 1], X[:, 0], c=pred_y)
plt.scatter(
    kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c=colors
)
plt.savefig("imgs/semblance_centroides.png")
plt.close()
