import numpy as np
import matplotlib.pyplot as plt

# Problemas lineares - Regressão Linear

# Equações Normais - Mínimos quadrados

x = np.linspace(-100, 100, 550)
y = 5 * x**3 + 2 * x**2 + 3 * x

noise = 1e6 * np.random.normal(size=len(x))
y2 = y + noise

plt.scatter(x, y2)
plt.plot(x, y, "r")
plt.savefig("imgs/NorEq.png")
plt.close()

A = np.array([x**3, x**2, x, np.ones(len(x))]).T
# print(A.shape)

# Vamos resolver o sistema de equações normais

C = np.linalg.pinv(A.T @ A) @ A.T @ y2  # Pseúdo inversa
# print(C)

y_prev = C[0] * x**3 + C[1] * x**2 + C[2] * x + C[3]
plt.scatter(x, y2, color="hotpink")
plt.plot(x, y, "y", label="Original")
plt.plot(x, y_prev, "k", label="Previsto")
plt.legend()
plt.savefig("imgs/NorEq_prev.png")
plt.close()
