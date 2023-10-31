# SVM - Support Vector Machines (Algoritmo de classificação)

# A principal ideia do SVM é encontrar um hiperplano ótimo que separe os dados

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, validation_curve

# Um exemplo de banco de dados (dígitos escritos à mão)
digits = datasets.load_digits()
# print(digits.images[1])
# plt.imshow(digits.images[102])
# plt.savefig("imgs/number.png")
# plt.close()

n_samples = len(digits.target)
data = digits.images.reshape((n_samples, -1))

# Vou criar um classificador (vou escolher o SVM)
classificador = svm.SVC(kernel="rbf", gamma=1e-4)

# Divido o dado em 2 conjuntos: 50% para o conjunto de treino e 50 % para o conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)
# Utilizamos o conjunto de treino para "aprender" como classificar
classificador.fit(X_train, y_train)
# E utilizamos o conjunto de teste para prever os rótulos das minhas imagens,
# utilizando o classificador treinado

previsao = classificador.predict(X_test)

# Vamos visualizar algumas imagens treinadas
k = 400
j = int(n_samples * 0.7 + k)
plt.imshow(digits.images[j], cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Rótulo: previsao {previsao[k]} e teste {y_test[k]}")
plt.savefig("imgs/imagem_treinada.png")
plt.close()

# print(metrics.classification_report(y_test, previsao))

Dados = {"y_Actual": y_test, "y_Predicted": previsao}
df = pd.DataFrame(Dados, columns=["y_Actual", "y_Predicted"])
confusion_matrix = pd.crosstab(
    df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
)
# print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.savefig("imgs/confusion_matrix.png")
plt.close()

X = data
y = digits.target

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    svm.SVC(),
    X,
    y,
    param_name="gamma",
    param_range=param_range,
    scoring="accuracy",
    n_jobs=1,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Curva de validação com SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(
    param_range, train_scores_mean, label="Score Treinamento", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range, test_scores_mean, label="Score validação cruzada", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig("imgs/curva_validacao.png")
plt.close()
