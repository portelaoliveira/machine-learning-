# SVM - Support Vector Machines (Algoritmo de classificação)

# A principal ideia do SVM é encontrar um hiperplano ótimo que separe os dados

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, validation_curve
