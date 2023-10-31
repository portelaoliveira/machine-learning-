import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filter(dados_sismicos):
    # Suponha que você tenha um dado sismológico em uma lista chamada "dados_sismicos"
    # Você também precisa conhecer a taxa de amostragem (sampling rate) do sinal

    # Defina a taxa de amostragem (exemplo: 1000 Hz)
    sampling_rate = 1000.0

    # Defina os limites de frequência para o filtro passa-baixa (em Hz)
    frequencies = [0.8, 2.0]

    # Calcule a ordem do filtro com base na taxa de amostragem
    nyquist = 0.5 * sampling_rate
    lowcut = frequencies[0] / nyquist
    highcut = frequencies[1] / nyquist

    # Calcule a ordem do filtro usando a função butter
    order = 4  # Ajuste a ordem do filtro conforme necessário

    # Projete o filtro passa-baixa Butterworth
    b, a = signal.butter(order, [lowcut, highcut], btype="band")

    # Aplique o filtro aos dados sismológicos
    filtered_data = signal.lfilter(b, a, dados_sismicos)

    return filtered_data


# # Plote os dados originais e filtrados
# plt.figure(figsize=(12, 6))
# plt.plot(dados_sismicos, label="Dados Originais", alpha=0.7)
# plt.plot(filtered_data, label="Dados Filtrados", color="red")
# plt.xlabel("Tempo (amostras)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)
# plt.title("Filtro Passa-Baixa")
# plt.show()
