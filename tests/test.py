from datetime import datetime, timedelta
from obspy import UTCDateTime as utc


def hourly_intervals(starttime, endtime):
    # Converter as strings de entrada em objetos datetime
    start_datetime = datetime.fromisoformat(starttime)
    end_datetime = datetime.fromisoformat(endtime)

    # Inicializar a lista de intervalos
    intervalos = []

    # Definir o intervalo de uma hora
    intervalo_hora = timedelta(hours=1)

    # Iniciar o loop para gerar os intervalos
    current_datetime = start_datetime
    while current_datetime < end_datetime:
        intervalos.append(current_datetime.isoformat())
        current_datetime += intervalo_hora

    return intervalos


# Exemplo de uso da função
starttime = "2023-05-01T00:00:00.000000"
endtime = "2023-05-01T23:00:00.000000"
lista_de_intervalos = hourly_intervals(starttime, endtime)

# for intervalo in lista_de_intervalos:
#     print(intervalo)

# Agora, percorra a lista e imprima starttime e endtime de cada intervalo
for intervalo in lista_de_intervalos:
    end_of_interval = utc(intervalo) + timedelta(hours=1)
    # print(utc(intervalo) + timedelta(hours=1))
    print("starttime:", utc(intervalo).isoformat())
    print("endtime:", utc(end_of_interval).isoformat())
    # print("endtime:", str("%02d" % utc(end_of_interval.isoformat()).hour))
    # print("endtime:", f"{utc(end_of_interval.isoformat()).hour:02d}")

    # break
