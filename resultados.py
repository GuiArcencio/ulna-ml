from os import listdir

import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error

def main():
    MODELOS = ['dt', 'knn', 'linear', 'rf', 'svm', 'xgb_linear', 'xgb_tree']

    def calcular_medida(medida, arquivo, *args, **kwargs):
        resultado = pd.read_csv(arquivo)
        return medida(resultado['altura'], resultado['altura_predita'], *args, **kwargs)

    print('ALTURA COMO PREDITOR\n')
    df = {
        'Modelo': list(),
        'RMSE': list(),
        'R2': list(),
        'MAPE': list()
    }
    for modelo in MODELOS:
        pasta = f'resultados/tempo_altura/{modelo}'

        resultados = listdir(pasta)
        erros = np.array([calcular_medida(root_mean_squared_error, f'{pasta}/{arquivo}') for arquivo in resultados])
        melhor = np.argmin(erros)

        caminho = f'{pasta}/{resultados[melhor]}'
        df['Modelo'].append(resultados[melhor].split('.')[0])
        df['RMSE'].append(calcular_medida(root_mean_squared_error, caminho))
        df['R2'].append(calcular_medida(r2_score, caminho))
        df['MAPE'].append(calcular_medida(mean_absolute_percentage_error, caminho)*100)

    print(pd.DataFrame(df).to_string(index=False))

    print('\n-----------\n')
    print('PREDIÇÃO NO TEMPO - PAQUÍMETRO\n')
    df = {
        'Modelo': list(),
        'RMSE': list(),
        'R2': list(),
        'MAPE': list()
    }
    for modelo in MODELOS:
        pasta = f'resultados/tempo/paquimetro/{modelo}'

        resultados = listdir(pasta)
        erros = np.array([calcular_medida(root_mean_squared_error, f'{pasta}/{arquivo}') for arquivo in resultados])
        melhor = np.argmin(erros)

        caminho = f'{pasta}/{resultados[melhor]}'
        df['Modelo'].append(resultados[melhor].split('.')[0])
        df['RMSE'].append(calcular_medida(root_mean_squared_error, caminho))
        df['R2'].append(calcular_medida(r2_score, caminho))
        df['MAPE'].append(calcular_medida(mean_absolute_percentage_error, caminho)*100)

    print(pd.DataFrame(df).to_string(index=False))

    print('\n-----------\n')
    print('PREDIÇÃO TABULAR - PAQUÍMETRO\n')
    df = {
        'Modelo': list(),
        'RMSE': list(),
        'R2': list(),
        'MAPE': list()
    }
    for modelo in MODELOS:
        pasta = f'resultados/tabular/paquimetro/{modelo}'

        resultados = listdir(pasta)
        erros = np.array([calcular_medida(root_mean_squared_error, f'{pasta}/{arquivo}') for arquivo in resultados])
        melhor = np.argmin(erros)

        caminho = f'{pasta}/{resultados[melhor]}'
        df['Modelo'].append(resultados[melhor].split('.')[0])
        df['RMSE'].append(calcular_medida(root_mean_squared_error, caminho))
        df['R2'].append(calcular_medida(r2_score, caminho))
        df['MAPE'].append(calcular_medida(mean_absolute_percentage_error, caminho)*100)

    print(pd.DataFrame(df).to_string(index=False))

    print('\n-----------\n')
    print('PREDIÇÃO NO TEMPO - PAPEL\n')
    df = {
        'Modelo': list(),
        'RMSE': list(),
        'R2': list(),
        'MAPE': list()
    }
    for modelo in MODELOS:
        pasta = f'resultados/tempo/papel/{modelo}'

        resultados = listdir(pasta)
        erros = np.array([calcular_medida(root_mean_squared_error, f'{pasta}/{arquivo}') for arquivo in resultados])
        melhor = np.argmin(erros)

        caminho = f'{pasta}/{resultados[melhor]}'
        df['Modelo'].append(resultados[melhor].split('.')[0])
        df['RMSE'].append(calcular_medida(root_mean_squared_error, caminho))
        df['R2'].append(calcular_medida(r2_score, caminho))
        df['MAPE'].append(calcular_medida(mean_absolute_percentage_error, caminho)*100)

    print(pd.DataFrame(df).to_string(index=False))

    print('\n-----------\n')
    print('PREDIÇÃO TABULAR - PAPEL\n')
    df = {
        'Modelo': list(),
        'RMSE': list(),
        'R2': list(),
        'MAPE': list()
    }
    for modelo in MODELOS:
        pasta = f'resultados/tabular/papel/{modelo}'

        resultados = listdir(pasta)
        erros = np.array([calcular_medida(root_mean_squared_error, f'{pasta}/{arquivo}') for arquivo in resultados])
        melhor = np.argmin(erros)

        caminho = f'{pasta}/{resultados[melhor]}'
        df['Modelo'].append(resultados[melhor].split('.')[0])
        df['RMSE'].append(calcular_medida(root_mean_squared_error, caminho))
        df['R2'].append(calcular_medida(r2_score, caminho))
        df['MAPE'].append(calcular_medida(mean_absolute_percentage_error, caminho)*100)

    print(pd.DataFrame(df).to_string(index=False))


if __name__ == '__main__':
    main()