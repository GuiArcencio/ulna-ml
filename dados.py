import pandas as pd

def ler_dados(coleta: str = 'paq', limpeza: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    dados = pd.read_csv(f'dados_ulna.csv')
    if limpeza:
        # Data cleaning
        dados = dados.drop(['ulna-paq-9', 'ulna-pap-9', 'peso-9', 'altura-9'], axis=1)
        dados = dados[dados['ulna-paq-6'] > dados['ulna-paq-0']]
        dados = dados[dados['ulna-pap-6'] > dados['ulna-pap-0']]
        dados = dados[dados['peso-6'] > dados['peso-0']]
        dados = dados[dados['altura-6'] > dados['altura-0']]

    dataframe = {
        'sexo': list(),
        'idade': list(),
        f'ulna-{coleta}': list(),
        'peso': list(),
        'altura': list(),
    }

    for _, row in dados.iterrows():
        for mes in ([0, 6] if limpeza else [0, 6, 9]):
            dataframe['sexo'].append(0 if row['sexo'] == 'masculino' else 1)
            dataframe['idade'].append(row['idade-meses'] + mes)
            dataframe[f'ulna-{coleta}'].append(row[f'ulna-{coleta}-{mes}'])
            dataframe['peso'].append(row[f'peso-{mes}'])
            dataframe['altura'].append(row[f'altura-{mes}'])

    dataframe = pd.DataFrame(dataframe)

    X = dataframe.drop(columns=['altura'])
    y = dataframe['altura']

    return X, y

def ler_dados_tempo(coleta: str = 'paq', limpeza: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    dados = pd.read_csv(f'dados_ulna.csv')
    if limpeza:
        # Data cleaning
        dados = dados[dados['ulna-paq-6'] > dados['ulna-paq-0']]
        dados = dados[dados['ulna-pap-6'] > dados['ulna-pap-0']]
        dados = dados[dados['peso-6'] > dados['peso-0']]
        dados = dados[dados['altura-6'] > dados['altura-0']]

    dataframe = {
        'sexo': list(),
        'idade': list(),
        f'ulna-{coleta}-0': list(),
        f'ulna-{coleta}-6': list(),
        'peso-0': list(),
        'peso-6': list(),
        'altura-9': list(),
    }

    for _, row in dados.iterrows():
        dataframe['sexo'].append(0 if row['sexo'] == 'masculino' else 1)
        dataframe['idade'].append(row['idade-meses'])
        dataframe[f'ulna-{coleta}-0'].append(row[f'ulna-{coleta}-0'])
        dataframe[f'ulna-{coleta}-6'].append(row[f'ulna-{coleta}-6'])
        dataframe['peso-0'].append(row[f'peso-0'])
        dataframe['peso-6'].append(row[f'peso-6'])
        dataframe['altura-9'].append(row[f'altura-9'])

    dataframe = pd.DataFrame(dataframe)

    X = dataframe.drop(columns=['altura-9'])
    y = dataframe['altura-9']

    return X, y
