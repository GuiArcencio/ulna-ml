from os import makedirs
from pathlib import Path

import pandas as pd
import numpy as np

from tqdm import tqdm

from dados import ler_dados, ler_dados_tempo, ler_dados_tempo_altura
from modelos import construir_modelos

from sklearn.base import clone, RegressorMixin
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import check_random_state

def validacao_cruzada(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        modelo: RegressorMixin
    ) -> np.ndarray:
    loo = LeaveOneOut()
    y_pred = np.empty_like(y)

    loo_iterator = tqdm(
        loo.split(X),
        total=len(y),
        ncols=50,
        leave=False,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )
    for indices_treino, indices_teste in loo_iterator:
        X_treino = X.iloc[indices_treino]
        X_teste = X.iloc[indices_teste]
        y_treino = y[indices_treino]

        modelo_sob_teste = clone(modelo)
        modelo_sob_teste.fit(X_treino, y_treino)
        y_pred[indices_teste] = modelo_sob_teste.predict(X_teste)

    return y_pred

def testar_modelos(
        coleta: str = 'paq',
        limpeza: bool = True,
        no_tempo: bool = False,
        usar_altura_como_preditor: bool = False,
        random_state: int | np.random.RandomState | None = None
    ) -> None:
        random_state = check_random_state(random_state)
        NOME_COLETA = {
            'paq': 'paquimetro',
            'pap': 'papel'
        }

        if no_tempo:
            if usar_altura_como_preditor:
                X, y = ler_dados_tempo_altura(limpeza)
                pasta = f'resultados/tempo_altura'
            else:
                X, y = ler_dados_tempo(coleta, limpeza)
                pasta = f'resultados/tempo/{NOME_COLETA[coleta]}'
        else:
            X, y = ler_dados(coleta, limpeza)
            pasta = f'resultados/tabular/{NOME_COLETA[coleta]}'

        algoritmos = construir_modelos(random_state)
        for i, alg in enumerate(algoritmos):
            print(f'CV: {alg.nome} ({i+1}/{len(algoritmos)})')
            classe = alg.nome.split('__')[0]

            if not Path(f'{pasta}/{classe}/{alg.nome}.csv').exists():
                makedirs(f'{pasta}/{classe}', exist_ok=True)

                resultado = pd.DataFrame({
                    'altura': y,
                    'altura_predita': validacao_cruzada(
                        X, y, alg.modelo
                    )
                })
                resultado.to_csv(f'{pasta}/{classe}/{alg.nome}.csv', index=None)

