import numpy as np
import pandas as pd

from os import makedirs

from sklearn.utils import check_random_state

from dados import ler_dados, ler_dados_tempo, ler_dados_tempo_altura
from modelos import Algoritmo, construir_modelos
from testes import validacao_cruzada

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

        algoritmos = [construir_modelos(random_state)[i] for i in [0, 4]] # TODO: testing
        for alg in algoritmos:
            classe = alg.nome.split('__')[0]
            makedirs(f'{pasta}/{classe}', exist_ok=True)

            resultado = pd.DataFrame({
                'altura': y,
                'altura_predita': validacao_cruzada(
                    X, y, alg.modelo
                )
            })
            resultado.to_csv(f'{pasta}/{classe}/{alg.nome}.csv', index=None)
