import pandas as pd
import numpy as np

from tqdm import tqdm

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
