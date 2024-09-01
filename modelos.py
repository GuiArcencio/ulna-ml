import numpy as np

from dataclasses import dataclass

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

from sklearn.base import RegressorMixin
from sklearn.utils import check_random_state
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

@dataclass
class Algoritmo:
    nome: str
    modelo: RegressorMixin

def construir_modelos(random_state: int | np.random.RandomState | None = None) -> list[Algoritmo]:
    modelos = list()
    random_state = check_random_state(random_state)

    # Árvores de decisão
    for criterio in ['squared_error', 'friedman_mse']:
        modelo = make_pipeline(
            SimpleImputer(strategy='mean'),
            DecisionTreeRegressor(
                criterion=criterio,
                random_state=random_state
            )
        )
        nome = f'dt__{criterio}'
        modelos.append(Algoritmo(nome, modelo))

    # k-Vizinhos
    for numero_vizinhos in [1, 5, 10]:
        for metrica in ['cityblock', 'euclidean', 'cosine']:
            for pesos in ['uniform', 'distance']:
                modelo = make_pipeline(
                    SimpleImputer(strategy='mean'),
                    StandardScaler(),
                    KNeighborsRegressor(
                        n_neighbors=numero_vizinhos,
                        metric=metrica,
                        weights=pesos,
                        n_jobs=-1
                    )
                )
                nome = f'knn__{numero_vizinhos}__{metrica}__{pesos}'
                modelos.append(Algoritmo(nome, modelo))

    # Support vector machines
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for C in [0.1, 1, 10, 100, 1000]:
            modelo = make_pipeline(
                SimpleImputer(strategy='mean'),
                StandardScaler(),
                SVR(
                    kernel=kernel,
                    C=C,
                )
            )
            nome = f'svm__{kernel}__{C}'
            modelos.append(Algoritmo(nome, modelo))

    # Florestas aleatórias
    for n_arvores in [10, 25, 50, 100]:
        for criterio in ['squared_error', 'friedman_mse']:
            for colunas_usadas in [0.5, 1.0]:
                modelo = make_pipeline(
                    SimpleImputer(strategy='mean'),
                    RandomForestRegressor(
                        n_estimators=n_arvores,
                        criterion=criterio,
                        max_features=colunas_usadas,
                        random_state=random_state,
                        n_jobs=-1
                    )
                )
                nome = f'rf__{n_arvores}__{criterio}__{colunas_usadas}'
                modelos.append(Algoritmo(nome, modelo))
    
    # Regressão linear
    modelo = make_pipeline(
        SimpleImputer(strategy='mean'),
        RidgeCV()
    )
    nome = 'linear'
    modelos.append(Algoritmo(nome, modelo))

    # XGBoost - Árvores
    for n_modelos in [50, 100, 200]:
        for eta in [0.1, 0.01, 0.001]:
            for profundidade_maxima in [3, 5, 7]:
                for subsample in [0.5, 0.75, 1.0]:
                    modelo = make_pipeline(
                        SimpleImputer(strategy='mean'),
                        XGBRegressor(
                            n_estimators=n_modelos,
                            learning_rate=eta,
                            max_depth=profundidade_maxima,
                            subsample=subsample,
                            random_state=random_state,
                            n_jobs=-1
                        )
                    )
                    nome = f'xgb_tree__{n_modelos}__{eta}__{profundidade_maxima}__{subsample}'
                    modelos.append(Algoritmo(nome, modelo))

    # XGBoost - Lineares
    for n_modelos in [50, 100, 200]:
        for selecionador in ['cyclic', 'random']:
            modelo = make_pipeline(
                SimpleImputer(strategy='mean'),
                XGBRegressor(
                    n_estimators=n_modelos,
                    booster='gblinear',
                    feature_selector=selecionador,
                    random_state=random_state,
                    n_jobs=-1
                )
            )
            nome = f'xgb_linear__{n_modelos}__{selecionador}'
            modelos.append(Algoritmo(nome, modelo))

    return modelos