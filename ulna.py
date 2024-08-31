from dados import ler_dados
from modelos import construir_modelos
from testes import validacao_cruzada
from sklearn.metrics import r2_score

X, y = ler_dados()
alg = construir_modelos()[0]
print(r2_score(y, validacao_cruzada(X, y, alg.modelo)))