import numpy as np

from validacao import testar_modelos

def main() -> None:
    random_state = np.random.RandomState(1156533041)

    for coleta in ['paq', 'pap']:
        for no_tempo in [False, True]:
            str_coleta = 'PAQUÍMETRO' if coleta == 'paq' else 'PAPEL'
            str_tempo = 'NO TEMPO' if no_tempo else 'TABULAR'
            print(f'--- TESTANDO ULNA-{str_coleta} {str_tempo} ---')

            testar_modelos(
                coleta=coleta,
                limpeza=True,
                no_tempo=no_tempo,
                usar_altura_como_preditor=False,
                random_state=random_state
            )

    print(f'--- TESTANDO ALTURA COMO PREDITOR ---')
    testar_modelos(
        limpeza=True,
        no_tempo=True,
        usar_altura_como_preditor=True,
        random_state=random_state
    )

if __name__ == '__main__':
    main()