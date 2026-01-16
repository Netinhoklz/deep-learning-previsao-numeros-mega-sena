import pandas as pd
import numpy as np
from collections import Counter

def gerar_relatorio_bottom_8():
    # Configurações baseadas nos notebooks analisados
    # Janela ótima para 8 números conforme mega_sena_resumo_final.ipynb
    JANELA_8_NUMEROS = 800
    ARQUIVO_DADOS = 'Mega-Sena.xlsx'
    
    print(f"============================================================")
    print(f"   RELATÓRIO: NÚMEROS MENOS FREQUENTES (8 DEZENAS)")
    print(f"============================================================")
    print(f"Base de Análise: Últimos {JANELA_8_NUMEROS} concursos (Janela Ótima para 8 números)")
    
    try:
        # Carregar dados
        try:
            df = pd.read_excel(ARQUIVO_DADOS)
        except Exception:
            # Tenta carregar sem depender do openpyxl se der erro, ou avisa
            print(f"Erro ao abrir '{ARQUIVO_DADOS}'. Certifique-se que o arquivo existe.")
            return

        ball_columns = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
        # Garantir que pegamos apenas as colunas de bolas e convertemos para inteiros
        sorteios = np.sort(df[ball_columns].dropna().astype(int).values, axis=1)
        
        total_concursos = len(sorteios)
        print(f"Total de concursos na base: {total_concursos}")
        
        # Selecionar a janela de análise
        if total_concursos < JANELA_8_NUMEROS:
            print(f"AVISO: Base de dados menor que a janela ideal ({JANELA_8_NUMEROS}). Usando todos os {total_concursos} jogos.")
            janela_dados = sorteios
        else:
            janela_dados = sorteios[-JANELA_8_NUMEROS:]
            
        print(f"Analisando intervalo: Concurso {total_concursos - len(janela_dados) + 1} até {total_concursos}")
        
        # Calcular frequência
        # Aplainar a matriz para contar todos os números sorteados na janela
        numeros_todos = janela_dados.flatten()
        contador = Counter(numeros_todos)
        
        # Mapear frequências de 1 a 60 (incluindo os que não saíram, se houver)
        freq_dict = {i: contador.get(i, 0) for i in range(1, 61)}
        
        # Ordenar por frequência CRESCENTE (do que menos saiu para o que mais saiu)
        # Critério de desempate: número da dezena (crescente)
        ordenados = sorted(freq_dict.items(), key=lambda x: (x[1], x[0]))
        
        # Selecionar os 8 primeiros (os que menos saíram)
        bottom_8 = ordenados[:8]
        
        print("\n------------------------------------------------------------")
        print(f" OS 8 NÚMEROS QUE MENOS SAÍRAM (Janela {JANELA_8_NUMEROS})")
        print("------------------------------------------------------------")
        print(f"{'Dezena':^10} | {'Vezes Sorteada':^16} | {'Frequência':^12}")
        print("-" * 46)
        
        for num, freq in bottom_8:
            # Frequência relativa ao número de concursos na janela
            porcentagem = (freq / len(janela_dados)) * 100
            print(f"{num:^10} | {freq:^16} | {porcentagem:^11.2f}%")
            
        lista_final = sorted([x[0] for x in bottom_8])
        
        print("\n============================================================")
        print(f" SUGESTÃO DE JOGO (BOTTOM 8):")
        print(f" {lista_final}")
        print("============================================================")
        print("\nNota: A estratégia 'Bottom N' aposta nos números 'frios',")
        print("acreditando na lei dos grandes números (equilíbrio a longo prazo).")

    except FileNotFoundError:
        print(f"\nERRO CRÍTICO: O arquivo '{ARQUIVO_DADOS}' não foi encontrado nesta pasta.")
        print("Por favor, verifique se o arquivo está no mesmo diretório do script.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")

if __name__ == "__main__":
    gerar_relatorio_bottom_8()
