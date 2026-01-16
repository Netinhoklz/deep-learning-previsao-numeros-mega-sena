"""
GERADOR DE JOGOS MEGA-SENA - PREVISAO BASEADA EM FREQUENCIA

Este script gera numeros para o proximo jogo da Mega-Sena com base
nos numeros que mais aparecem dentro de cada janela otima.

Para cada quantidade de numeros, usa-se a janela otima correspondente:
- 6 numeros -> os 6 mais frequentes na janela de 1000 jogos
- 7 numeros -> os 7 mais frequentes na janela de 1200 jogos
- 8 numeros -> os 8 mais frequentes na janela de 800 jogos
- 9 numeros -> os 9 mais frequentes na janela de 850 jogos
- 10 numeros -> os 10 mais frequentes na janela de 850 jogos
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# ============================================================================
# CONFIGURACOES PRINCIPAIS
# ============================================================================

# Janelas otimas para cada quantidade de numeros
# (valores obtidos do notebook analise_comparativa_metodos_v3.ipynb)
JANELAS_OTIMAS = {
    6: 1000,
    7: 1200,
    8: 800,
    9: 850,
    10: 850
}

# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def carregar_dados(caminho_arquivo='Mega-Sena.xlsx'):
    """Carrega e processa os dados da Mega-Sena"""
    print(f"[*] Carregando dados de '{caminho_arquivo}'...")
    df = pd.read_excel(caminho_arquivo)
    ball_columns = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    sorteios = np.sort(df[ball_columns].dropna().astype(int).values, axis=1)
    
    # Obter informacao do ultimo concurso
    ultimo_concurso = df['Concurso'].dropna().astype(int).iloc[-1]
    ultima_data = df['Data do Sorteio'].iloc[-1]
    
    print(f"[OK] {len(sorteios):,} concursos carregados!")
    print(f"[OK] Ultimo concurso: {ultimo_concurso}")
    print(f"[OK] Data do ultimo sorteio: {ultima_data}\n")
    
    return sorteios, ultimo_concurso, ultima_data


def calcular_frequencias(sorteios_arr):
    """Calcula a frequencia de cada numero (1-60) em um conjunto de sorteios"""
    counter = Counter(sorteios_arr.flatten())
    return np.array([counter.get(i, 0) for i in range(1, 61)])


def get_top_n_numeros(frequencias, n):
    """Retorna os N numeros mais frequentes, ordenados numericamente"""
    indices = np.argsort(frequencias)[-n:]  # Indices dos N maiores
    numeros = sorted(indices + 1)  # Adiciona 1 (numeros sao 1-60) e ordena
    return numeros


def get_top_n_numeros_com_freq(frequencias, n):
    """Retorna os N numeros mais frequentes com suas frequencias, ordenados por frequencia"""
    numeros_freq = [(i + 1, frequencias[i]) for i in range(60)]
    ordenados = sorted(numeros_freq, key=lambda x: x[1], reverse=True)
    return ordenados[:n]


def formatar_jogo(numeros):
    """Formata uma lista de numeros para exibicao"""
    return " - ".join(f"{n:02d}" for n in sorted(numeros))


# ============================================================================
# GERADOR PRINCIPAL DE JOGOS
# ============================================================================

def gerar_previsao_jogos(sorteios, janelas_otimas=JANELAS_OTIMAS):
    """
    Gera previsao de jogos da Mega-Sena baseados nos numeros mais frequentes.
    
    Para cada quantidade de numeros (6, 7, 8, 9, 10):
    - Usa a janela otima correspondente
    - Pega exatamente N numeros mais frequentes
    """
    
    resultados = {}
    
    print("=" * 80)
    print("GERADOR DE PREVISAO MEGA-SENA")
    print("=" * 80)
    print(f"\nTotal de concursos disponiveis: {len(sorteios):,}")
    print(f"Data da analise: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    print("\n" + "=" * 80)
    print("JANELAS OTIMAS UTILIZADAS")
    print("=" * 80)
    for n_nums, janela in janelas_otimas.items():
        print(f"   {n_nums:2d} numeros -> Janela de {janela:4d} concursos")
    
    # Calcular frequencias para cada janela unica
    janelas_unicas = set(janelas_otimas.values())
    freq_por_janela = {}
    
    print("\n" + "=" * 80)
    print("CALCULANDO FREQUENCIAS...")
    print("=" * 80)
    for janela in sorted(janelas_unicas):
        if janela <= len(sorteios):
            freq_por_janela[janela] = calcular_frequencias(sorteios[-janela:])
            print(f"   [OK] Janela {janela:4d}: analisando ultimos {janela} concursos")
    
    print("\n" + "=" * 80)
    print("JOGOS PREVISTOS PARA O PROXIMO CONCURSO")
    print("=" * 80)
    
    # Processar cada quantidade de numeros
    for n_nums in [6, 7, 8, 9, 10]:
        janela = janelas_otimas[n_nums]
        
        # Obter frequencias da janela correspondente
        freq = freq_por_janela[janela]
        
        # Obter exatamente os N numeros mais frequentes
        top_numeros = get_top_n_numeros(freq, n_nums)
        top_numeros_com_freq = get_top_n_numeros_com_freq(freq, n_nums)
        
        # Armazenar resultados
        resultados[n_nums] = {
            'janela_otima': janela,
            'numeros': top_numeros,
            'numeros_com_freq': top_numeros_com_freq
        }
        
        # Exibir resultado
        print(f"\n   {'#' * 60}")
        print(f"   # JOGO COM {n_nums} NUMEROS (Janela: {janela} concursos)")
        print(f"   {'#' * 60}")
        print(f"\n   Numeros por frequencia (maior -> menor):")
        print(f"   ", end="")
        for i, (num, freq_val) in enumerate(top_numeros_com_freq):
            print(f"{num:02d}({freq_val}x)", end="")
            if i < len(top_numeros_com_freq) - 1:
                print(" > ", end="")
        print()
        print(f"\n   >>> JOGO: [ {formatar_jogo(top_numeros)} ]")
        
    return resultados


def salvar_relatorio(resultados, ultimo_concurso, ultima_data, nome_arquivo='previsao_megasena.txt'):
    """Salva um relatorio detalhado em arquivo texto"""
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PREVISAO DE JOGOS MEGA-SENA\n")
        f.write("Baseado nos numeros mais frequentes por janela otima\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nGerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Ultimo concurso analisado: {ultimo_concurso}\n")
        f.write(f"Data do ultimo sorteio: {ultima_data}\n")
        f.write(f"Previsao para o concurso: {ultimo_concurso + 1}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("METODOLOGIA\n")
        f.write("=" * 80 + "\n")
        f.write("""
Este relatorio apresenta jogos previstos para a Mega-Sena com base
nos numeros mais frequentes dentro de janelas otimas de analise.

As janelas otimas foram determinadas empiricamente atraves do notebook
'analise_comparativa_metodos_v3.ipynb', que identificou os melhores
periodos de analise para cada quantidade de numeros apostados.

Para cada jogo, sao selecionados exatamente N numeros mais frequentes
dentro da janela otima correspondente:

- 6 numeros  -> os 6 mais frequentes na janela de 1000 jogos
- 7 numeros  -> os 7 mais frequentes na janela de 1200 jogos
- 8 numeros  -> os 8 mais frequentes na janela de 800 jogos
- 9 numeros  -> os 9 mais frequentes na janela de 850 jogos
- 10 numeros -> os 10 mais frequentes na janela de 850 jogos
""")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"JOGOS PREVISTOS PARA O CONCURSO {ultimo_concurso + 1}\n")
        f.write("=" * 80 + "\n")
        
        for n_nums in [6, 7, 8, 9, 10]:
            dados = resultados[n_nums]
            
            f.write(f"\n{'#' * 60}\n")
            f.write(f"# JOGO COM {n_nums} NUMEROS\n")
            f.write(f"# Janela Otima: {dados['janela_otima']} concursos\n")
            f.write(f"{'#' * 60}\n")
            
            f.write(f"\n   Numeros ordenados por frequencia (maior -> menor):\n")
            f.write("   ")
            for i, (num, freq_val) in enumerate(dados['numeros_com_freq']):
                f.write(f"{num:02d}({freq_val}x)")
                if i < len(dados['numeros_com_freq']) - 1:
                    f.write(" > ")
            f.write("\n")
            
            f.write(f"\n   >>> JOGO: [ {formatar_jogo(dados['numeros'])} ]\n")
        
        # Resumo final
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESUMO DOS JOGOS\n")
        f.write("=" * 80 + "\n\n")
        
        for n_nums in [6, 7, 8, 9, 10]:
            dados = resultados[n_nums]
            f.write(f"   {n_nums:2d} numeros (janela {dados['janela_otima']:4d}): ")
            f.write(f"[ {formatar_jogo(dados['numeros'])} ]\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("AVISO IMPORTANTE\n")
        f.write("=" * 80 + "\n")
        f.write("""
Este gerador e baseado em analise estatistica de frequencia historica.
A Mega-Sena e um jogo de azar com resultados aleatorios e INDEPENDENTES.
O desempenho passado NAO garante resultados futuros.
Jogue com responsabilidade!
""")
    
    print(f"\n[>>] Relatorio salvo em: {nome_arquivo}")
    return nome_arquivo


# ============================================================================
# EXECUCAO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   INICIANDO GERADOR DE PREVISAO MEGA-SENA")
    print("=" * 80 + "\n")
    
    # Carregar dados
    sorteios, ultimo_concurso, ultima_data = carregar_dados('Mega-Sena.xlsx')
    
    # Gerar previsao
    resultados = gerar_previsao_jogos(sorteios)
    
    # Exibir resumo final
    print("\n" + "=" * 80)
    print(f"RESUMO - JOGOS PREVISTOS PARA O CONCURSO {ultimo_concurso + 1}")
    print("=" * 80)
    
    for n_nums in [6, 7, 8, 9, 10]:
        dados = resultados[n_nums]
        print(f"\n   {n_nums:2d} numeros (janela {dados['janela_otima']:4d}): [ {formatar_jogo(dados['numeros'])} ]")
    
    print()
    
    # Salvar relatorio
    salvar_relatorio(resultados, ultimo_concurso, ultima_data)
    
    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUIDO!")
    print("=" * 80 + "\n")
