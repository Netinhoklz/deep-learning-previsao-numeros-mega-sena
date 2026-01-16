
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import hypergeom
from collections import Counter
import warnings
import sys

# Force UTF-8 encoding for stdout if possible, or just avoid unicode
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 120
np.random.seed(42)

TOTAL_NUMBERS = 60
NUMBERS_DRAWN = 6  # Mega-Sena sempre sorteia 6

print("[OK] Bibliotecas carregadas!")

# Carregar dados
try:
    df = pd.read_excel('Mega-Sena.xlsx')
    ball_columns = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    # If standard columns fail, try adapting like in previous scripts
    if not all(col in df.columns for col in ball_columns):
         ball_columns = []
         for col in df.columns:
            if 'bola' in col.lower() or 'dezena' in col.lower():
                ball_columns.append(col)
         ball_columns = ball_columns[:6]

    sorteios = np.sort(df[ball_columns].dropna().astype(int).values, axis=1)
    print(f"[OK] Total de concursos: {len(sorteios):,}")
except Exception as e:
    print(f"[ERRO] Erro ao carregar dados: {e}")
    # Create dummy data for testing if file missing
    sorteios = np.random.randint(1, 61, (100, 6))
    for i in range(len(sorteios)):
        sorteios[i] = np.sort(np.random.choice(range(1, 61), 6, replace=False))
    print("[WARN] Usando dados aleatorios para teste!")

# --- Funcoes ---
def calc_freq_janela(sorteios_janela):
    if len(sorteios_janela) == 0:
        return np.zeros(TOTAL_NUMBERS)
    counter = Counter(sorteios_janela.flatten())
    # Count frequency for numbers 1 to 60
    return np.array([counter.get(i, 0) for i in range(1, TOTAL_NUMBERS + 1)])

def estrategia_bottomN(freq, n):
    """Retorna os N numeros menos frequentes."""
    # argsort returns indices that sort array. Indices are 0-59, so add 1 to get numbers 1-60.
    # We take the first n indices for bottom n (lowest frequency)
    return sorted(np.argsort(freq)[:n] + 1)

def contar_acertos(aposta, sorteio):
    return len(set(aposta) & set(sorteio))

# --- Busca de Parametros ---
# Janelas: 10 a 1000, passo 10
JANELA_MIN, JANELA_MAX, JANELA_STEP = 50, 2000, 50
janelas = list(range(JANELA_MIN, JANELA_MAX + 1, JANELA_STEP))

# Quantidade de numeros apostados: 6 a 15 (focando em bottom, as vezes jogar mais numeros ajuda)
NUM_MIN, NUM_MAX = 6, 10
qtd_numeros = list(range(NUM_MIN, NUM_MAX + 1))

print(f"[INFO] Configuracoes:")
print(f"   Janelas: {len(janelas)} ({JANELA_MIN} a {JANELA_MAX})")
print(f"   Numeros apostados: {qtd_numeros}")
print(f"   Total de combinacoes (Janela x Qtd): {len(janelas) * len(qtd_numeros)}")

# Estrutura para armazenar resultados: resultados[n_nums][janela] = {stat dict}
resultados = {n: {} for n in qtd_numeros}

print("[...] Iniciando simulacao Bottom N...\n")

for n_nums in qtd_numeros:
    print(f"[*] Processando Bottom {n_nums}...")
    
    for janela in janelas:
        acertos = []
        
        # Test historical performance
        # We predict for game 'i' using window [i-janela : i]
        for i in range(janela, len(sorteios)):
            freq = calc_freq_janela(sorteios[i-janela:i])
            sorteio_real = sorteios[i]
            
            aposta = estrategia_bottomN(freq, n_nums)
            acertos.append(contar_acertos(aposta, sorteio_real))
        
        arr = np.array(acertos)
        resultados[n_nums][janela] = {
            'media': np.mean(arr),
            'std': np.std(arr),
            'max': np.max(arr),
            'terno': np.sum(arr == 3),
            'quadra': np.sum(arr == 4),
            'quina': np.sum(arr == 5),
            'sena': np.sum(arr == 6),
            'total_jogos': len(arr)
        }

print("\n[OK] Simulacao concluida!")

# --- Analise e Visualizacao ---

# Valores teoricos para aleatorio
valores_teoricos = {}
for n in qtd_numeros:
    valores_teoricos[n] = hypergeom.mean(60, 6, n)

print("\n" + "="*80)
print("MELHORES JANELAS PARA ESTRATEGIA BOTTOM N")
print("="*80)

melhores_configs = {}

for n in qtd_numeros:
    # Find best window based on mean hits
    melhor_janela = max(janelas, key=lambda j: resultados[n][j]['media'])
    stats = resultados[n][melhor_janela]
    
    melhores_configs[n] = (melhor_janela, stats)
    
    teorico = valores_teoricos[n]
    diff_pct = ((stats['media'] - teorico) / teorico) * 100
    
    print(f"\n[+] Bottom {n}:")
    print(f"   Melhor Janela: {melhor_janela} concursos")
    print(f"   Media Acertos: {stats['media']:.4f} (Teorico: {teorico:.4f} | Diff: {diff_pct:+.2f}%)")
    print(f"   Distribuicao (em {stats['total_jogos']} jogos):")
    print(f"      3 acertos: {stats['terno']} ({stats['terno']/stats['total_jogos']*100:.2f}%)")
    print(f"      4 acertos: {stats['quadra']} ({stats['quadra']/stats['total_jogos']*100:.2f}%)")
    print(f"      5 acertos: {stats['quina']} ({stats['quina']/stats['total_jogos']*100:.2f}%)")
    print(f"      6 acertos: {stats['sena']} ({stats['sena']/stats['total_jogos']*100:.2f}%)")

# Plot Heatmap of Mean Hits (Window vs N Numbers)
matrix_media = np.zeros((len(qtd_numeros), len(janelas)))
for i, n in enumerate(qtd_numeros):
    for j, win in enumerate(janelas):
        matrix_media[i, j] = resultados[n][win]['media']

plt.figure(figsize=(15, 8))
plt.imshow(matrix_media, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Media de Acertos')
plt.title('Heatmap: Media de Acertos - Estrategia Bottom N', fontsize=16)
plt.xlabel('Tamanho da Janela', fontsize=12)
plt.ylabel('Qtde Numeros Apostados', fontsize=12)

# Set ticks
x_ticks_loc = np.linspace(0, len(janelas)-1, 10, dtype=int)
plt.xticks(x_ticks_loc, [janelas[i] for i in x_ticks_loc])
plt.yticks(range(len(qtd_numeros)), qtd_numeros)

plt.tight_layout()
plt.savefig('heatmap_bottom_n.png')
print("\n[OK] Heatmap salvo como 'heatmap_bottom_n.png'")

# Plot Best Window Performance over time (optional or just best windows)
plt.figure(figsize=(12, 6))
# Plot mean hits for N=6, 8, 10 across all windows
for n in [6, 8, 10, 15]:
    if n in qtd_numeros:
        medias = [resultados[n][w]['media'] for w in janelas]
        plt.plot(janelas, medias, label=f'Bottom {n}')
        plt.axhline(y=valores_teoricos[n], linestyle='--', alpha=0.5, color=plt.gca().lines[-1].get_color())

plt.title('Performance por Tamanho de Janela (Bottom N)', fontsize=16)
plt.xlabel('Janela Historica', fontsize=12)
plt.ylabel('Media de Acertos', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('janelas_bottom_n.png')
print("[OK] Grafico de janelas salvo como 'janelas_bottom_n.png'")

# --- Predicao para o Proximo Concurso ---
print("\n" + "="*80)
print(f"PREDICAO PARA O PROXIMO CONCURSO ({len(sorteios) + 1})")
print("="*80)

print(f"[INFO] Ultimo Jogo ({len(sorteios)}): {list(sorteios[-1])}")

print("\n[+] Sugestoes de Aposta (Estrategia Bottom N com Melhor Janela):")

# Encontrar qual N teve a maior media absoluta
best_n_overall = max(qtd_numeros, key=lambda n: melhores_configs[n][1]['media'])

for n in qtd_numeros:
    melhor_janela, stats = melhores_configs[n]
    
    # Pegar ultimos 'melhor_janela' jogos
    if len(sorteios) < melhor_janela:
        history_to_use = sorteios
        actual_window = len(sorteios)
    else:
        history_to_use = sorteios[-melhor_janela:]
        actual_window = melhor_janela
    
    # Calcular frequencia
    freq = calc_freq_janela(history_to_use)
    
    # Pegar Bottom N
    sugestao = estrategia_bottomN(freq, n)
    
    media = stats['media']
    teorico = valores_teoricos[n]
    gain = ((media - teorico) / teorico) * 100
    
    marker = "<< MELHOR DESEMPENHO" if n == best_n_overall else ""
    
    print(f"\n   [+] Bottom {n} (Janela {actual_window}) {marker}")
    print(f"       Numeros: {sugestao}")
    print(f"       Media Historica: {media:.4f} | Ganho sobre acaso: {gain:+.2f}%")

print("\n" + "="*80)

