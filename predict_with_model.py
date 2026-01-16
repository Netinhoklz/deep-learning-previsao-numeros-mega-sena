
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import warnings
import sys

# Force UTF-8 encoding for stdout if possible, or just avoid unicode
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

warnings.filterwarnings('ignore')

# --- Configurações ---
MODEL_PATH = 'model_epoch_035.keras'
DATA_PATH = 'Mega-Sena.xlsx'
WINDOW_SIZE = 500  
TOTAL_NUMBERS = 60

print(f"[INFO] Carregando modelo de: {MODEL_PATH}")
print(f"[INFO] Carregando dados de: {DATA_PATH}")

# --- Métricas Customizadas (Necessário para carregar o modelo) ---
def precision_metric(y_true, y_pred):
    y_pred_round = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred_round)
    fp = K.sum((1 - y_true) * y_pred_round)
    precision = tp / (tp + fp + K.epsilon())
    return precision

def recall_metric(y_true, y_pred):
    y_pred_round = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred_round)
    fn = K.sum(y_true * (1 - y_pred_round))
    recall = tp / (tp + fn + K.epsilon())
    return recall

def f1_score_metric(y_true, y_pred):
    y_pred_round = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred_round)
    fp = K.sum((1 - y_true) * y_pred_round)
    fn = K.sum(y_true * (1 - y_pred_round))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1

# --- Carregar Modelo ---
try:
    model = load_model(MODEL_PATH, custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_score_metric': f1_score_metric
    })
    print("[OK] Modelo carregado com sucesso!")
except Exception as e:
    print(f"[ERRO] Erro ao carregar modelo: {e}")
    # Tentar carregar sem metricas customizadas se falhar, apenas para predicao
    try:
        print("[INFO] Tentando carregar sem métricas customizadas (apenas para inferência)...")
        model = load_model(MODEL_PATH, compile=False)
        print("[OK] Modelo carregado (sem compilação)!")
    except Exception as e2:
        print(f"[ERRO] Falha fatal ao carregar modelo: {e2}")
        exit()

# --- Carregar e Preparar Dados ---
try:
    df = pd.read_excel(DATA_PATH)
    
    # Identificar colunas de bolas
    ball_columns = []
    for col in df.columns:
        if 'bola' in col.lower() or 'dezena' in col.lower():
            ball_columns.append(col)
            
    if len(ball_columns) != 6:
         ball_columns = []
         for col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0:
                if numeric_col.min() >= 1 and numeric_col.max() <= 60:
                    ball_columns.append(col)
    ball_columns = ball_columns[:6]

    # Preparar dados ordenados
    data_balls = df[ball_columns].dropna().astype(int)
    data_full = np.sort(data_balls.values, axis=1)
    
    print(f"[OK] Total de jogos carregados: {len(data_full)}")
    print(f"   Colunas identificadas: {ball_columns}")
    
except Exception as e:
    print(f"[ERRO] Erro ao ler Excel: {e}")
    exit()

# --- Função de Preparação de Input ---
def prepare_single_input(sequence_window):
    # sequence_window: shape (window_size, 6)
    
    # 1. Norm
    seq_norm = (sequence_window / 60.0).astype(np.float32)
    
    # 2. Freq
    freq = np.zeros(60, dtype=np.float32)
    all_nums = sequence_window.flatten().astype(int)
    for num in all_nums:
        if 1 <= num <= 60:
            freq[num-1] += 1
    freq_norm = freq / 60.0
    
    # 3. Gap
    window_sz = len(sequence_window)
    total_nums = window_sz * 6
    expected = total_nums / 60.0
    gap = expected - freq
    gap_norm = gap / expected
    
    # 4. Top/Bottom 10
    sorted_idx = np.argsort(freq)
    top10 = np.zeros(60, dtype=np.float32)
    bottom10 = np.zeros(60, dtype=np.float32)
    top10[sorted_idx[-10:]] = 1.0
    bottom10[sorted_idx[:10]] = 1.0
    
    # Expandir dims para (1, ...)
    return (
        np.expand_dims(seq_norm, 0),
        np.expand_dims(freq_norm, 0),
        np.expand_dims(gap_norm, 0),
        np.expand_dims(top10, 0),
        np.expand_dims(bottom10, 0)
    )

# --- Fazer Predição ---
print("\n" + "="*80)
print(f"PREDICAO PARA O PROXIMO CONCURSO ({len(data_full) + 1})")
print("="*80)

# Verificar se temos jogos suficientes para a janela
if len(data_full) < WINDOW_SIZE:
    print(f"[WARN] Historico ({len(data_full)}) menor que janela ({WINDOW_SIZE}). Usando todo historico disponivel.")
    last_window = data_full
else:
    # Pegar a última janela disponível
    last_window = data_full[-WINDOW_SIZE:]

# Tentar ajustar input shape se janela for diferente do esperado pelo modelo
try:
    expected_window = model.input_shape[0][1]
    if expected_window is not None and len(last_window) != expected_window:
        print(f"[WARN] Modelo espera janela de {expected_window}, mas temos {len(last_window)}.")
        if len(data_full) >= expected_window:
             last_window = data_full[-expected_window:]
             print(f"   -> Ajustado para usar ultimos {expected_window} jogos.")
        else:
             print("   -> ERRO: Dados insuficientes para a janela do modelo.")
except:
    pass 

next_inputs = prepare_single_input(last_window)

print("[...] Executando modelo...")
next_pred = model.predict(list(next_inputs), verbose=0)[0]

# Top 8 para o próximo jogo
top_8_idx = np.argsort(next_pred)[-8:]
top_8_nums = sorted(top_8_idx + 1)
top_8_probs = next_pred[top_8_idx]

# Top 6 para comparação (sugestão simples)
top_6_nums = sorted(np.argsort(next_pred)[-6:] + 1)

print(f"\n[INFO] Ultimo Jogo ({len(data_full)}): {list(data_full[-1])}")
print(f"\n[+] OS 8 NUMEROS MAIS PROVAVEIS PARA O PROXIMO JOGO:")
print(f"   -> {top_8_nums}")

print(f"\n[+] Aposta Sugerida (6 Numeros):")
print(f"   -> {top_6_nums}")

print("\n[INFO] Probabilidades dos Top 8:")
for num in top_8_nums:
    # Index is num-1
    prob = next_pred[num-1]
    print(f"   Num {num:02d}: {prob*100:.2f}%")

print("="*80)
