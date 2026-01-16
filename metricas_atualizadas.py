# ============================================================================
# MÉTRICAS ATUALIZADAS PARA O MODELO RESNET + LSTM
# ============================================================================
#
# Copie este código para substituir a célula de compilação do modelo
# no notebook train_model_v2_resnet.ipynb
#
# NOVAS MÉTRICAS INCLUÍDAS:
#   - Accuracy: Porcentagem de acertos gerais
#   - Precision: Quantos positivos preditos são corretos
#   - Recall: Quantos positivos reais foram encontrados
#   - F1 Score: Média harmônica entre Precision e Recall
#   - AUC: Área sob a curva ROC (capacidade de discriminação)
# ============================================================================

# ========================
# Métricas Customizadas
# ========================

def precision_metric(y_true, y_pred):
    """
    Calcula a Precision durante o treinamento
    
    Precision = TP / (TP + FP)
    
    Mede quantos dos números preditos como positivos eram realmente corretos.
    Alta precision = poucas predições falsas positivas.
    """
    y_pred_round = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred_round)
    fp = K.sum((1 - y_true) * y_pred_round)
    precision = tp / (tp + fp + K.epsilon())
    return precision


def recall_metric(y_true, y_pred):
    """
    Calcula o Recall durante o treinamento
    
    Recall = TP / (TP + FN)
    
    Mede quantos dos números que deveriam ser preditos foram encontrados.
    Alto recall = encontra a maioria dos números corretos.
    """
    y_pred_round = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred_round)
    fn = K.sum(y_true * (1 - y_pred_round))
    recall = tp / (tp + fn + K.epsilon())
    return recall


def f1_score_metric(y_true, y_pred):
    """
    Calcula o F1 Score durante o treinamento
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Média harmônica entre Precision e Recall.
    Útil quando precisamos balancear ambas as métricas.
    """
    y_pred_round = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred_round)
    fp = K.sum((1 - y_true) * y_pred_round)
    fn = K.sum(y_true * (1 - y_pred_round))
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1


# ========================
# Compilar Modelo
# ========================
print("📊 Configurando métricas de monitoramento...")
print("   - Accuracy: Porcentagem de acertos gerais")
print("   - Precision: Quantos positivos preditos são corretos")
print("   - Recall: Quantos positivos reais foram encontrados")
print("   - F1 Score: Média harmônica entre Precision e Recall")
print("   - AUC: Área sob a curva ROC (capacidade de discriminação)")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        precision_metric,
        recall_metric,
        f1_score_metric,
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("\n✅ Modelo compilado com sucesso!")
print("\n📊 Resumo do Modelo:")
model.summary()
