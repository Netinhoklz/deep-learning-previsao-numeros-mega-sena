"""
Script para adicionar inputs de top/bottom 10 números e aprofundar a rede neural
"""
import json

# ============================================================================
# PARTE 1: Atualizar prepare_data_simple.ipynb
# ============================================================================
prepare_path = r'c:\Users\netin\Downloads\Codigos Python\mega_sena\prepare_data_simple.ipynb'

with open(prepare_path, 'r', encoding='utf-8') as f:
    prepare_nb = json.load(f)

# Nova célula para criar vetores de top/bottom 10
new_top_bottom_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "top_bottom_10_code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# CRIAÇÃO DOS VETORES DE TOP 10 E BOTTOM 10 NÚMEROS\n",
        "# ============================================================================\n",
        "\n",
        "def create_top_bottom_vectors(X_data):\n",
        "    \"\"\"\n",
        "    Cria vetores one-hot para os 10 números mais e menos frequentes.\n",
        "    \n",
        "    Args:\n",
        "        X_data: array de shape (n_samples, window_size, 6)\n",
        "    \n",
        "    Returns:\n",
        "        top10_vectors: array (n_samples, 60) - one-hot dos 10 mais frequentes\n",
        "        bottom10_vectors: array (n_samples, 60) - one-hot dos 10 menos frequentes\n",
        "    \"\"\"\n",
        "    n_samples = X_data.shape[0]\n",
        "    \n",
        "    top10_vectors = np.zeros((n_samples, 60), dtype=np.float32)\n",
        "    bottom10_vectors = np.zeros((n_samples, 60), dtype=np.float32)\n",
        "    \n",
        "    print(f\"Criando vetores Top/Bottom 10 para {n_samples} amostras...\")\n",
        "    \n",
        "    for i in range(n_samples):\n",
        "        # Contar frequência de cada número\n",
        "        freq = np.zeros(60)\n",
        "        all_numbers = X_data[i].flatten().astype(int)\n",
        "        \n",
        "        for num in all_numbers:\n",
        "            if 1 <= num <= 60:\n",
        "                freq[num - 1] += 1\n",
        "        \n",
        "        # Ordenar índices por frequência\n",
        "        sorted_indices = np.argsort(freq)\n",
        "        \n",
        "        # Bottom 10: 10 números menos frequentes\n",
        "        bottom10_indices = sorted_indices[:10]\n",
        "        bottom10_vectors[i, bottom10_indices] = 1.0\n",
        "        \n",
        "        # Top 10: 10 números mais frequentes\n",
        "        top10_indices = sorted_indices[-10:]\n",
        "        top10_vectors[i, top10_indices] = 1.0\n",
        "    \n",
        "    print(f\"   Top10 shape: {top10_vectors.shape}\")\n",
        "    print(f\"   Bottom10 shape: {bottom10_vectors.shape}\")\n",
        "    \n",
        "    return top10_vectors, bottom10_vectors\n",
        "\n",
        "\n",
        "print(\"=\"*60)\n",
        "print(\"📊 CRIANDO VETORES TOP 10 E BOTTOM 10\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "print(\"\\n🔹 Conjunto de Treino:\")\n",
        "X_train_top10, X_train_bottom10 = create_top_bottom_vectors(X_train)\n",
        "\n",
        "print(\"\\n🔹 Conjunto de Validação:\")\n",
        "X_val_top10, X_val_bottom10 = create_top_bottom_vectors(X_val)\n",
        "\n",
        "print(\"\\n🔹 Conjunto de Teste:\")\n",
        "X_test_top10, X_test_bottom10 = create_top_bottom_vectors(X_test)\n",
        "\n",
        "print(\"\\n✅ Vetores Top/Bottom 10 criados com sucesso!\")\n"
    ]
}

new_top_bottom_markdown = {
    "cell_type": "markdown",
    "id": "top_bottom_10_md",
    "metadata": {},
    "source": [
        "---\n",
        "## 4.5 🔝 Criação dos Vetores Top 10 e Bottom 10\n",
        "\n",
        "Para cada amostra, identificamos:\n",
        "- **Top 10**: Os 10 números que mais apareceram na janela (one-hot encoded)\n",
        "- **Bottom 10**: Os 10 números que menos apareceram na janela (one-hot encoded)\n",
        "\n",
        "Estes vetores ajudam o modelo a focar nos números \"quentes\" e \"frios\".\n"
    ]
}

# Atualizar célula de salvar dados
new_save_source = [
    "# Salvar arquivos .npy\n",
    "np.save('X_train.npy', X_train)\n",
    "np.save('y_train.npy', y_train)\n",
    "np.save('X_val.npy', X_val)\n",
    "np.save('y_val.npy', y_val)\n",
    "np.save('X_test.npy', X_test)\n",
    "np.save('y_test.npy', y_test)\n",
    "\n",
    "# Salvar vetores Top/Bottom 10\n",
    "np.save('X_train_top10.npy', X_train_top10)\n",
    "np.save('X_train_bottom10.npy', X_train_bottom10)\n",
    "np.save('X_val_top10.npy', X_val_top10)\n",
    "np.save('X_val_bottom10.npy', X_val_bottom10)\n",
    "np.save('X_test_top10.npy', X_test_top10)\n",
    "np.save('X_test_bottom10.npy', X_test_bottom10)\n",
    "\n",
    "print(\"\\n✅ Dados salvos com sucesso!\")\n",
    "print(\"\\nArquivos criados:\")\n",
    "print(\"   - X_train.npy, y_train.npy\")\n",
    "print(\"   - X_val.npy, y_val.npy\")\n",
    "print(\"   - X_test.npy, y_test.npy\")\n",
    "print(\"   - X_train_top10.npy, X_train_bottom10.npy\")\n",
    "print(\"   - X_val_top10.npy, X_val_bottom10.npy\")\n",
    "print(\"   - X_test_top10.npy, X_test_bottom10.npy\")\n"
]

# Inserir células no prepare_data_simple
# Encontrar célula de salvar dados
save_idx = None
for i, cell in enumerate(prepare_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "np.save('X_train.npy'" in source_text:
            save_idx = i
            break

if save_idx:
    print(f"[prepare_data] Inserindo celulas antes do indice {save_idx}")
    prepare_nb['cells'].insert(save_idx, new_top_bottom_markdown)
    prepare_nb['cells'].insert(save_idx + 1, new_top_bottom_code)
    
    # Atualizar célula de salvar (agora está 2 posições depois)
    prepare_nb['cells'][save_idx + 2]['source'] = new_save_source
    prepare_nb['cells'][save_idx + 2]['outputs'] = []

# Salvar prepare_data_simple.ipynb
with open(prepare_path, 'w', encoding='utf-8') as f:
    json.dump(prepare_nb, f, ensure_ascii=False, indent=1)

print("[prepare_data] Notebook atualizado!")

# ============================================================================
# PARTE 2: Atualizar train_model_v2_resnet.ipynb
# ============================================================================
train_path = r'c:\Users\netin\Downloads\Codigos Python\mega_sena\train_model_v2_resnet.ipynb'

with open(train_path, 'r', encoding='utf-8') as f:
    train_nb = json.load(f)

# Nova célula para carregar dados Top/Bottom 10
new_load_top_bottom = {
    "cell_type": "code",
    "execution_count": None,
    "id": "load_top_bottom_code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# CARREGAR VETORES TOP 10 E BOTTOM 10\n",
        "# ============================================================================\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"📊 CARREGANDO VETORES TOP 10 E BOTTOM 10\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "try:\n",
        "    X_train_top10 = np.load('X_train_top10.npy')\n",
        "    X_train_bottom10 = np.load('X_train_bottom10.npy')\n",
        "    X_val_top10 = np.load('X_val_top10.npy')\n",
        "    X_val_bottom10 = np.load('X_val_bottom10.npy')\n",
        "    X_test_top10 = np.load('X_test_top10.npy')\n",
        "    X_test_bottom10 = np.load('X_test_bottom10.npy')\n",
        "    \n",
        "    print(\"\\n✅ Vetores Top/Bottom 10 carregados!\")\n",
        "    print(f\"   X_train_top10: {X_train_top10.shape}\")\n",
        "    print(f\"   X_train_bottom10: {X_train_bottom10.shape}\")\n",
        "    \n",
        "except FileNotFoundError:\n",
        "    print(\"\\n⚠️ Arquivos Top/Bottom 10 não encontrados.\")\n",
        "    print(\"   Execute prepare_data_simple.ipynb novamente!\")\n",
        "    \n",
        "    # Criar vetores temporários\n",
        "    def create_top_bottom_vectors(X_data):\n",
        "        n_samples = X_data.shape[0]\n",
        "        top10_vectors = np.zeros((n_samples, 60), dtype=np.float32)\n",
        "        bottom10_vectors = np.zeros((n_samples, 60), dtype=np.float32)\n",
        "        \n",
        "        for i in range(n_samples):\n",
        "            freq = np.zeros(60)\n",
        "            all_numbers = X_data[i].flatten().astype(int)\n",
        "            for num in all_numbers:\n",
        "                if 1 <= num <= 60:\n",
        "                    freq[num - 1] += 1\n",
        "            sorted_indices = np.argsort(freq)\n",
        "            bottom10_vectors[i, sorted_indices[:10]] = 1.0\n",
        "            top10_vectors[i, sorted_indices[-10:]] = 1.0\n",
        "        return top10_vectors, bottom10_vectors\n",
        "    \n",
        "    print(\"   Criando vetores temporários...\")\n",
        "    X_train_top10, X_train_bottom10 = create_top_bottom_vectors(X_train)\n",
        "    X_val_top10, X_val_bottom10 = create_top_bottom_vectors(X_val)\n",
        "    X_test_top10, X_test_bottom10 = create_top_bottom_vectors(X_test)\n",
        "    print(\"   ✅ Vetores criados!\")\n"
    ]
}

# Nova arquitetura PROFUNDA do modelo (5 inputs)
new_deep_model_source = [
    "def build_resnet_lstm_model(window_size, input_features):\n",
    "    \"\"\"\n",
    "    Modelo ULTRA PROFUNDO com CINCO INPUTS:\n",
    "    1. Sequência de jogos: (window_size, 6)\n",
    "    2. Vetor de frequências: (60,)\n",
    "    3. Vetor de gap (lacuna): (60,)\n",
    "    4. Vetor Top 10 (mais frequentes): (60,)\n",
    "    5. Vetor Bottom 10 (menos frequentes): (60,)\n",
    "    \n",
    "    Arquitetura PROFUNDA:\n",
    "    - 4 Blocos ResNet\n",
    "    - 8 Camadas Attention + LSTM\n",
    "    - 12 Camadas Densas\n",
    "    \"\"\"\n",
    "    print(\"\\n\" + \"=\"*80)\n",
    "    print(\"🧠 CONSTRUINDO MODELO ULTRA PROFUNDO (5 INPUTS)\")\n",
    "    print(\"=\"*80)\n",
    "    \n",
    "    # ==========================================\n",
    "    # INPUTS\n",
    "    # ==========================================\n",
    "    input_sequence = Input(shape=(window_size, input_features), name=\"Input_Sequence\")\n",
    "    input_frequency = Input(shape=(60,), name=\"Input_Frequency\")\n",
    "    input_gap = Input(shape=(60,), name=\"Input_Gap\")\n",
    "    input_top10 = Input(shape=(60,), name=\"Input_Top10\")\n",
    "    input_bottom10 = Input(shape=(60,), name=\"Input_Bottom10\")\n",
    "    \n",
    "    print(f\"   Input 1 (Sequência): ({window_size}, {input_features})\")\n",
    "    print(f\"   Input 2 (Frequência): (60,)\")\n",
    "    print(f\"   Input 3 (Gap): (60,)\")\n",
    "    print(f\"   Input 4 (Top10): (60,)\")\n",
    "    print(f\"   Input 5 (Bottom10): (60,)\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # CAMINHO 1: Processamento da Sequência (PROFUNDO)\n",
    "    # ==========================================\n",
    "    print(\"\\n   [CAMINHO 1] Sequência - ResNet + Attention + LSTM:\")\n",
    "    \n",
    "    # Expansão profunda\n",
    "    x = Dense(128, activation='relu', name=\"Expand_1\")(input_sequence)\n",
    "    x = Dense(256, activation='relu', name=\"Expand_2\")(x)\n",
    "    x = Dense(512, activation='relu', name=\"Expand_3\")(x)\n",
    "    x = BatchNormalization(name=\"BN_Expand\")(x)\n",
    "    print(\"   [1.0] Expansão: 6 → 128 → 256 → 512\")\n",
    "    \n",
    "    # 4 Blocos ResNet\n",
    "    for i in range(4):\n",
    "        x = resnet_block(x, filters=512, name_prefix=f\"ResNet_{i+1}\")\n",
    "    x = Dropout(0.2, name=\"Dropout_ResNet\")(x)\n",
    "    print(\"   [1.1] 4 Blocos ResNet: 512 filtros cada\")\n",
    "    \n",
    "    # 8 camadas Attention + LSTM\n",
    "    lstm_units = [512, 512, 512, 512, 256, 256, 256, 128]\n",
    "    \n",
    "    for i, units in enumerate(lstm_units):\n",
    "        # Multi-Head Attention\n",
    "        attn = tf.keras.layers.MultiHeadAttention(\n",
    "            num_heads=8, key_dim=64, name=f\"Attention_{i+1}\"\n",
    "        )(x, x)\n",
    "        attn = tf.keras.layers.Add(name=f\"Attn_Res_{i+1}\")([x, attn])\n",
    "        attn = tf.keras.layers.LayerNormalization(name=f\"Attn_LN_{i+1}\")(attn)\n",
    "        \n",
    "        # LSTM\n",
    "        return_seq = (i < len(lstm_units) - 1)  # False apenas no último\n",
    "        lstm = LSTM(units, return_sequences=return_seq, name=f\"LSTM_{i+1}\",\n",
    "                    kernel_regularizer=l2(0.001))(attn)\n",
    "        \n",
    "        if return_seq:\n",
    "            lstm = BatchNormalization(name=f\"BN_LSTM_{i+1}\")(lstm)\n",
    "            lstm = Dropout(0.2)(lstm)\n",
    "            x = lstm\n",
    "        else:\n",
    "            lstm = BatchNormalization(name=f\"BN_LSTM_{i+1}\")(lstm)\n",
    "            lstm = Dropout(0.3)(lstm)\n",
    "            sequence_output = lstm\n",
    "    \n",
    "    print(f\"   [1.2] 8 Attention + LSTM: {lstm_units}\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # CAMINHO 2: Frequências\n",
    "    # ==========================================\n",
    "    print(\"\\n   [CAMINHO 2] Frequências:\")\n",
    "    freq = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(input_frequency)\n",
    "    freq = BatchNormalization()(freq)\n",
    "    freq = Dropout(0.2)(freq)\n",
    "    freq = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(freq)\n",
    "    freq = BatchNormalization()(freq)\n",
    "    freq = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(freq)\n",
    "    freq = BatchNormalization()(freq)\n",
    "    print(\"   60 → 256 → 128 → 64\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # CAMINHO 3: Gap\n",
    "    # ==========================================\n",
    "    print(\"\\n   [CAMINHO 3] Gap:\")\n",
    "    gap = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(input_gap)\n",
    "    gap = BatchNormalization()(gap)\n",
    "    gap = Dropout(0.2)(gap)\n",
    "    gap = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(gap)\n",
    "    gap = BatchNormalization()(gap)\n",
    "    gap = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(gap)\n",
    "    gap = BatchNormalization()(gap)\n",
    "    print(\"   60 → 256 → 128 → 64\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # CAMINHO 4: Top 10\n",
    "    # ==========================================\n",
    "    print(\"\\n   [CAMINHO 4] Top 10:\")\n",
    "    top10 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_top10)\n",
    "    top10 = BatchNormalization()(top10)\n",
    "    top10 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(top10)\n",
    "    top10 = BatchNormalization()(top10)\n",
    "    print(\"   60 → 128 → 64\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # CAMINHO 5: Bottom 10\n",
    "    # ==========================================\n",
    "    print(\"\\n   [CAMINHO 5] Bottom 10:\")\n",
    "    bottom10 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_bottom10)\n",
    "    bottom10 = BatchNormalization()(bottom10)\n",
    "    bottom10 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(bottom10)\n",
    "    bottom10 = BatchNormalization()(bottom10)\n",
    "    print(\"   60 → 128 → 64\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # MERGE - Concatenação\n",
    "    # ==========================================\n",
    "    print(\"\\n   [MERGE] Concatenação:\")\n",
    "    merged = tf.keras.layers.Concatenate(name=\"Merge\")([\n",
    "        sequence_output, freq, gap, top10, bottom10\n",
    "    ])\n",
    "    print(\"   128 + 64 + 64 + 64 + 64 = 384\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # TORRE DENSA PROFUNDA (12 camadas)\n",
    "    # ==========================================\n",
    "    x = merged\n",
    "    dense_dims = [1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64]\n",
    "    \n",
    "    print(f\"\\n   [DENSE] Torre Profunda: {dense_dims}\")\n",
    "    \n",
    "    for i, dim in enumerate(dense_dims):\n",
    "        x = Dense(dim, activation='swish', name=f\"Dense_{i+1}\",\n",
    "                  kernel_regularizer=l2(0.001))(x)\n",
    "        x = BatchNormalization(name=f\"BN_Dense_{i+1}\")(x)\n",
    "        if i % 2 == 0:\n",
    "            x = Dropout(0.25)(x)\n",
    "    \n",
    "    # ==========================================\n",
    "    # SAÍDA\n",
    "    # ==========================================\n",
    "    outputs = Dense(TOTAL_NUMBERS, activation='sigmoid', name=\"Output\")(x)\n",
    "    print(\"   [OUTPUT] 60 probabilidades (sigmoid)\")\n",
    "    \n",
    "    # ==========================================\n",
    "    # CRIAR MODELO\n",
    "    # ==========================================\n",
    "    model = Model(\n",
    "        inputs=[input_sequence, input_frequency, input_gap, input_top10, input_bottom10],\n",
    "        outputs=outputs,\n",
    "        name=\"MegaSena_UltraDeep_5Inputs\"\n",
    "    )\n",
    "    \n",
    "    print(\"\\n\" + \"=\"*80)\n",
    "    print(f\"   Total de camadas: {len(model.layers)}\")\n",
    "    print(f\"   Parâmetros treináveis: {model.count_params():,}\")\n",
    "    print(\"=\"*80)\n",
    "    \n",
    "    return model\n",
    "\n",
    "\n",
    "# Construir o modelo\n",
    "model = build_resnet_lstm_model(WINDOW_SIZE, INPUT_FEATURES)\n"
]

# Novo código de treinamento (5 inputs)
new_training_source = [
    "# ============================================================================\n",
    "# CALLBACKS E TREINAMENTO\n",
    "# ============================================================================\n",
    "\n",
    "class SaveEveryNEpochs(tf.keras.callbacks.Callback):\n",
    "    def __init__(self, save_freq=5, filepath='checkpoints/model_epoch_{epoch:03d}.keras'):\n",
    "        super().__init__()\n",
    "        self.save_freq = save_freq\n",
    "        self.filepath = filepath\n",
    "        \n",
    "    def on_epoch_end(self, epoch, logs=None):\n",
    "        if (epoch + 1) % self.save_freq == 0:\n",
    "            filepath = self.filepath.format(epoch=epoch+1)\n",
    "            self.model.save(filepath)\n",
    "            print(f\"\\n💾 Modelo salvo: {filepath}\")\n",
    "\n",
    "import os\n",
    "os.makedirs('checkpoints', exist_ok=True)\n",
    "\n",
    "callbacks = [\n",
    "    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),\n",
    "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),\n",
    "    SaveEveryNEpochs(save_freq=5)\n",
    "]\n",
    "\n",
    "print(\"✅ Callbacks configurados\")\n",
    "\n",
    "# ============================================================================\n",
    "# TREINAMENTO COM 5 INPUTS\n",
    "# ============================================================================\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"🏋️ INICIANDO TREINAMENTO (MODELO ULTRA PROFUNDO)\")\n",
    "print(\"=\"*80)\n",
    "print(f\"\\n   Épocas: {EPOCHS}\")\n",
    "print(f\"   Batch Size: {BATCH_SIZE}\")\n",
    "print(f\"   5 Inputs:\")\n",
    "print(f\"      - Sequência: {X_train_norm.shape}\")\n",
    "print(f\"      - Frequência: {X_train_freq.shape}\")\n",
    "print(f\"      - Gap: {X_train_gap.shape}\")\n",
    "print(f\"      - Top10: {X_train_top10.shape}\")\n",
    "print(f\"      - Bottom10: {X_train_bottom10.shape}\")\n",
    "print(\"\\n\")\n",
    "\n",
    "history = model.fit(\n",
    "    [X_train_norm, X_train_freq, X_train_gap, X_train_top10, X_train_bottom10],\n",
    "    y_train,\n",
    "    validation_data=(\n",
    "        [X_val_norm, X_val_freq, X_val_gap, X_val_top10, X_val_bottom10],\n",
    "        y_val\n",
    "    ),\n",
    "    epochs=EPOCHS,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    callbacks=callbacks,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"✅ TREINAMENTO CONCLUÍDO!\")\n",
    "print(\"=\"*80)\n"
]

# Atualizar avaliação, predição e outras funções para 5 inputs
new_evaluation_source = [
    "# ============================================================================\n",
    "# AVALIAÇÃO DO MODELO (5 INPUTS)\n",
    "# ============================================================================\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"📊 AVALIAÇÃO NO CONJUNTO DE TESTE\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "test_results = model.evaluate(\n",
    "    [X_test_norm, X_test_freq, X_test_gap, X_test_top10, X_test_bottom10],\n",
    "    y_test,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "print(\"\\n📋 Resultados:\")\n",
    "for metric_name, value in zip(model.metrics_names, test_results):\n",
    "    print(f\"   {metric_name}: {value:.4f}\")\n"
]

new_prediction_source = [
    "# ============================================================================\n",
    "# PREDIÇÃO DE EXEMPLO (5 INPUTS)\n",
    "# ============================================================================\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"🔮 EXEMPLO DE PREDIÇÃO\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "idx = np.random.randint(0, len(X_test))\n",
    "\n",
    "predictions = model.predict([\n",
    "    X_test_norm[idx:idx+1],\n",
    "    X_test_freq[idx:idx+1],\n",
    "    X_test_gap[idx:idx+1],\n",
    "    X_test_top10[idx:idx+1],\n",
    "    X_test_bottom10[idx:idx+1]\n",
    "], verbose=0)[0]\n",
    "\n",
    "top_6_indices = np.argsort(predictions)[-6:]\n",
    "top_6_numbers = top_6_indices + 1\n",
    "top_6_probs = predictions[top_6_indices]\n",
    "\n",
    "true_indices = np.where(y_test[idx] == 1)[0]\n",
    "true_numbers = true_indices + 1\n",
    "\n",
    "print(f\"\\n📊 Amostra #{idx}\")\n",
    "print(f\"\\n🎯 Números Verdadeiros: {sorted(true_numbers)}\")\n",
    "print(f\"🔮 Números Previstos:   {sorted(top_6_numbers)}\")\n",
    "\n",
    "matches = len(set(true_numbers) & set(top_6_numbers))\n",
    "print(f\"\\n✅ Acertos: {matches}/6\")\n",
    "print(\"=\"*80)\n"
]

# Atualizar funções de avaliação
new_eval_hits_source = [
    "# ============================================================================\n",
    "# AVALIAÇÃO DE ACERTOS (5 INPUTS)\n",
    "# ============================================================================\n",
    "\n",
    "def evaluate_hits_by_prediction_count(model, inputs, y_test, pred_counts=[6, 7, 8, 9, 10]):\n",
    "    print(\"=\"*80)\n",
    "    print(\"🎯 AVALIAÇÃO DE ACERTOS\")\n",
    "    print(\"=\"*80)\n",
    "    \n",
    "    predictions = model.predict(inputs, verbose=0)\n",
    "    results = {}\n",
    "    \n",
    "    for n_pred in pred_counts:\n",
    "        hits_distribution = {k: 0 for k in range(7)}\n",
    "        total_hits = 0\n",
    "        \n",
    "        for i in range(len(predictions)):\n",
    "            pred_set = set(np.argsort(predictions[i])[-n_pred:] + 1)\n",
    "            true_set = set(np.where(y_test[i] == 1)[0] + 1)\n",
    "            hits = len(pred_set & true_set)\n",
    "            hits_distribution[hits] += 1\n",
    "            total_hits += hits\n",
    "        \n",
    "        n_samples = len(predictions)\n",
    "        results[n_pred] = {\n",
    "            'avg_hits': total_hits / n_samples,\n",
    "            'distribution': hits_distribution,\n",
    "            'terno_or_more': sum(hits_distribution[k] for k in [3,4,5,6]) / n_samples * 100\n",
    "        }\n",
    "    \n",
    "    print(f\"\\n{'Nums':<8} {'Média':<10} {'Terno+':<10}\")\n",
    "    print(\"-\"*30)\n",
    "    for n_pred in pred_counts:\n",
    "        r = results[n_pred]\n",
    "        print(f\"{n_pred:<8} {r['avg_hits']:<10.3f} {r['terno_or_more']:<10.2f}%\")\n",
    "    \n",
    "    return results\n",
    "\n",
    "# Executar\n",
    "test_inputs = [X_test_norm, X_test_freq, X_test_gap, X_test_top10, X_test_bottom10]\n",
    "results = evaluate_hits_by_prediction_count(model, test_inputs, y_test)\n"
]

# Aplicar alterações no train_model_v2_resnet.ipynb

# 1. Inserir célula de carregar top/bottom após carregar dados
for i, cell in enumerate(train_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'X_test_gap = create_gap_vectors' in source_text:
            print(f"[train] Inserindo celula de carregar top/bottom apos indice {i}")
            train_nb['cells'].insert(i + 1, new_load_top_bottom)
            break

# 2. Atualizar modelo
for i, cell in enumerate(train_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'def build_resnet_lstm_model' in source_text:
            print(f"[train] Atualizando modelo (indice {i})")
            train_nb['cells'][i]['source'] = new_deep_model_source
            train_nb['cells'][i]['outputs'] = []
            break

# 3. Atualizar treinamento
for i, cell in enumerate(train_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'model.fit' in source_text and 'history' in source_text:
            print(f"[train] Atualizando treinamento (indice {i})")
            train_nb['cells'][i]['source'] = new_training_source
            train_nb['cells'][i]['outputs'] = []
            break

# 4. Atualizar avaliação
for i, cell in enumerate(train_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'model.evaluate' in source_text and 'test_results' in source_text:
            print(f"[train] Atualizando avaliacao (indice {i})")
            train_nb['cells'][i]['source'] = new_evaluation_source
            train_nb['cells'][i]['outputs'] = []
            break

# 5. Atualizar predição
for i, cell in enumerate(train_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'EXEMPLO DE PREDIÇÃO' in source_text:
            print(f"[train] Atualizando predicao (indice {i})")
            train_nb['cells'][i]['source'] = new_prediction_source
            train_nb['cells'][i]['outputs'] = []
            break

# 6. Atualizar avaliação de hits
for i, cell in enumerate(train_nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'def evaluate_hits_by_prediction_count' in source_text:
            print(f"[train] Atualizando avaliacao de hits (indice {i})")
            train_nb['cells'][i]['source'] = new_eval_hits_source
            train_nb['cells'][i]['outputs'] = []
            break

# Salvar train_model_v2_resnet.ipynb
with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_nb, f, ensure_ascii=False, indent=1)

print("\n" + "="*60)
print("NOTEBOOKS ATUALIZADOS COM SUCESSO!")
print("="*60)
print("\nMudancas:")
print("  - prepare_data_simple.ipynb: Cria e salva vetores Top10/Bottom10")
print("  - train_model_v2_resnet.ipynb:")
print("      * 5 inputs (Sequencia, Freq, Gap, Top10, Bottom10)")
print("      * 4 Blocos ResNet (512 filtros)")
print("      * 8 camadas Attention + LSTM")
print("      * 12 camadas Densas")
print("      * ~10M+ parametros")
