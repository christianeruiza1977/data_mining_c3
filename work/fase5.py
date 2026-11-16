# %% [que]
# Se entrena una red recurrente en Keras (TextVectorization, Embedding, GRU bidireccional, softmax de 13 clases) con la misma partición de la Fase 4, parada temprana sobre validación y pesos de clase para el desbalance. Se compara su F1 macro, accuracy, tiempo de entrenamiento y tamaño contra NB y SVM.

# %% [md]
# **Configuración.** Se fija la semilla de TensorFlow y se limitan los hilos para no saturar la CPU compartida. La función `entrenar_rnn` encapsula todo el modelo para poder reutilizarla en la Fase 6 con otros subconjuntos de clases.

# %% [code]
import os, time, json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np, pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

tf.keras.utils.set_random_seed(SEMILLA)           # semilla de numpy, python y TF
tf.config.threading.set_intra_op_parallelism_threads(2)   # limita hilos (CPU compartida)
tf.config.threading.set_inter_op_parallelism_threads(2)

def entrenar_rnn(X_tr, y_tr, X_va, y_va, X_te, y_te, clases, max_tokens=20000, largo=40,
                 dim_emb=64, unidades=64, max_epocas=10, paciencia=2, batch=128, verbose=0):
    """Entrena TextVectorization -> Embedding -> BiGRU -> Dropout -> softmax.
    Early stopping sobre val_loss y class_weight balanceado. Evalúa en prueba una vez."""
    tf.keras.utils.set_random_seed(SEMILLA)
    mapa = {c: i for i, c in enumerate(clases)}                 # etiqueta -> entero (mismo orden que ETIQUETAS)
    ytr, yva, yte = (np.array([mapa[c] for c in y]) for y in (y_tr, y_va, y_te))

    vectorizador = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=largo)
    vectorizador.adapt(np.array(X_tr))                          # el vocabulario se aprende solo con train
    vec = lambda x: vectorizador(np.array(x)).numpy()
    Xtr, Xva, Xte = vec(X_tr), vec(X_va), vec(X_te)

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(largo,), dtype='int64'),
        tf.keras.layers.Embedding(max_tokens, dim_emb, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(unidades)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(clases), activation='softmax')])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    # Pesos de clase 'balanced': n / (k * n_clase); las clases raras pesan más en la pérdida
    pesos = compute_class_weight('balanced', classes=np.arange(len(clases)), y=ytr)
    pesos = dict(enumerate(pesos))
    parada = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=paciencia, restore_best_weights=True)

    t0 = time.perf_counter()
    hist = modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=max_epocas, batch_size=batch,
                      class_weight=pesos, callbacks=[parada], verbose=verbose)
    tiempo = time.perf_counter() - t0

    pred = modelo.predict(Xte, batch_size=512, verbose=0).argmax(axis=1)   # única evaluación en prueba
    return {'modelo': modelo, 'historial': hist.history, 'tiempo_s': tiempo,
            'epocas': len(hist.history['loss']), 'mejor_epoca': int(np.argmin(hist.history['val_loss'])) + 1,
            'parametros': int(modelo.count_params()), 'pesos': pesos,
            'f1_macro': f1_score(yte, pred, average='macro', labels=np.arange(len(clases)), zero_division=0),
            'accuracy': accuracy_score(yte, pred), 'pred': np.array(clases)[pred]}

# %% [md]
# **Entrenamiento con las 13 emociones.** Vocabulario de 20 000 tokens, secuencias de 40 palabras, embedding de 64 y GRU bidireccional de 64 unidades; máximo 10 épocas, lote de 128, paciencia 2 sobre `val_loss`.

# %% [code]
res_rnn = entrenar_rnn(X_train, y_train, X_val, y_val, X_test, y_test, ETIQUETAS, verbose=2)
print(f"\nÉpocas ejecutadas: {res_rnn['epocas']} (mejor época: {res_rnn['mejor_epoca']})")
print(f"Tiempo de entrenamiento: {res_rnn['tiempo_s']:.1f} s | parámetros: {res_rnn['parametros']:,}")
print(f"Prueba -> F1 macro: {res_rnn['f1_macro']:.4f}, accuracy: {res_rnn['accuracy']:.4f}")

# Pesos de clase aplicados (efecto del balanceo)
pesos_tabla = pd.Series({ETIQUETAS[i]: p for i, p in res_rnn['pesos'].items()}).sort_values()
print("\nPeso de clase 'balanced' (neutral pesa poco, anger mucho):")
print(pesos_tabla.round(2).to_string())

# %% [md]
# **Qué se analiza:** la curva de pérdida (entropía cruzada) en entrenamiento y validación por época. Sirve para ver cuándo empieza el sobreajuste y por qué la parada temprana recupera los pesos de la mejor época (marcada en ACENTO).

# %% [code]
TINTA, ACENTO, GRIS = "#14323F", "#6E5BD6", "#B7C3C9"
h = res_rnn['historial']
epocas = np.arange(1, len(h['loss']) + 1)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(epocas, h['loss'], color=GRIS, marker='o', label='Entrenamiento (con pesos de clase)')
ax.plot(epocas, h['val_loss'], color=ACENTO, marker='o', label='Validación')
ax.axvline(res_rnn['mejor_epoca'], color=TINTA, ls='--', lw=1)
ax.text(res_rnn['mejor_epoca'] + 0.05, max(h['loss']) * 0.98, 'mejor época', color=TINTA, fontsize=8, va='top')
ax.set_xlabel('Época', color=TINTA); ax.set_ylabel('Pérdida', color=TINTA); ax.legend(frameon=False)
ax.set_xticks(epocas)
for lado in ['top', 'right']:
    ax.spines[lado].set_visible(False)
plt.tight_layout(); plt.savefig('figuras/fase5_curva_entrenamiento.png', dpi=150); plt.show()

# %% [md]
# **Comparación NB vs SVM vs RNN.** Se añade la fila de la RNN a `resultados` y se arma una tabla con F1 macro, accuracy, tiempo de entrenamiento y tamaño (parámetros). Para NB y SVM el tamaño es el número de coeficientes aprendidos.

# %% [code]
resultados = pd.concat([resultados, pd.DataFrame([{'modelo': 'RNN', 'fase': 5,
                        'f1_macro': res_rnn['f1_macro'], 'accuracy': res_rnn['accuracy'],
                        'tiempo_s': res_rnn['tiempo_s']}])], ignore_index=True)

def n_parametros(pipe):
    # NB: log-probabilidades por clase y término; SVM: coeficientes + sesgos
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_log_prob_'):
        return int(clf.feature_log_prob_.size + clf.class_log_prior_.size)
    return int(clf.coef_.size + clf.intercept_.size)

params = {'NB': n_parametros(modelos_finales['NB']), 'SVM': n_parametros(modelos_finales['SVM']),
          'RNN': res_rnn['parametros']}
comparacion = resultados[resultados.fase.isin([4, 5])].copy()
comparacion['parametros'] = comparacion['modelo'].map(params)
print(comparacion.round(4).to_string(index=False))
rel = comparacion.set_index('modelo')
print(f"\nLa RNN tarda {rel.loc['RNN','tiempo_s']/rel.loc['SVM','tiempo_s']:.0f} veces más que el SVM "
      f"y {rel.loc['RNN','tiempo_s']/rel.loc['NB','tiempo_s']:.0f} veces más que NB.")
print(f"Diferencia de F1 macro RNN - SVM: {rel.loc['RNN','f1_macro']-rel.loc['SVM','f1_macro']:+.4f}")

# %% [md]
# **Qué se analiza:** F1 por clase de la RNN frente al SVM, para ver si el balanceo con pesos ayuda en las emociones raras o solo reparte los errores de otra forma.

# %% [code]
f1_rnn = pd.Series(f1_score(y_test, res_rnn['pred'], labels=ETIQUETAS, average=None), index=ETIQUETAS)
tabla_clases = pd.DataFrame({'SVM': f1_por_clase['SVM'], 'RNN': f1_rnn, 'n_prueba': f1_por_clase['n_prueba']})
tabla_clases = tabla_clases.sort_values('n_prueba', ascending=False)
print(tabla_clases.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 4.6))
pos = np.arange(len(tabla_clases)); ancho = 0.4
ax.bar(pos - ancho/2, tabla_clases['SVM'], ancho, color=GRIS, label='SVM')
ax.bar(pos + ancho/2, tabla_clases['RNN'], ancho, color=ACENTO, label='RNN')
ax.set_xticks(pos); ax.set_xticklabels(tabla_clases.index, rotation=45, ha='right', color=TINTA)
ax.set_ylabel('F1 por clase (prueba)', color=TINTA); ax.legend(frameon=False)
for lado in ['top', 'right']:
    ax.spines[lado].set_visible(False)
plt.tight_layout(); plt.savefig('figuras/fase5_f1_por_clase_rnn_svm.png', dpi=150); plt.show()

# %% [code]
salida5 = {'rnn': {'f1_macro': res_rnn['f1_macro'], 'accuracy': res_rnn['accuracy'],
                   'tiempo_s': res_rnn['tiempo_s'], 'epocas': res_rnn['epocas'],
                   'mejor_epoca': res_rnn['mejor_epoca'], 'parametros': res_rnn['parametros'],
                   'config': {'max_tokens': 20000, 'largo': 40, 'dim_emb': 64, 'unidades': 64,
                              'batch': 128, 'class_weight': 'balanced', 'paciencia': 2}},
           'peso_clase': {k: round(float(v), 3) for k, v in pesos_tabla.items()},
           'comparacion': comparacion.round(5).to_dict('records'),
           'f1_por_clase': {'RNN': f1_rnn.round(4).to_dict(), 'SVM': f1_por_clase['SVM'].round(4).to_dict()},
           'historial': {k: [round(float(x), 5) for x in v] for k, v in h.items()}}
with open('results/fase5.json', 'w') as archivo:
    json.dump(salida5, archivo, indent=2)

# %% [conclusion]
# **Conclusión de negocio (Fase 5).** La RNN (embedding 64, GRU bidireccional de 64, class_weight balanceado) NO supera al SVM: obtiene F1 macro de 0.149 y accuracy de 0.176 en prueba, frente a 0.198 y 0.270 del SVM y 0.144 y 0.299 de NB; queda prácticamente empatada con NB en F1 macro y por debajo en accuracy. Además es mucho más cara: 95 s de entrenamiento en 5 épocas (la mejor fue la 3; la pérdida de validación empieza a subir desde ahí mientras la de entrenamiento sigue bajando, es decir, sobreajuste rápido) frente a 1.7 s del SVM y 0.7 s de NB (alrededor de 55 y 140 veces más, con variaciones entre corridas por la CPU compartida), y con 1.33 millones de parámetros frente a 0.50 millones de coeficientes. Para el desbalance se usaron pesos de clase 'balanced' (neutral y worry pesan 0.36; boredom 17.1 y anger 27.8): eso obliga al modelo a prestar atención a las emociones pequeñas, pero baja la accuracy porque sacrifica las grandes (F1 de neutral 0.10 y worry 0.11, contra 0.31 y 0.32 del SVM) y, con solo 110 tweets de anger, los pesos altos tienden a memorizar esos ejemplos. La alternativa de muestreo (sobremuestrear clases raras o submuestrear neutral/worry) tendría efectos parecidos y no se probó. La RNN solo gana al SVM en sadness (0.30 vs 0.29) y boredom, con diferencias pequeñas. Recomendación: con 28 mil tweets cortos y etiquetas ruidosas, un modelo lineal TF-IDF es la opción de mejor costo-beneficio; una RNN solo se justificaría con más datos o embeddings preentrenados.
