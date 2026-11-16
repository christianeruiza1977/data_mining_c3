# %% [que]
# Se exploran cuatro tendencias sobre el mismo corpus de tweets: BERTopic frente a LDA, un LLM zero-shot y embeddings frente a TF-IDF + SVM, el drift medido con PSI entre un tramo antiguo y uno reciente (ordenado por tweet_id) y el formato Parquet frente a CSV leído con Spark. Todo se ejecutó con datos reales en un entorno con acceso a Hugging Face (no en Colab): BERTopic y los embeddings MiniLM se descargaron de ahí y el LLM zero-shot es un modelo pequeño (Qwen2.5-0.5B-Instruct, 0,5 mil millones de parámetros) en CPU, por lo que sus cifras no representan a un LLM grande. Si se reconstruye el notebook sin acceso a Hugging Face, esos tres bloques se saltan con un aviso y no se inventa ninguna cifra.

# %% [md]
# **Guarda de acceso.** BERTopic y el modelo de embeddings `all-MiniLM-L6-v2` se descargan de Hugging Face, y el LLM puede ser local (Ollama) o un modelo de Hugging Face. Esta celda comprueba (con 5 s de espera) si se puede abrir https://huggingface.co. Si no se puede, cada bloque que lo necesita se salta con un aviso y NO se inventa ningún resultado. En la corrida de este notebook SÍ hubo acceso (`HF_OK = True`, en un entorno con acceso a Hugging Face, no en Colab), así que los tres bloques se ejecutaron; la guarda solo protege a quien reconstruya el notebook sin acceso.

# %% [code]
import os, re, time, json, tempfile, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

def hay_huggingface():
    """True si https://huggingface.co responde en 5 s (en Colab responde; sin acceso devuelve error 403 o timeout)."""
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        return True
    except Exception:
        return False

HF_OK = hay_huggingface()
# El LLM puede correr sin Hugging Face si hay Ollama (variable OLLAMA de la celda del modelo de lenguaje)
LLM_OK = bool(OLLAMA) or HF_OK
AVISO = "No ejecutado en este entorno: sin acceso a Hugging Face; ejecutar en Colab"
print("HF_OK =", HF_OK, "| Ollama disponible =", bool(OLLAMA), "| LLM_OK =", LLM_OK)
if not HF_OK:
    print("AVISO:", AVISO, "(BERTopic y embeddings).")
if not LLM_OK:
    print("AVISO:", AVISO, "(LLM zero-shot: tampoco hay Ollama).")

res9 = {'hf_ok': bool(HF_OK), 'llm_ok': bool(LLM_OK), 'bertopic': {'ejecutado': False}, 'llm_zero_shot': {'ejecutado': False},
        'embeddings': {'ejecutado': False}}   # aquí se acumulan las cifras de toda la fase (results/fase9.json)

# %% [md]
# **a) BERTopic frente a LDA.** BERTopic convierte cada tweet en un embedding, reduce dimensiones (UMAP), agrupa (HDBSCAN) y describe cada grupo con c-TF-IDF (Grootendorst, 2022). A diferencia de LDA, no obliga a elegir K y deja en el tema -1 los tweets que no encajan en ningún grupo. Se usa la misma muestra de 5 000 tweets de la Fase 3 (`muestra_temas`) y `min_topic_size=30`.

# %% [code]
modelo_emb = None
if HF_OK:
    try:
        from sentence_transformers import SentenceTransformer
        modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')   # 384 dimensiones, ~90 MB (se descarga la 1.ª vez)
    except Exception as e:
        print("No se pudo cargar all-MiniLM-L6-v2:", e)
        HF_OK = False
        print("AVISO:", AVISO)

if HF_OK:
    from bertopic import BERTopic
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer
    textos_temas = muestra_temas['texto_limpio'].tolist()
    t0 = time.time()
    embs_temas = modelo_emb.encode(textos_temas, show_progress_bar=False)   # embeddings de los 5 000 tweets
    umap_fijo = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=SEMILLA)  # semilla -> repetible
    bert = BERTopic(embedding_model=modelo_emb, min_topic_size=30, language='english', umap_model=umap_fijo,
                    vectorizer_model=CountVectorizer(stop_words='english'))   # stopwords fuera para palabras clave legibles
    temas_bert, _ = bert.fit_transform(textos_temas, embeddings=embs_temas)   # tema de cada tweet (-1 = sin tema)
    t_bert = time.time() - t0
    temas_bert = np.array(temas_bert)
    info_bert = bert.get_topic_info()
    print(f"BERTopic ajustado en {t_bert:.1f} s")
    print(info_bert.head(12).to_string(index=False))
else:
    print(AVISO, "-> bloque a) BERTopic saltado.")

# %% [md]
# **Comparación BERTopic vs LDA.** Se comparan el número de temas, el porcentaje de tweets sin tema, las palabras clave y, para el cruce con emociones, la «pureza» de cada tema: qué fracción de sus tweets comparte la emoción más frecuente del tema (un valor alto indicaría que el tema captura una emoción; se contrasta con el peso de la emoción más común en toda la muestra).

# %% [code]
if HF_OK:
    emo_muestra = muestra_temas['sentiment']
    n_bert = int((info_bert['Topic'] != -1).sum())
    pct_sin_tema = float((temas_bert == -1).mean() * 100)
    # LDA solo asigna tema a los tweets con palabras útiles tras quitar stopwords; los demás quedan sin tema
    pct_lda_sin_tema = round(100 * (1 - len(tema_dom) / len(muestra_temas)), 1)
    claves_bert = {int(t): [w for w, _ in bert.get_topic(int(t))[:8]] for t in info_bert['Topic'] if t != -1}

    def pureza(temas, emociones):
        """Fracción de tweets (con tema) que pertenecen a la emoción mayoritaria de su tema."""
        tabla = pd.crosstab(pd.Series(temas, name='tema'), emociones.reset_index(drop=True).rename('emocion'))
        return float(tabla.max(axis=1).sum() / tabla.values.sum()), tabla

    tema_bert_s = pd.Series(temas_bert, index=muestra_temas.index)
    con_tema = tema_bert_s != -1
    pureza_bert, cruce_bert = pureza(tema_bert_s[con_tema].values, emo_muestra[con_tema])
    pureza_lda, cruce_lda = pureza(tema_dom.values, muestra_temas.loc[tema_dom.index, 'sentiment'])
    base_emocion = float(emo_muestra.value_counts(normalize=True).iloc[0])
    comparacion = pd.DataFrame({'LDA': [K_LDA, pct_lda_sin_tema, round(pureza_lda, 3)],
                                'BERTopic': [n_bert, round(pct_sin_tema, 1), round(pureza_bert, 3)]},
                               index=['n.º de temas', '% de tweets sin tema', 'pureza de emoción'])
    print(comparacion.to_string())
    print(f"Emoción más común en la muestra: {base_emocion:.3f} (referencia para la pureza)")
    print("\nPalabras clave LDA:")
    for i, p in enumerate(palabras_lda):
        print(f"  tema {i}: {', '.join(p[:8])}")
    print("\nPalabras clave BERTopic:")
    for t, p in claves_bert.items():
        print(f"  tema {t}: {', '.join(p)}")
    print("\nCruce BERTopic (tema x emoción, conteos):")
    print(cruce_bert.to_string())
    res9['bertopic'] = {'ejecutado': True, 'n_temas_bertopic': n_bert, 'n_temas_lda': int(K_LDA),
                        'pct_sin_tema_bertopic': round(pct_sin_tema, 2), 'pct_sin_tema_lda': pct_lda_sin_tema,
                        'pureza_bertopic': round(pureza_bert, 4), 'pureza_lda': round(pureza_lda, 4),
                        'palabras_bertopic': {str(k): v for k, v in claves_bert.items()}, 'tiempo_s': round(t_bert, 1)}
else:
    print(AVISO, "-> comparación BERTopic vs LDA pendiente.")

# %% [md]
# **Qué se analiza:** qué emociones aparecen dentro de cada tema de BERTopic (proporción por fila). Si un tema se concentra en una sola emoción, el tema aporta señal emocional; si se reparte, el tema es temático (cumpleaños, trabajo, música) y no emocional.

# %% [code]
if HF_OK:
    prop = cruce_bert.div(cruce_bert.sum(axis=1), axis=0).loc[cruce_bert.sum(axis=1).sort_values(ascending=False).index[:12]]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(prop.values, cmap='Purples', aspect='auto')
    ax.set_xticks(range(prop.shape[1])); ax.set_xticklabels(prop.columns, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(prop.shape[0])); ax.set_yticklabels([f"{t}: {', '.join(claves_bert[t][:3])}" for t in prop.index], fontsize=8)
    ax.set_title('BERTopic: proporción de cada emoción dentro del tema', color=TINTA, loc='left')
    plt.colorbar(im, ax=ax, label='proporción'); plt.tight_layout()
    plt.savefig('figuras/fase9_bertopic_tema_emocion.png', dpi=150); plt.show()
else:
    print(AVISO, "-> figura BERTopic no generada.")

# %% [md]
# **b) LLM zero-shot.** Se clasifican los 100 primeros tweets de prueba con `preguntar_llm()` (tal como está definida en el notebook) sin entrenar nada, pidiendo UNA palabra de la lista de emociones. Se mide el F1 macro frente al mejor modelo de la Fase 4 en esos mismos 100 tweets, el tiempo por tweet y el porcentaje de respuestas fuera de la lista. Con solo 100 tweets y 13 clases, el F1 macro es muy ruidoso (varias emociones tienen 0 o 1 ejemplo): se calcula solo sobre las emociones presentes en esos 100.

# %% [code]
etiquetas = list(ETIQUETAS)
X100, y100 = X_test.iloc[:100], y_test.iloc[:100]          # los 100 primeros tweets de prueba
if LLM_OK:
    def limpiar_respuesta(r):
        """Toma la primera palabra de la respuesta, en minúscula y sin signos (el LLM a veces añade puntos o frases)."""
        palabras = re.findall(r"[a-z]+", r.lower())
        return palabras[0] if palabras else ''
    pred_llm, tiempos_llm = [], []
    for texto in X100:
        t0 = time.time()
        crudo = preguntar_llm(f"Clasifica la emoción del tweet con UNA palabra de esta lista {etiquetas}. "
                              f"Tweet: {texto}\nEmoción:", max_tokens=5)
        tiempos_llm.append(time.time() - t0)
        pred_llm.append(limpiar_respuesta(crudo))
    fuera_lista = [p not in etiquetas for p in pred_llm]
    pct_fuera = 100 * float(np.mean(fuera_lista))
    pred_llm_ok = [p if p in etiquetas else 'fuera_de_lista' for p in pred_llm]   # una respuesta inválida cuenta como error
    emociones_100 = sorted(y100.unique())                    # solo las emociones que existen en estos 100
    f1_llm = f1_score(y100, pred_llm_ok, average='macro', labels=emociones_100, zero_division=0)
    t0 = time.time()
    pred_clasico_100 = mejor_modelo.predict(X100)            # mejor modelo de la Fase 4 en los MISMOS 100
    t_clasico_100 = time.time() - t0
    f1_clasico_100 = f1_score(y100, pred_clasico_100, average='macro', labels=emociones_100, zero_division=0)
    t_llm_tweet = float(np.mean(tiempos_llm))
    print(f"LLM zero-shot ({MODELO_OLLAMA if OLLAMA else MODELO_HF}): F1 macro = {f1_llm:.3f} | {t_llm_tweet:.2f} s/tweet | fuera de la lista = {pct_fuera:.0f} %")
    print(f"Mejor modelo Fase 4 ({mejor_nombre}): F1 macro = {f1_clasico_100:.3f} | {1000 * t_clasico_100 / 100:.2f} ms/tweet")
    res9['llm_zero_shot'] = {'ejecutado': True, 'modelo': MODELO_OLLAMA if OLLAMA else MODELO_HF, 'f1_macro_llm': round(f1_llm, 4),
                             'f1_macro_mejor_modelo_mismos_100': round(f1_clasico_100, 4), 's_por_tweet_llm': round(t_llm_tweet, 3),
                             'ms_por_tweet_mejor_modelo': round(1000 * t_clasico_100 / 100, 3), 'pct_fuera_de_lista': round(pct_fuera, 1)}
    resultados = pd.concat([resultados, pd.DataFrame([{'modelo': f'LLM zero-shot (100 tweets, {MODELO_OLLAMA if OLLAMA else MODELO_HF})', 'fase': 9,
                            'f1_macro': round(f1_llm, 4), 'accuracy': round(accuracy_score(y100, pred_llm_ok), 4),
                            'tiempo_s': round(float(np.sum(tiempos_llm)), 2)}])], ignore_index=True)
else:
    print(AVISO, "-> bloque b) LLM zero-shot saltado.")

# %% [md]
# **Costo aproximado por 1 000 tweets (este bloque sí se ejecuta).** El tamaño del prompt se mide con el texto real; la conversión a tokens usa la regla práctica ≈ 4 caracteres por token (es una aproximación, no el tokenizador de ningún modelo). NO se incluye ninguna tarifa de proveedor: el costo en dinero es tokens × tarifa vigente del proveedor elegido, que debe consultarse. Para el modelo clásico se mide el tiempo real de predicción de 1 000 tweets.

# %% [code]
def prompt_de(texto):
    return (f"Clasifica la emoción del tweet con UNA palabra de esta lista {etiquetas}. Tweet: {texto}\nEmoción:")
chars_prompt = X_test.iloc[:1000].map(lambda t: len(prompt_de(t)))
tokens_entrada_1000 = float(chars_prompt.sum()) / 4          # aproximación: 4 caracteres ≈ 1 token
t0 = time.time()
_ = mejor_modelo.predict(X_test.iloc[:1000])
t_clasico_1000 = time.time() - t0
print(f"Prompt zero-shot: {chars_prompt.mean():.0f} caracteres por tweet (la lista de 13 emociones y la instrucción ocupan ≈ {len(prompt_de('')):.0f} caracteres fijos)")
print(f"1 000 tweets con LLM por API: ≈ {tokens_entrada_1000:,.0f} tokens de entrada (+ ≈ 5 de salida por tweet); costo = tokens x tarifa del proveedor.")
print(f"1 000 tweets con {mejor_nombre} (TF-IDF): {t_clasico_1000:.3f} s de CPU de predicción, sin enviar datos fuera de la organización.")
res9['costo_1000_tweets'] = {'tokens_entrada_aprox': round(tokens_entrada_1000), 'caracteres_prompt_medio': round(float(chars_prompt.mean()), 1),
                             'tiempo_prediccion_mejor_modelo_s': round(t_clasico_1000, 4)}

# %% [md]
# **c) Embeddings frente a TF-IDF.** Se codifica cada tweet con `all-MiniLM-L6-v2` (vector de 384 dimensiones que captura significado) y se entrena el mismo clasificador lineal (`LinearSVC`) sobre esos vectores. Se elige C en validación y se evalúa en prueba una vez, igual que con TF-IDF. En esta corrida, codificar los 39 688 tweets (entrenamiento, validación y prueba) en CPU tardó unos 94 s; con GPU sería más rápido. Si hiciera falta acortar, se puede entrenar con una sub-muestra de `X_train` (variable `N_ENT_EMB`).

# %% [code]
N_ENT_EMB = None                       # None = todo X_train; p. ej. 10000 para una sub-muestra en CPU
if HF_OK:
    X_ent_emb = X_train if N_ENT_EMB is None else X_train.sample(N_ENT_EMB, random_state=SEMILLA)
    y_ent_emb = y_train.loc[X_ent_emb.index]
    t0 = time.time()
    E_train = modelo_emb.encode(X_ent_emb.tolist(), batch_size=128, show_progress_bar=False)
    E_val = modelo_emb.encode(X_val.tolist(), batch_size=128, show_progress_bar=False)
    E_test = modelo_emb.encode(X_test.tolist(), batch_size=128, show_progress_bar=False)
    t_encode = time.time() - t0
    mejor_c, mejor_f1_val = None, -1
    for C in [0.1, 0.3, 1.0]:                                    # C del SVM lineal, elegido en validación
        f1_v = f1_score(y_val, LinearSVC(C=C, random_state=SEMILLA).fit(E_train, y_ent_emb).predict(E_val), average='macro')
        if f1_v > mejor_f1_val:
            mejor_c, mejor_f1_val = C, f1_v
    t0 = time.time()
    clf_emb = LinearSVC(C=mejor_c, random_state=SEMILLA).fit(E_train, y_ent_emb)
    t_fit_emb = time.time() - t0
    pred_emb = clf_emb.predict(E_test)
    f1_emb, acc_emb = f1_score(y_test, pred_emb, average='macro'), accuracy_score(y_test, pred_emb)
    f1_tfidf = f1_score(y_test, pred_test_mejor, average='macro')   # mejor modelo TF-IDF de la Fase 4
    print(f"Embeddings + LinearSVC (C={mejor_c}): F1 macro = {f1_emb:.3f}, accuracy = {acc_emb:.3f} | codificar {t_encode:.0f} s + entrenar {t_fit_emb:.1f} s")
    print(f"TF-IDF + {mejor_nombre} (Fase 4): F1 macro = {f1_tfidf:.3f}")
    res9['embeddings'] = {'ejecutado': True, 'f1_macro_embeddings': round(f1_emb, 4), 'f1_macro_tfidf_mejor': round(f1_tfidf, 4),
                          'accuracy_embeddings': round(acc_emb, 4), 'C': mejor_c, 'n_train_usado': int(len(X_ent_emb)),
                          'tiempo_codificar_s': round(t_encode, 1), 'tiempo_entrenar_s': round(t_fit_emb, 2)}
    resultados = pd.concat([resultados, pd.DataFrame([{'modelo': f'Embeddings MiniLM + LinearSVC (C={mejor_c})', 'fase': 9,
                            'f1_macro': round(f1_emb, 4), 'accuracy': round(acc_emb, 4), 'tiempo_s': round(t_encode + t_fit_emb, 2)}])], ignore_index=True)
else:
    print(AVISO, "-> bloque c) embeddings saltado.")

# %% [md]
# **d) Drift con PSI.** El drift ocurre cuando los datos de producción dejan de parecerse a los de entrenamiento. Se ordenan los tweets por `tweet_id` (proxy del tiempo: los ids de Twitter crecen con la fecha), se toma el 70 % más antiguo como «tramo antiguo» y el 30 % final como «tramo reciente». Se entrena TF-IDF + LinearSVC con el 80 % del tramo antiguo y se reserva el 20 % restante como referencia (hold-out). El PSI compara la distribución de clases PREDICHAS en la referencia (esperada) con la del tramo reciente (actual): PSI = Σ (a% − e%) · ln(a% / e%), con un epsilon para evitar ceros. Lectura habitual: < 0.1 sin cambio importante; 0.1–0.25 cambio moderado; > 0.25 cambio fuerte (hay que reentrenar o investigar).

# %% [code]
from sklearn.model_selection import train_test_split
ordenado = df.sort_values('tweet_id').reset_index(drop=True)           # orden por tweet_id
corte = int(len(ordenado) * 0.70)
antiguo, reciente = ordenado.iloc[:corte], ordenado.iloc[corte:]       # 70 % antiguo, 30 % reciente
ent_a, ref_a = train_test_split(antiguo, test_size=0.20, stratify=antiguo['sentiment'], random_state=SEMILLA)  # entrenamiento y referencia

def psi(esperado, actual, clases, epsilon=1e-4):
    """PSI entre dos distribuciones de clases. epsilon evita log(0) y divisiones por cero."""
    e = np.array([np.mean(np.array(esperado) == c) for c in clases]) + epsilon
    a = np.array([np.mean(np.array(actual) == c) for c in clases]) + epsilon
    e, a = e / e.sum(), a / a.sum()                                    # vuelven a sumar 1 tras el epsilon
    return float(np.sum((a - e) * np.log(a / e)))

def semaforo(valor):
    return 'sin cambio importante' if valor < 0.1 else ('cambio moderado' if valor < 0.25 else 'cambio fuerte')

t0 = time.time()
modelo_drift = Pipeline([('tfidf', TfidfVectorizer(min_df=2, sublinear_tf=True)),
                         ('clf', LinearSVC(C=0.3, random_state=SEMILLA))]).fit(ent_a['texto_limpio'], ent_a['sentiment'])
t_drift = time.time() - t0
pred_ref = modelo_drift.predict(ref_a['texto_limpio'])
pred_rec = modelo_drift.predict(reciente['texto_limpio'])
psi_pred = psi(pred_ref, pred_rec, ETIQUETAS)
psi_real = psi(ref_a['sentiment'], reciente['sentiment'], ETIQUETAS)   # drift en las etiquetas verdaderas (informativo)
f1_ref = f1_score(ref_a['sentiment'], pred_ref, average='macro')
f1_rec = f1_score(reciente['sentiment'], pred_rec, average='macro')
print(f"tweet_id: antiguo {antiguo['tweet_id'].min()}-{antiguo['tweet_id'].max()} ({len(antiguo)}), reciente {reciente['tweet_id'].min()}-{reciente['tweet_id'].max()} ({len(reciente)})")
print(f"Entrenamiento: {len(ent_a)} tweets | referencia: {len(ref_a)} | reciente: {len(reciente)} | modelo entrenado en {t_drift:.1f} s")
print(f"PSI de las predicciones (referencia vs reciente) = {psi_pred:.4f} -> {semaforo(psi_pred)}")
print(f"PSI de las etiquetas reales (informativo)         = {psi_real:.4f} -> {semaforo(psi_real)}")
print(f"F1 macro: referencia (tramo antiguo) = {f1_ref:.3f} | tramo reciente = {f1_rec:.3f} | diferencia = {f1_rec - f1_ref:+.3f}")

# %% [md]
# **Qué se analiza:** cómo se reparten las clases predichas por el modelo en la referencia (tramo antiguo) y en el tramo reciente. Barras muy parecidas implican PSI bajo; una barra que cambia mucho (en ACENTO, la emoción con mayor aporte al PSI) indica dónde se movió la distribución.

# %% [code]
dist_ref = pd.Series(pred_ref).value_counts(normalize=True).reindex(ETIQUETAS, fill_value=0) * 100
dist_rec = pd.Series(pred_rec).value_counts(normalize=True).reindex(ETIQUETAS, fill_value=0) * 100
aporte = (dist_rec / 100 - dist_ref / 100) * np.log((dist_rec / 100 + 1e-4) / (dist_ref / 100 + 1e-4))   # aporte de cada clase al PSI
clave = aporte.abs().idxmax()                                           # clase que más aporta
x = np.arange(len(ETIQUETAS)); ancho = 0.4
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(x - ancho / 2, dist_ref.values, ancho, color=GRIS, label='Referencia (tramo antiguo)')
ax.bar(x + ancho / 2, dist_rec.values, ancho, color=[ACENTO if e == clave else TINTA for e in ETIQUETAS], label='Tramo reciente')
ax.set_xticks(x); ax.set_xticklabels(ETIQUETAS, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('% de predicciones', color=TINTA)
ax.set_title(f'Distribución de clases predichas: PSI = {psi_pred:.3f} ({semaforo(psi_pred)}); mayor aporte: {clave}', color=TINTA, loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.spines[['top', 'right']].set_visible(False); ax.grid(False)
plt.tight_layout(); plt.savefig('figuras/fase9_drift_distribuciones.png', dpi=150); plt.show()
res9['drift'] = {'psi_predicciones': round(psi_pred, 4), 'interpretacion': semaforo(psi_pred), 'psi_etiquetas_reales': round(psi_real, 4),
                 'f1_macro_referencia': round(f1_ref, 4), 'f1_macro_reciente': round(f1_rec, 4), 'n_entrenamiento': int(len(ent_a)),
                 'n_referencia': int(len(ref_a)), 'n_reciente': int(len(reciente)), 'clase_mayor_aporte': clave,
                 'tweet_id_corte': int(reciente['tweet_id'].min()),
                 'pct_predicho_referencia': dist_ref.round(1).to_dict(), 'pct_predicho_reciente': dist_rec.round(1).to_dict()}

# %% [md]
# **e) Parquet y Spark.** Parquet es un formato columnar comprimido: guarda cada columna junta, la comprime mejor y Spark solo lee las columnas que necesita. Se guarda la tabla `resultados` en `data/resultados_emociones.parquet` (pandas + pyarrow), se lee con la sesión de Spark de la Fase 7 y se comparan tamaños con el CSV equivalente. Como la tabla de resultados es diminuta, se repite la comparación con el corpus completo (tweet_id, sentiment, texto_limpio) en CSV frente a Parquet con compresión snappy.

# %% [code]
try:
    spark                                    # la sesión de la Fase 7 sigue activa
except NameError:                            # si se ejecuta esta fase sola
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('emociones').master('local[*]').config('spark.driver.memory', '4g').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

RUTA_PQ = 'data/resultados_emociones.parquet'
resultados.to_parquet(RUTA_PQ, engine='pyarrow', index=False)          # pandas -> Parquet
print("Tabla de resultados guardada en Parquet:", resultados.shape)
spark.read.parquet(RUTA_PQ).groupBy('modelo').avg('f1_macro').show(truncate=False)   # leída con Spark

def tam_mb(ruta):
    """Tamaño en MB de un archivo o de una carpeta."""
    if os.path.isdir(ruta):
        return sum(os.path.getsize(os.path.join(ruta, f)) for f in os.listdir(ruta)) / 1e6
    return os.path.getsize(ruta) / 1e6

with tempfile.TemporaryDirectory() as tmp:                             # archivos de la comparación: se borran al terminar
    csv_res, csv_cor, pq_cor = f'{tmp}/resultados.csv', f'{tmp}/corpus.csv', f'{tmp}/corpus.parquet'
    resultados.to_csv(csv_res, index=False)
    corpus = df[['tweet_id', 'sentiment', 'texto_limpio']]
    corpus.to_csv(csv_cor, index=False)
    corpus.to_parquet(pq_cor, engine='pyarrow', compression='snappy', index=False)
    tam = {'resultados_csv_mb': tam_mb(csv_res), 'resultados_parquet_mb': tam_mb(RUTA_PQ),
           'corpus_csv_mb': tam_mb(csv_cor), 'corpus_parquet_snappy_mb': tam_mb(pq_cor)}
    # Lectura con Spark del corpus en Parquet: conteo y plan (se ve que solo lee las columnas necesarias)
    sdf_pq = spark.read.parquet(pq_cor)
    n_pq = sdf_pq.count()
    print("Filas leídas por Spark desde el corpus en Parquet:", n_pq, "(igual que df:", n_pq == len(df), ")")
    sdf_pq.groupBy('sentiment').count().explain(mode='formatted')

razon_res = tam['resultados_csv_mb'] / tam['resultados_parquet_mb']
razon_cor = tam['corpus_csv_mb'] / tam['corpus_parquet_snappy_mb']
tabla_tam = pd.DataFrame({'CSV (MB)': [tam['resultados_csv_mb'], tam['corpus_csv_mb']],
                          'Parquet (MB)': [tam['resultados_parquet_mb'], tam['corpus_parquet_snappy_mb']],
                          'razón CSV/Parquet': [razon_res, razon_cor]}, index=['tabla resultados', 'corpus completo (snappy)']).round(6)
print(tabla_tam.to_string())

# %% [md]
# **Qué se analiza:** el tamaño en disco del mismo dato en CSV y en Parquet. En la tabla diminuta de resultados Parquet no ayuda (los metadatos pesan más que los datos); en el corpus completo, sí.

# %% [code]
fig, ax = plt.subplots(figsize=(7, 3.6))
etiq = ['tabla resultados', 'corpus completo']
ancho = 0.38; x = np.arange(2)
ax.bar(x - ancho / 2, tabla_tam['CSV (MB)'], ancho, color=GRIS, label='CSV')
ax.bar(x + ancho / 2, tabla_tam['Parquet (MB)'], ancho, color=ACENTO, label='Parquet (snappy)')
for i in range(2):
    ax.text(i - ancho / 2, tabla_tam['CSV (MB)'].iloc[i], f"{tabla_tam['CSV (MB)'].iloc[i]:.3g}", ha='center', va='bottom', fontsize=8, color=TINTA)
    ax.text(i + ancho / 2, tabla_tam['Parquet (MB)'].iloc[i], f"{tabla_tam['Parquet (MB)'].iloc[i]:.3g}", ha='center', va='bottom', fontsize=8, color=TINTA)
ax.set_xticks(x); ax.set_xticklabels(etiq)
ax.set_ylabel('MB en disco', color=TINTA)
ax.set_title('CSV vs Parquet', color=TINTA, loc='left')
ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False); ax.grid(False)
plt.tight_layout(); plt.savefig('figuras/fase9_tamano_csv_parquet.png', dpi=150); plt.show()
res9['parquet'] = {k: round(v, 6) for k, v in tam.items()}
res9['parquet'].update({'razon_resultados': round(razon_res, 3), 'razon_corpus': round(razon_cor, 3), 'filas_resultados': int(len(resultados))})
with open('results/fase9.json', 'w', encoding='utf-8') as f:
    json.dump(res9, f, ensure_ascii=False, indent=2, default=str)
print("Guardado results/fase9.json")

# %% [md]
# **Cierre: ¿cuándo se justifica un LLM o un modelo de embeddings frente a TF-IDF + SVM?** Se resume en cuatro criterios, con las cifras medidas en esta fase.
# - **Costo.** TF-IDF + SVM entrena en 1,7 s y predice 1 000 tweets en 0,019 s de CPU; su costo marginal es casi cero. Los embeddings quedan en medio: codificar los 39 688 tweets tardó 94 s y entrenar el clasificador 49 s en CPU (≈ 143 s), pero después cada texto es un vector reutilizable (búsqueda, agrupamiento, clasificación). Un LLM por API cobra por token (el prompt de este ejercicio ocupa unos 276 caracteres por tweet, ≈ 69 000 tokens de entrada por cada 1 000 tweets con la regla aproximada de 4 caracteres por token); el costo en dinero es tokens × tarifa del proveedor, que no se asume aquí. Un LLM local necesita GPU o paga latencia.
# - **Latencia.** El SVM respondió en 0,04 ms por tweet; el LLM pequeño de 0,5B en CPU, en 0,33 s (del orden de 9 000 veces más lento). Un LLM grande por API puede ser más rápido o más lento según el proveedor, pero no se midió. Para moderación en línea o paneles en tiempo real esto pesa mucho.
# - **Privacidad (LOPDP, Ecuador).** Enviar tweets con datos personales a una API externa es un tratamiento y posible transferencia internacional de datos personales, que exige base legítima, deber de información y garantías con el proveedor. Un modelo clásico o un LLM/embedding local (Ollama, `sentence-transformers` en servidor propio, como en esta fase: solo se descargaron los pesos) mantiene los datos dentro de la organización.
# - **Mantenimiento.** TF-IDF + SVM se reentrena en segundos y es explicable (palabras con más peso). Un LLM cambia con las versiones del proveedor, requiere vigilar el prompt y las respuestas fuera de la lista (87 % en nuestra prueba con el modelo pequeño) y es más difícil de auditar.
#
# **Resultado honesto de esta comparación:** TF-IDF + SVM ganó en F1 macro (0,198 frente a 0,159 de embeddings y 0,030 del LLM pequeño, este último medido solo en 100 tweets). Los embeddings lograron la accuracy más alta (0,356) pero peor F1 macro, y el LLM de 0,5B en CPU no sirve para esta tarea tal como se probó; no representa a un LLM grande.
#
# **Regla práctica:** TF-IDF + SVM como línea base y para volumen alto con latencia baja; embeddings cuando importe el significado más allá de las palabras (sinónimos, poco texto etiquetado, búsqueda semántica) y se pueda pagar la codificación; LLM zero-shot cuando no hay datos etiquetados, el volumen es pequeño o se necesita explicar/resumir, y el dato puede salir (o el modelo corre en local).
#
# **Fuentes.** Lectura de tendencia de la semana: *¿LLM o embeddings? El dilema del costo en la minería de textos* (se cita por su título y su idea general: comparar el costo de usar LLM frente a embeddings en tareas de minería de textos; no se reproducen datos de la lectura). Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794. Documentación de spacy-llm (spaCy): integra LLM como componentes de un pipeline de spaCy (tareas como clasificación de texto con prompts), https://spacy.io/usage/large-language-models.

# %% [conclusion]
# **Conclusión de negocio (Fase 9).**
# **Cómo se ejecutó.** Todos los bloques de esta fase (BERTopic, LLM zero-shot, embeddings, drift con PSI y Parquet) se ejecutaron con datos reales en un entorno con acceso a Hugging Face (no en Colab). Los tweets no se enviaron a ningún servicio externo: solo se descargaron los pesos de los modelos y todo corrió en local. El LLM es un modelo pequeño (Qwen2.5-0.5B-Instruct, 0,5 mil millones de parámetros) en CPU: sus cifras NO representan a un LLM grande.
# 1. **BERTopic frente a LDA (misma muestra de 5 000 tweets).** BERTopic encontró 25 temas (`min_topic_size=30`) frente a 5 de LDA, pero dejó 47,2 % de los tweets sin tema (2 358 en el tema -1) frente a 5,0 % de LDA (los 252 tweets sin palabras útiles; LDA fuerza un tema a todos los demás). Los temas de BERTopic son mucho más específicos (transporte, teléfonos, comida, dolor, música) que los de LDA (mezclas como `good, thanks, nice`). La pureza emocional (fracción de los tweets con tema que comparte la emoción mayoritaria de su tema) sube de 0,229 (LDA) a 0,296 (BERTopic), pero la emoción más común de la muestra ya pesa 0,222 y BERTopic tiene temas más pequeños (donde la pureza sube con más facilidad), así que ambos agrupan sobre todo por asunto y no por emoción. Tres temas de BERTopic: el tema 4 (`day, happy, mothers, mother, mom, flowers`; 173 tweets, Día de la Madre) concentra `love` 63 (36 %) y `happiness` 37 (21 %); el tema 5 (`hurts, pain, headache, stomach, sore, hospital, sick`; 162 tweets, dolor y salud) concentra `worry` 82 (51 %) y `sadness` 26 (16 %); el tema 0 (`car, bus, london, traffic, bike`; 225 tweets) es puramente temático (`worry` 24 %, `neutral` 20 %). BERTopic tardó 45 s en ajustarse (LDA final: 7,8 s). Lectura de negocio: sirve para explorar de qué hablan los clientes con mayor detalle y para saber qué mensajes no encajan en ningún tema, no para predecir la emoción.
# 2. **LLM zero-shot (100 tweets de prueba, sin entrenar).** F1 macro de 0,030 frente a 0,176 del mejor modelo (SVM) en esos mismos 100 tweets (calculado solo sobre las emociones presentes en esos 100; con tan pocos tweets y 13 clases el valor es ruidoso). Tardó 0,33 s por tweet frente a 0,04 ms del SVM (del orden de 9 000 veces más lento) y el 87 % de sus respuestas quedó fuera de la lista de 13 emociones (cuentan como error). Es un modelo pequeño de 0,5B corriendo en CPU, así que este resultado NO permite concluir nada sobre un LLM grande o con GPU, que probablemente siga mejor las instrucciones; eso no se midió aquí. Costo: el prompt ocupa unos 276 caracteres por tweet (≈ 69 000 tokens de entrada por cada 1 000 tweets, con la regla de 4 caracteres por token, que es una aproximación); el costo en dinero de una API es tokens × tarifa del proveedor, que no se asume. El SVM predijo 1 000 tweets en 0,019 s.
# 3. **Embeddings frente a TF-IDF.** MiniLM (`all-MiniLM-L6-v2`) + LinearSVC (C = 1,0) obtuvo en prueba F1 macro de 0,159 frente a 0,198 de TF-IDF + SVM; quedó por encima de NB (0,144) y de la RNN (0,149), pero por debajo del SVM. Su accuracy fue la más alta de todos los modelos (0,356 frente a 0,270 del SVM y 0,299 de NB), lo que sugiere que acierta más en las clases grandes y menos en las raras (no se midió el F1 por clase de este modelo). Costó mucho más: codificar los 39 688 tweets tardó 94 s y entrenar 49 s en CPU (≈ 143 s) frente a 1,7 s del SVM. Ojo con la comparación: el clasificador sobre embeddings se ajustó sin `class_weight='balanced'` (el SVM de la Fase 4 sí lo usa), así que parte de la diferencia podría deberse a eso; no se probó. Con esta evidencia, TF-IDF + SVM ganó en F1 macro y en costo.
# 4. **Drift: hay un cambio fuerte.** Ordenando por `tweet_id`, se entrenó TF-IDF + LinearSVC con 22 224 tweets del tramo antiguo (70 % = 27 781) y se midió en 5 557 tweets de referencia y en los 11 907 más recientes. El PSI de las clases predichas es 0,350, por encima del umbral de 0,25 («cambio fuerte»; el de las etiquetas reales, solo informativo, es 0,519). El modelo predice mucho más «worry» (21,2 % → 37,1 %) y «sadness» (6,8 % → 14,8 %) y mucho menos «happiness» (20,0 % → 8,0 %, la clase que más aporta al PSI), «love» (11,2 % → 4,2 %) y «neutral» (36,5 % → 31,3 %). El F1 macro baja de 0,166 en la referencia a 0,142 en el tramo reciente (−0,024). Lectura de negocio: el tono de los tweets cambió con el tiempo (más preocupación y tristeza), así que un modelo entrenado solo con el pasado subestima la alegría; conviene vigilar el PSI de forma periódica y reentrenar cuando supere 0,25. Es un PSI sobre predicciones: indica que cambió el dato o el mundo, no por sí solo que el modelo esté peor (eso lo dice el F1, que sí cayó).
# 5. **Parquet.** La tabla de resultados (6 filas: NB, SVM, RNN, Spark MLlib, LLM y embeddings) es diminuta: en CSV ocupa ≈ 0,43 KB y en Parquet ≈ 3,8 KB (razón CSV/Parquet = 0,11: Parquet pesa MÁS, porque guarda metadatos de esquema). Con el corpus completo (39 688 tweets, `tweet_id` + `sentiment` + `texto_limpio`) el CSV pesa 3,39 MB y el Parquet con snappy 2,25 MB (razón 1,51, un 34 % menos). Spark lee el Parquet directamente (39 688 filas, igual que `df`) y el plan muestra `ReadSchema` con solo las columnas necesarias. La ganancia de tamaño es moderada porque el dato es sobre todo texto libre, difícil de comprimir; Parquet vale más por el esquema tipado y la lectura por columnas que por el tamaño en este corpus.
# **Discusión de cierre: costo, latencia y privacidad (LOPDP).** Con la evidencia de este notebook, TF-IDF + SVM ganó en F1 macro (0,198 frente a 0,159 de embeddings y 0,030 del LLM pequeño) y también en costo (1,7 s de entrenamiento) y latencia (0,04 ms por tweet frente a 0,33 s del LLM). Los embeddings quedan como candidato si importa el significado más allá de las palabras (con otro clasificador o con pesos de clase habría que volver a medirlos); el LLM zero-shot pequeño no sirve para esta tarea tal como se probó. En privacidad, enviar mensajes de clientes a una API externa sería un tratamiento y una posible transferencia internacional de datos personales (LOPDP: base legítima, información al titular, garantías con el proveedor); los modelos de esta fase corrieron en local y no sacaron ningún tweet. Los datos del ejercicio son tweets públicos en inglés; con mensajes reales de clientes esa exigencia es mayor.
# **Decisión.** Mantener TF-IDF + SVM como modelo, añadir monitoreo del drift con PSI (reentrenar si supera 0,25) y usar BERTopic como herramienta de exploración de temas. Un LLM grande (por API o en local) o un clasificador sobre embeddings con ajuste de desbalance solo se recomendaría si una nueva prueba, con más tweets y las mismas métricas, supera el F1 macro del SVM de forma que justifique su costo y su latencia.
