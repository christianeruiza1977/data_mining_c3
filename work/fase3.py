# %% [que]
# Se aplica LDA (scikit-learn) sobre la bolsa de palabras de `muestra_temas` (5 000 tweets, la misma que usará BERTopic en la Fase 9). El número de temas se elige con la perplejidad sobre un conjunto de validación (80/20) con la coherencia NPMI como confirmación (rejilla de K = 5 a 20); luego se interpretan los temas y se cruzan con las emociones.

# %% [code]
# Muestra de 5 000 tweets: es la MISMA que usará BERTopic en la Fase 9, para poder comparar ambos métodos
muestra_temas = df.sample(5000, random_state=SEMILLA)
print(muestra_temas.shape)

# %% [code]
import time, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
t_ini_fase3 = time.time()
TINTA, ACENTO, GRIS = "#14323F", "#6E5BD6", "#B7C3C9"

# Bolsa de palabras: stop words en inglés + tokens basura típicos de tweets (contracciones partidas, risas, etc.)
basura = ['don', 'll', 've', 're', 'm', 's', 'd', 't', 'im', 'ur', 'u', 'lol', 'haha', 'quot', 'amp', 'just', 'got', 'like', 'get', 'know', 'think', 'want', 'did', 'does', 'doesn', 'didn', 'isn', 'wasn', 'won', 'cant', 'dont']
stop_lda = list(ENGLISH_STOP_WORDS.union(basura))
vec_lda = CountVectorizer(stop_words=stop_lda, min_df=5, max_df=0.5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")   # solo palabras de 2+ letras; min_df=5 evita palabras rarísimas; max_df=0.5 quita las muy comunes
bolsa = vec_lda.fit_transform(muestra_temas['texto_limpio'])
vocab = np.array(vec_lda.get_feature_names_out())
# tweets que quedaron sin ninguna palabra útil no aportan a LDA
n_vacios = int((bolsa.sum(axis=1) == 0).sum())
print("Matriz documentos x palabras:", bolsa.shape, "| tweets sin palabras útiles:", n_vacios)

# %% [md]
# **Qué se analiza:** cómo cambian dos medidas de calidad al aumentar el número de temas K: la perplejidad en un 20 % de tweets apartado (menor = mejor) y la coherencia NPMI de las 10 palabras principales de cada tema (mayor = temas más interpretables). Criterio explícito: se elige el K con menor perplejidad de validación, siempre que la coherencia NPMI lo confirme; la rejilla parte de K = 5 como mínimo para no fundir todos los asuntos en 2-3 temas.

# %% [code]
# Partición 80/20 de la muestra (la bolsa de palabras ya está construida con todo el vocabulario)
idx_a, idx_v = train_test_split(np.arange(bolsa.shape[0]), test_size=0.20, random_state=SEMILLA)
bolsa_ent, bolsa_val = bolsa[idx_a], bolsa[idx_v]

# Coherencia NPMI: ¿las palabras principales de un tema aparecen juntas en los mismos tweets?
presencia = (bolsa_ent > 0).astype(float).tocsc()
coocurrencia = (presencia.T @ presencia).toarray()    # nº de tweets donde coinciden dos palabras
frec_doc = np.diag(coocurrencia)                       # nº de tweets donde aparece cada palabra
n_docs = bolsa_ent.shape[0]
def coherencia_npmi(top):
    valores = []
    for i in range(1, len(top)):
        for j in range(i):
            p_ij = coocurrencia[top[i], top[j]] / n_docs
            if p_ij == 0:
                valores.append(-1.0)                   # nunca coinciden
            else:
                pmi = np.log(p_ij / ((frec_doc[top[i]] / n_docs) * (frec_doc[top[j]] / n_docs)))
                valores.append(pmi / -np.log(p_ij))
    return np.mean(valores)

rejilla = [5, 8, 10, 12, 15, 20]
perplejidad_val, perplejidad_ent, coherencia = [], [], []
for k in rejilla:
    t0 = time.time()
    m = LatentDirichletAllocation(n_components=k, learning_method='batch', max_iter=30, random_state=SEMILLA, n_jobs=1)
    m.fit(bolsa_ent)
    perplejidad_val.append(m.perplexity(bolsa_val))
    perplejidad_ent.append(m.perplexity(bolsa_ent))
    tops = [np.argsort(comp)[::-1][:10] for comp in m.components_]
    coherencia.append(np.mean([coherencia_npmi(t) for t in tops]))
    print(f"K={k:2d} | perplejidad validación = {perplejidad_val[-1]:.0f} | entrenamiento = {perplejidad_ent[-1]:.0f} | coherencia NPMI = {coherencia[-1]:.3f} | {time.time() - t0:.1f} s")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
ejes[0].plot(rejilla, perplejidad_val, marker='o', color=ACENTO, lw=2, label='validación (20 %)')
ejes[0].plot(rejilla, perplejidad_ent, marker='o', color=GRIS, lw=1.5, ls='--', label='entrenamiento (80 %)')
ejes[0].set_xlabel("Número de temas K"); ejes[0].set_ylabel("Perplejidad (menor = mejor)"); ejes[0].legend(frameon=False)
ejes[1].plot(rejilla, coherencia, marker='o', color=ACENTO, lw=2)
ejes[1].set_xlabel("Número de temas K"); ejes[1].set_ylabel("Coherencia NPMI (mayor = mejor)")
for ax in ejes:
    ax.set_xticks(rejilla); ax.grid(False); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/fase3_perplejidad.png", dpi=150)
plt.show()

# %% [md]
# **Qué se analiza:** se aplica el criterio de elección y se comprueba que las dos medidas coincidan. Nota técnica: con tweets muy cortos (pocas palabras útiles por documento) la perplejidad de validación de LDA tiende a SUBIR con K porque los temas extra se sobreajustan a muy poco texto; por eso se usa el menor K permitido por la rejilla si la coherencia lo respalda.

# %% [code]
# Criterio: menor perplejidad de validación, confirmada por la mayor coherencia NPMI
K_perplejidad = rejilla[int(np.argmin(perplejidad_val))]
K_coherencia = rejilla[int(np.argmax(coherencia))]
K_LDA = K_perplejidad
print(f"K por perplejidad de validación = {K_perplejidad} | K por coherencia NPMI = {K_coherencia}")
print(f"K elegido = {K_LDA} ({'ambas medidas coinciden' if K_perplejidad == K_coherencia else 'las medidas NO coinciden'})")

# %% [code]
# Ajuste final con todos los tweets de la muestra y K elegido
t0 = time.time()
lda = LatentDirichletAllocation(n_components=K_LDA, learning_method='batch', max_iter=30, random_state=SEMILLA, n_jobs=1)
doc_tema = lda.fit_transform(bolsa)
print(f"LDA final K={K_LDA} ajustado en {time.time() - t0:.1f} s")

# 10 palabras principales de cada tema
palabras_lda = [list(vocab[np.argsort(comp)[::-1][:10]]) for comp in lda.components_]
for i, p in enumerate(palabras_lda):
    print(f"Tema {i:2d}: {', '.join(p)}")

# Tema dominante de cada tweet (índice = índice de df); tweets sin palabras útiles quedan fuera
tiene_palabras = np.asarray(bolsa.sum(axis=1)).ravel() > 0
tema_dom = pd.Series(doc_tema.argmax(axis=1), index=muestra_temas.index)[tiene_palabras]
print("Tweets por tema dominante:\n", tema_dom.value_counts().sort_index().to_string())

# %% [md]
# **Qué se analiza:** qué proporción de los tweets de cada emoción tiene como tema dominante cada tema de LDA. Una fila (emoción) suma 100 %; los temas con colores más fuertes son los más típicos de esa emoción.

# %% [code]
emocion = muestra_temas.loc[tema_dom.index, 'sentiment']
cruce = pd.crosstab(emocion, tema_dom, normalize='index') * 100            # % de tweets de la emoción en cada tema
cruce_col = pd.crosstab(emocion, tema_dom, normalize='columns') * 100      # % de cada tema que es de la emoción
print(cruce.round(1).to_string())
base = emocion.value_counts(normalize=True) * 100   # peso de cada emoción en la muestra
# lift: cuántas veces más frecuente es la emoción en el tema que en la muestra completa (solo con >= 15 tweets en la celda)
conteo_cruce = pd.crosstab(emocion, tema_dom)
lift = cruce_col.div(base, axis=0).where(conteo_cruce >= 15)
print("\nPares emoción-tema más sobre-representados (lift, >= 15 tweets):")
print(lift.stack().sort_values(ascending=False).head(8).round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(cruce.values, cmap='Purples', aspect='auto')
ax.set_xticks(range(K_LDA)); ax.set_xticklabels([f"T{t}" for t in cruce.columns])
ax.set_yticks(range(len(cruce))); ax.set_yticklabels(cruce.index)
for i in range(cruce.shape[0]):
    for j in range(cruce.shape[1]):
        ax.text(j, i, f"{cruce.values[i, j]:.0f}", ha='center', va='center', fontsize=8,
                color='white' if cruce.values[i, j] > cruce.values.max() * 0.6 else TINTA)
ax.set_xlabel("Tema dominante de LDA"); ax.set_ylabel("Emoción")
ax.spines[['top', 'right']].set_visible(False)
fig.colorbar(im, ax=ax, label="% de tweets de la emoción")
plt.tight_layout()
plt.savefig("figuras/fase3_temas_vs_emociones.png", dpi=150)
plt.show()

# %% [code]
# Guardar cifras clave de la fase
resultados_fase3 = {
    "tweets_muestra": int(len(muestra_temas)), "tweets_modelados": int(tema_dom.shape[0]),
    "vocabulario": int(bolsa.shape[1]), "tweets_sin_palabras": n_vacios,
    "rejilla_K": rejilla, "perplejidad_validacion": [float(x) for x in perplejidad_val],
    "perplejidad_entrenamiento": [float(x) for x in perplejidad_ent],
    "coherencia_npmi": [float(x) for x in coherencia], "K_perplejidad": int(K_perplejidad), "K_coherencia": int(K_coherencia),
    "K_LDA": int(K_LDA), "palabras_lda": palabras_lda,
    "tweets_por_tema": {int(k): int(v) for k, v in tema_dom.value_counts().sort_index().items()},
    "cruce_pct_por_emocion": {e: {int(t): float(round(v, 2)) for t, v in fila.items()} for e, fila in cruce.iterrows()},
    "lift_top": {f"{e}|T{t}": float(v) for (e, t), v in lift.stack().sort_values(ascending=False).head(8).items()},
    "tiempo_s": round(time.time() - t_ini_fase3, 1),
}
with open("results/fase3.json", "w") as f:
    json.dump(resultados_fase3, f, ensure_ascii=False, indent=2)
print("Guardado results/fase3.json")

# %% [conclusion]
# **Conclusión de negocio (Fase 3).** LDA sobre `muestra_temas` (5 000 tweets; 4 748 con palabras útiles tras quitar stop words y tokens basura; vocabulario de 1 038 palabras).
# - Número de temas: K = 5. Criterio: menor perplejidad en el 20 % de validación, confirmada por la coherencia NPMI. La perplejidad de validación sube con K (2 843 con K = 5; 10 703 con K = 20) y la coherencia NPMI baja en general (de -0,193 con K = 5 a -0,401 con K = 20; el mínimo fue -0,427 con K = 15), porque los tweets tienen muy pocas palabras útiles y los temas extra se sobreajustan. Ningún K mayor mejora estas medidas: es una limitación del texto corto, y por eso K = 5 es un tema-resumen amplio, no una taxonomía fina.
# - Tres temas interpretables: Tema 4 (`day, happy, make, fun, mothers, mother, mom`) = celebraciones y Día de la Madre; Tema 3 (`work, miss, need, week, wish, feel, feeling`) = trabajo, añoranza y sentimientos; Tema 0 (`home, today, night, working, going, house, wait`) = rutina diaria en casa y por la noche. Los temas 1 y 2 mezclan cortesías y expresiones sueltas (`good, thanks, nice` / `love, really, sad, hey`) y son menos claros.
# - Cruce con emociones: el Tema 4 concentra 31,4 % de los tweets de `love` (1,68 veces lo esperado), 25,2 % de `happiness` y 24,5 % de `fun`, frente a solo 10,2 % de `hate`; el Tema 3 reúne 24,4 % de `sadness` y de `enthusiasm` y 23,9 % de `hate`. Casi todas las demás celdas quedan entre 14 % y 27 %, cerca del 20 % de un reparto al azar; salen de ese rango `empty` en el Tema 1 (33,0 %) y en el Tema 4 (9,9 %), `happiness` en el Tema 3 (13,6 %) y `love` en el Tema 3 (9,4 %).
# - Lectura honesta: los temas describen de qué hablan los tweets (celebraciones, trabajo, casa), no cómo se sienten. Separan emociones positivas de afecto (love, happiness, fun) de las negativas, pero no distinguen emociones vecinas. Valen como contexto de negocio (p. ej. picos de positividad en fechas como el Día de la Madre), no como clasificador. `anger` (10 tweets) y `boredom` (26) tienen muy pocos tweets en la muestra y sus porcentajes no son fiables.
