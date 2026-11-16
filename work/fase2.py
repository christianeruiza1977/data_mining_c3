# %% [que]
# Se analiza la estructura lingüística de una muestra estratificada de tweets con spaCy (`en_core_web_sm`): tokenización, lematización, etiquetado gramatical (POS), reconocimiento de entidades (NER) y árbol de dependencias. Se compara la proporción de adjetivos, verbos e interjecciones entre emociones y se mide con un clasificador simple si estas características ayudan a clasificar.

# %% [code]
import time, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from scipy.stats import kruskal
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
t_ini_fase2 = time.time()

# Muestra estratificada: hasta 200 tweets por emoción (anger y boredom tienen menos de 200: se toman todos)
partes = [g.sample(min(200, len(g)), random_state=SEMILLA) for _, g in df.groupby('sentiment')]
muestra = pd.concat(partes).reset_index(drop=True)
print("Muestra:", muestra.shape)
print(muestra['sentiment'].value_counts().to_string())

# Pipeline sin el parser (más rápido): tokenizador + tagger + lematizador + NER
nlp = spacy.load('en_core_web_sm', disable=['parser'])
t0 = time.time()
docs = list(nlp.pipe(muestra['texto_limpio'], batch_size=500))
print(f"spaCy procesó {len(docs)} tweets en {time.time() - t0:.1f} s")

# %% [md]
# **Qué se analiza:** cómo spaCy tokeniza y lematiza el texto informal de los tweets y qué etiqueta gramatical (POS) asigna a cada palabra.

# %% [code]
# Ejemplos de tokenización, lematización y POS (3 tweets de la muestra)
for i in [0, 700, 1500]:
    doc = docs[i]
    print(f"[{muestra.loc[i, 'sentiment']}] {doc.text}")
    print(pd.DataFrame({'token': [t.text for t in doc], 'lema': [t.lemma_ for t in doc],
                        'POS': [t.pos_ for t in doc]}).T.to_string(header=False))
    print()
# Tokens cuyo lema difiere del texto (la lematización agrupa formas de una misma palabra)
cambios = Counter((t.text.lower(), t.lemma_.lower()) for d in docs for t in d
                  if t.is_alpha and t.text.lower() != t.lemma_.lower())
print("Pares palabra -> lema más frecuentes:", cambios.most_common(12))

# %% [md]
# **Qué se analiza:** proporción de adjetivos (ADJ), verbos (VERB) e interjecciones (INTJ) sobre el total de tokens de cada tweet, promediada por emoción. Si una emoción usa más un tipo de palabra, esa diferencia podría servir al clasificador.

# %% [code]
# Proporción de cada categoría gramatical por tweet (sobre tokens que no son espacios)
categorias = ['ADJ', 'VERB', 'INTJ', 'NOUN', 'ADV', 'PRON', 'PROPN', 'ADP', 'AUX', 'DET', 'NUM', 'PUNCT', 'SYM', 'X']
filas = []
for doc in docs:
    toks = [t.pos_ for t in doc if not t.is_space]
    n = max(len(toks), 1)
    conteo_pos = Counter(toks)
    filas.append({c: conteo_pos.get(c, 0) / n for c in categorias})
prop_pos = pd.DataFrame(filas)
prop_pos['sentiment'] = muestra['sentiment'].values
por_emocion = prop_pos.groupby('sentiment')[['ADJ', 'VERB', 'INTJ']].mean() * 100   # en %
por_emocion = por_emocion.round(2)
print(por_emocion.sort_values('INTJ', ascending=False).to_string())

# Magnitud de las diferencias: rango entre emociones y prueba de Kruskal-Wallis (¿difiere alguna emoción?)
tabla_dif = []
for c in ['ADJ', 'VERB', 'INTJ']:
    grupos = [g[c].values for _, g in prop_pos.groupby('sentiment')]
    H, p = kruskal(*grupos)
    eta2 = max((H - len(grupos) + 1) / (len(prop_pos) - len(grupos)), 0)   # tamaño de efecto (eta² de Kruskal)
    tabla_dif.append({'categoria': c, 'media_global_%': prop_pos[c].mean() * 100,
                      'max_emocion': por_emocion[c].idxmax(), 'max_%': por_emocion[c].max(),
                      'min_emocion': por_emocion[c].idxmin(), 'min_%': por_emocion[c].min(),
                      'dif_puntos_pp': por_emocion[c].max() - por_emocion[c].min(), 'p_kruskal': p, 'eta2': eta2})
tabla_dif = pd.DataFrame(tabla_dif).round(4)
print(tabla_dif.to_string(index=False))

TINTA, ACENTO, GRIS = "#14323F", "#6E5BD6", "#B7C3C9"
fig, ejes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
for ax, c in zip(ejes, ['ADJ', 'VERB', 'INTJ']):
    serie = por_emocion[c].sort_values()
    colores = [ACENTO if e == serie.idxmax() else GRIS for e in serie.index]   # ACENTO: la emoción con mayor proporción
    ax.barh(serie.index, serie.values, color=colores)
    ax.axvline(prop_pos[c].mean() * 100, color=TINTA, lw=1, ls='--')   # media global
    ax.set_title(f"{c} (% de tokens)", color=TINTA)
    ax.grid(False); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/fase2_pos_por_emocion.png", dpi=150)
plt.show()

# %% [md]
# **Qué se analiza:** qué entidades nombradas (personas, organizaciones, lugares, fechas) aparecen más en la muestra y qué tan frecuentes son por emoción.

# %% [code]
# Entidades nombradas (NER)
entidades = [(e.text, e.label_, s) for doc, s in zip(docs, muestra['sentiment']) for e in doc.ents]
tabla_ent = pd.DataFrame(entidades, columns=['entidad', 'tipo', 'emocion'])
print("Entidades totales:", len(tabla_ent), "| tweets con al menos una:", sum(len(d.ents) > 0 for d in docs), "de", len(docs))
top10 = tabla_ent['entidad'].str.lower().value_counts().head(10)
print("\nTop 10 entidades:\n", top10.to_string())
print("\nTipos de entidad:\n", tabla_ent['tipo'].value_counts().head(8).to_string())
for tipo in ['PERSON', 'ORG', 'GPE', 'DATE']:
    top_tipo = tabla_ent.loc[tabla_ent['tipo'] == tipo, 'entidad'].str.lower().value_counts().head(5)
    print(f"\nTop {tipo}:", top_tipo.to_dict())
# Densidad de entidades por emoción: % de tweets con al menos una entidad
con_ent = pd.Series([len(d.ents) > 0 for d in docs]).groupby(muestra['sentiment'].values).mean() * 100
print("\n% de tweets con entidad por emoción:\n", con_ent.sort_values(ascending=False).round(1).to_string())

# %% [md]
# **Qué se analiza:** el árbol de dependencias de 3 tweets (qué palabra depende de cuál). Para esto se usa el modelo completo de spaCy, con parser.

# %% [code]
# Dependencias: modelo completo (con parser) solo en 3 tweets cortos de emociones distintas
nlp_dep = spacy.load('en_core_web_sm')
from spacy import displacy
elegidos = []
for emo in ['happiness', 'hate', 'worry']:
    cand = muestra[(muestra['sentiment'] == emo) & (muestra['texto_limpio'].str.split().str.len().between(6, 11))]
    elegidos.append(cand.index[0])
for k, i in enumerate(elegidos):
    doc = nlp_dep(muestra.loc[i, 'texto_limpio'])
    print(f"\n[{muestra.loc[i, 'sentiment']}] {doc.text}")
    print(pd.DataFrame({'token': [t.text for t in doc], 'dep': [t.dep_ for t in doc],
                        'cabeza': [t.head.text for t in doc]}).T.to_string(header=False))
    opciones = {'compact': True, 'distance': 90}
    svg = displacy.render(doc, style='dep', jupyter=False, options=opciones)
    with open(f"figuras/fase2_dependencias_{k + 1}.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    displacy.render(doc, style='dep', jupyter=True, options=opciones)   # se dibuja en el notebook

# %% [md]
# **Qué se analiza:** las etiquetas de dependencia sobre toda la muestra (parser completo): negaciones (`neg`), sujetos, interjecciones y modificadores. Se comparan entre emociones igual que el POS.

# %% [code]
# Parser completo sobre toda la muestra (solo para contar relaciones de dependencia)
t0 = time.time()
docs_dep = list(nlp_dep.pipe(muestra['texto_limpio'], batch_size=500))
print(f"Parser completo: {time.time() - t0:.1f} s")
deps_interes = ['neg', 'nsubj', 'dobj', 'amod', 'advmod', 'intj', 'aux', 'prep', 'ROOT']
filas = []
for doc in docs_dep:
    n = max(sum(1 for t in doc if not t.is_space), 1)
    c = Counter(t.dep_ for t in doc)
    filas.append({d: c.get(d, 0) / n for d in deps_interes})
prop_dep = pd.DataFrame(filas)
prop_dep['sentiment'] = muestra['sentiment'].values
dep_emocion = (prop_dep.groupby('sentiment')[['neg', 'intj', 'amod', 'advmod']].mean() * 100).round(2)
print(dep_emocion.sort_values('neg', ascending=False).to_string())
# ¿Difiere la negación entre emociones? (misma prueba y tamaño de efecto que con POS)
grupos_neg = [g['neg'].values for _, g in prop_dep.groupby('sentiment')]
H_neg, p_neg = kruskal(*grupos_neg)
eta2_neg = max((H_neg - len(grupos_neg) + 1) / (len(prop_dep) - len(grupos_neg)), 0)
print(f"Negación (neg): Kruskal-Wallis p = {p_neg:.2g}, eta2 = {eta2_neg:.3f}")

# %% [md]
# **Qué se analiza:** si estas características gramaticales sirven para clasificar. Se entrena una regresión logística SOLO con proporciones POS (y con POS + dependencias + entidades) con validación cruzada de 5 particiones y se compara con el azar y con la clase mayoritaria (macro-F1). Como referencia se añade una bolsa de palabras TF-IDF en la misma muestra.

# %% [code]
y_m = muestra['sentiment'].to_numpy(dtype=str)
X_pos = prop_pos[categorias].values
X_pos_dep = np.hstack([X_pos, prop_dep[deps_interes].values,
                       np.array([[len(d.ents) > 0, len(d.ents)] for d in docs])])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

def f1_cv(modelo, X):
    # macro-F1 medio en 5 particiones estratificadas
    return cross_val_score(modelo, X, y_m, cv=cv, scoring='f1_macro')

lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEMILLA))
# Azar: macro-F1 de predicciones aleatorias uniformes sobre las 13 clases (50 repeticiones)
from sklearn.metrics import f1_score
rng = np.random.default_rng(SEMILLA)
clases_m = np.unique(y_m)
f1_azar = np.array([f1_score(y_m, rng.choice(clases_m, size=len(y_m)), average='macro') for _ in range(50)])
res_clf = {
    'mayoritaria': f1_cv(DummyClassifier(strategy='most_frequent'), X_pos),
    'azar (uniforme, 50 repeticiones)': f1_azar,
    'LogReg solo proporciones POS': f1_cv(lr(), X_pos),
    'LogReg POS + dependencias + entidades': f1_cv(lr(), X_pos_dep),
    'LogReg bolsa de palabras TF-IDF (referencia)': f1_cv(make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)),
                                      LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEMILLA)), muestra['texto_limpio']),
}
tabla_clf = pd.DataFrame({k: {'f1_macro': v.mean(), 'desv': v.std()} for k, v in res_clf.items()}).T.round(3)
print(tabla_clf.to_string())

fig, ax = plt.subplots(figsize=(8, 3.6))
nombres = list(tabla_clf.index)[::-1]
vals = tabla_clf.loc[nombres, 'f1_macro']
ax.barh(nombres, vals, xerr=tabla_clf.loc[nombres, 'desv'], color=[ACENTO if n == 'LogReg solo proporciones POS' else GRIS for n in nombres])
ax.set_xlabel("F1 macro (validación cruzada, 5 particiones)")
ax.grid(False); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/fase2_clasificacion_pos.png", dpi=150)
plt.show()

# %% [code]
# Guardar cifras clave de la fase
resultados_fase2 = {
    "tamano_muestra": int(len(muestra)),
    "por_emocion_ADJ_VERB_INTJ_pct": por_emocion.to_dict(orient='index'),
    "diferencias": tabla_dif.to_dict(orient='records'),
    "entidades_totales": int(len(tabla_ent)),
    "tweets_con_entidad_pct": float(np.mean([len(d.ents) > 0 for d in docs]) * 100),
    "top10_entidades": top10.to_dict(),
    "tipos_entidad": tabla_ent['tipo'].value_counts().head(8).to_dict(),
    "dep_por_emocion_pct": dep_emocion.to_dict(orient='index'),
    "neg_kruskal_p": float(p_neg), "neg_eta2": float(eta2_neg),
    "f1_macro_cv": {k: {"media": float(v.mean()), "desv": float(v.std())} for k, v in res_clf.items()},
    "tiempo_s": round(time.time() - t_ini_fase2, 1),
}
with open("results/fase2.json", "w") as f:
    json.dump(resultados_fase2, f, ensure_ascii=False, indent=2, default=float)
print("Guardado results/fase2.json")

# %% [conclusion]
# **Conclusión de negocio (Fase 2).** Muestra estratificada de 2 489 tweets (200 por emoción; `boredom` 179 y `anger` 110, todos los disponibles).
# - Las diferencias gramaticales existen pero son pequeñas. Adjetivos: de 4,59 % (`neutral`) a 7,16 % (`surprise`), media 6,13 %. Verbos: de 11,37 % (`love`) a 14,40 % (`hate`), media 12,81 %. Ambas son estadísticamente detectables (Kruskal-Wallis p = 0,005 y 0,003) pero con tamaño de efecto minúsculo (eta² cercano a 0,007, es decir, la emoción explica menos del 1 % de la variación entre tweets). Las interjecciones (media 1,55 %; `empty` 2,80 % vs `anger` 0,83 %) NO difieren de forma significativa (p = 0,66).
# - Las dependencias dan la señal gramatical más nítida (negación: Kruskal-Wallis p = 1,8e-11, eta² = 0,026, aun así pequeño): la negación (`neg`) pesa 2,39 % de los tokens en `worry` y 2,25 % en `hate` frente a 0,62 % en `happiness` y 0,75 % en `love`; los modificadores adjetivales (`amod`) son más frecuentes en `happiness` (4,34 %) y `love` (4,22 %) que en `neutral` (2,85 %). Es decir, las emociones positivas describen con adjetivos y las negativas niegan o se preocupan por acciones.
# - NER aporta poco en tweets informales: 1 926 entidades en 1 228 de 2 489 tweets (49,3 %), dominadas por DATE (447: today, tonight, tomorrow), PERSON (423) y ORG (333), con errores evidentes ('awww' como PERSON, 'lol' como ORG). Las entidades más frecuentes (today 98, tonight 48, tomorrow 33) son marcas temporales, no emociones. Los tweets con entidad van de 60,0 % (`happiness`) a 39,0 % (`worry`).
# - ¿Aporta para clasificar? Muy poco. Con validación cruzada (5 particiones), regresión logística con solo proporciones POS logra F1 macro 0,095 (+- 0,011), frente a 0,076 del azar uniforme y 0,011 de la clase mayoritaria; añadir dependencias y entidades sube solo a 0,098. Una bolsa de palabras TF-IDF en la misma muestra llega a 0,184, el doble. Decisión: las características gramaticales se pueden usar como complemento (la negación es la más informativa), pero el contenido léxico es la señal principal para clasificar emociones.
