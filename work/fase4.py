# %% [que]
# Se entrenan dos clasificadores clásicos de emociones (13 clases) sobre el texto limpio: TF-IDF + Naive Bayes multinomial y TF-IDF + SVM lineal. Se ajustan pocos hiperparámetros usando SOLO la validación, se evalúa en prueba una única vez y se miran F1 macro, matrices de confusión y las emociones que más se confunden.

# %% [md]
# **Partición de los datos.** Se usa una partición estratificada 70/15/15 con semilla fija. Estratificar significa que cada subconjunto conserva la proporción de cada emoción, algo clave porque hay clases muy pequeñas (anger, boredom).

# %% [code]
import time, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# Partición estratificada 70/15/15 con SEMILLA (mismo código que common.py)
idx_ent, idx_tmp = train_test_split(df.index, test_size=0.30, stratify=df['sentiment'], random_state=SEMILLA)
idx_val, idx_pru = train_test_split(idx_tmp, test_size=0.50, stratify=df.loc[idx_tmp, 'sentiment'], random_state=SEMILLA)
X_train, y_train = df.loc[idx_ent, 'texto_limpio'], df.loc[idx_ent, 'sentiment']
X_val,   y_val   = df.loc[idx_val, 'texto_limpio'], df.loc[idx_val, 'sentiment']
X_test,  y_test  = df.loc[idx_pru, 'texto_limpio'], df.loc[idx_pru, 'sentiment']

# Tamaños y proporción por clase en cada subconjunto (deben ser casi idénticas)
print(f"Tamaños -> train: {len(X_train)}, validación: {len(X_val)}, prueba: {len(X_test)}")
proporciones = pd.DataFrame({
    'total':      df['sentiment'].value_counts(normalize=True),
    'train':      y_train.value_counts(normalize=True),
    'validacion': y_val.value_counts(normalize=True),
    'prueba':     y_test.value_counts(normalize=True)}).round(4) * 100
proporciones['n_total'] = df['sentiment'].value_counts()
print("Proporción (%) de cada emoción por subconjunto:")
print(proporciones.to_string())
print("Diferencia máxima entre train y total (puntos porcentuales):",
      round((proporciones['train'] - proporciones['total']).abs().max(), 3))

# %% [md]
# **Línea base.** Antes de mirar los modelos, cuánto se obtiene sin aprender nada: elegir al azar entre 13 clases (1/13) o predecir siempre la clase mayoritaria.

# %% [code]
azar = 1 / len(ETIQUETAS)
clase_mayoritaria = y_train.value_counts().idxmax()
pred_mayoritaria = np.full(len(y_test), clase_mayoritaria)
base_acc = accuracy_score(y_test, pred_mayoritaria)
base_f1 = f1_score(y_test, pred_mayoritaria, average='macro', labels=ETIQUETAS, zero_division=0)
print(f"Azar uniforme: accuracy = F1 macro esperado ≈ {azar:.3f}")
print(f"Siempre '{clase_mayoritaria}': accuracy = {base_acc:.3f}, F1 macro = {base_f1:.3f}")

# %% [md]
# **Ajuste de hiperparámetros sobre validación.** Se prueba una rejilla pequeña (n-gramas, alpha de NB, C y class_weight de SVM). El criterio es el F1 macro en validación; la prueba no se toca aquí.

# %% [code]
def crear_nb(alpha, ngramas):
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=ngramas, min_df=2, sublinear_tf=True)),
                     ('clf', MultinomialNB(alpha=alpha))])

def crear_svm(C, ngramas, pesos):
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=ngramas, min_df=2, sublinear_tf=True)),
                     ('clf', LinearSVC(C=C, class_weight=pesos, random_state=SEMILLA))])

filas = []
for ngr in [(1, 1), (1, 2)]:
    for alpha in [0.03, 0.1, 0.3, 1.0]:
        m = crear_nb(alpha, ngr).fit(X_train, y_train)
        filas.append({'modelo': 'NB', 'ngramas': ngr, 'alpha': alpha, 'C': None, 'class_weight': None,
                      'f1_val': f1_score(y_val, m.predict(X_val), average='macro')})
    for C in [0.03, 0.1, 0.3, 1.0]:
        for pesos in [None, 'balanced']:
            m = crear_svm(C, ngr, pesos).fit(X_train, y_train)
            filas.append({'modelo': 'SVM', 'ngramas': ngr, 'alpha': None, 'C': C, 'class_weight': pesos,
                          'f1_val': f1_score(y_val, m.predict(X_val), average='macro')})
rejilla = pd.DataFrame(filas)
print(rejilla.round(4).to_string(index=False))

mejor_nb  = rejilla[rejilla.modelo == 'NB'].sort_values('f1_val').iloc[-1]
mejor_svm = rejilla[rejilla.modelo == 'SVM'].sort_values('f1_val').iloc[-1]
print("\nMejor NB : ngramas", mejor_nb["ngramas"], "alpha", mejor_nb["alpha"], "F1 val", round(mejor_nb["f1_val"], 4))
print("Mejor SVM: ngramas", mejor_svm["ngramas"], "C", mejor_svm["C"], "class_weight", mejor_svm["class_weight"], "F1 val", round(mejor_svm["f1_val"], 4))

# %% [md]
# **Evaluación final en prueba (una sola vez).** Con los hiperparámetros elegidos se reentrena cada modelo solo con train, se mide el tiempo del ajuste y se evalúa en el conjunto de prueba.

# %% [code]
# Hiperparámetros elegidos (los reutiliza la Fase 6)
params_nb  = {'alpha': float(mejor_nb['alpha']), 'ngramas': mejor_nb['ngramas']}
params_svm = {'C': float(mejor_svm['C']), 'ngramas': mejor_svm['ngramas'], 'pesos': mejor_svm['class_weight']}

modelos_finales = {'NB':  crear_nb(params_nb['alpha'], params_nb['ngramas']),
                   'SVM': crear_svm(params_svm['C'], params_svm['ngramas'], params_svm['pesos'])}
predicciones, filas_res, tiempos = {}, [], {}
for nombre, modelo in modelos_finales.items():
    t0 = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempos[nombre] = time.perf_counter() - t0
    predicciones[nombre] = modelo.predict(X_test)            # única evaluación en prueba
    filas_res.append({'modelo': nombre, 'fase': 'Fase 4',
                      'f1_macro': f1_score(y_test, predicciones[nombre], average='macro'),
                      'accuracy': accuracy_score(y_test, predicciones[nombre]),
                      'tiempo_s': tiempos[nombre]})
resultados = pd.DataFrame(filas_res)
print(resultados.round(4).to_string(index=False))

# F1 por clase en prueba, para ambos modelos
f1_por_clase = pd.DataFrame({
    n: pd.Series(f1_score(y_test, predicciones[n], labels=ETIQUETAS, average=None), index=ETIQUETAS)
    for n in modelos_finales})
f1_por_clase['n_prueba'] = y_test.value_counts()
f1_por_clase = f1_por_clase.sort_values('n_prueba', ascending=False)
print("\nF1 por clase (prueba):")
print(f1_por_clase.round(3).to_string())

# Mejor modelo por F1 macro de prueba (ambos ya estaban fijados con validación)
mejor_nombre = resultados.sort_values('f1_macro').iloc[-1]['modelo']
mejor_modelo = modelos_finales[mejor_nombre]
pred_test_mejor = predicciones[mejor_nombre]
print("\nMejor modelo:", mejor_nombre)

# %% [md]
# **Qué se analiza:** la matriz de confusión 13×13 de cada modelo, normalizada por fila (cada fila suma 100 %: de los tweets de una emoción real, qué porcentaje se predijo como cada emoción). Las emociones van ordenadas de más a menos frecuentes; la diagonal es el recall de cada clase y lo que queda fuera de ella son las confusiones.

# %% [code]
TINTA, ACENTO, GRIS = "#14323F", "#6E5BD6", "#B7C3C9"
orden = list(y_train.value_counts().index)        # emociones de más a menos frecuentes

def dibujar_matriz(pred, titulo, archivo):
    cm = confusion_matrix(y_test, pred, labels=orden, normalize='true') * 100
    fig, ax = plt.subplots(figsize=(9.5, 8))
    from matplotlib.colors import LinearSegmentedColormap
    mapa = LinearSegmentedColormap.from_list('tinta', ['#FFFFFF', ACENTO])
    ax.imshow(cm, cmap=mapa, vmin=0, vmax=60)
    for i in range(len(orden)):
        for j in range(len(orden)):
            if cm[i, j] >= 1:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > 35 else TINTA)
    ax.set_xticks(range(len(orden))); ax.set_xticklabels(orden, rotation=45, ha='right', color=TINTA)
    ax.set_yticks(range(len(orden))); ax.set_yticklabels(orden, color=TINTA)
    ax.set_xlabel('Emoción predicha', color=TINTA); ax.set_ylabel('Emoción real', color=TINTA)
    ax.set_title(titulo, color=TINTA, loc='left')
    for lado in ['top', 'right', 'left', 'bottom']:
        ax.spines[lado].set_visible(False)
    plt.tight_layout(); plt.savefig(archivo, dpi=150); plt.show()

dibujar_matriz(predicciones['NB'], 'Naive Bayes: matriz de confusión (% por fila, prueba)', 'figuras/fase4_matriz_nb.png')
dibujar_matriz(predicciones['SVM'], 'SVM lineal: matriz de confusión (% por fila, prueba)', 'figuras/fase4_matriz_svm.png')

# %% [md]
# **Qué se analiza:** los 5 pares (real → predicha) con más confusión fuera de la diagonal, medidos como porcentaje de la clase real, para ver qué emociones se parecen entre sí desde el punto de vista del texto.

# %% [code]
def top_confusiones(pred, n=5):
    cm = confusion_matrix(y_test, pred, labels=ETIQUETAS, normalize='true') * 100
    cuentas = confusion_matrix(y_test, pred, labels=ETIQUETAS)
    pares = [(ETIQUETAS[i], ETIQUETAS[j], cm[i, j], cuentas[i, j])
             for i in range(len(ETIQUETAS)) for j in range(len(ETIQUETAS)) if i != j]
    pares.sort(key=lambda p: p[3], reverse=True)       # por número de tweets mal clasificados
    return pd.DataFrame(pares[:n], columns=['real', 'predicha', 'pct_de_la_real', 'tweets'])

confusiones = {n: top_confusiones(predicciones[n]) for n in modelos_finales}
for n, tabla in confusiones.items():
    print(f"Top 5 confusiones de {n} (por número de tweets):")
    print(tabla.round(1).to_string(index=False), "\n")

# %% [code]
# Figura de apoyo: F1 por clase de ambos modelos (ACENTO para el mejor modelo)
fig, ax = plt.subplots(figsize=(9, 4.8))
posicion = np.arange(len(orden)); ancho = 0.4
tabla_plot = f1_por_clase.loc[orden]
otro = 'NB' if mejor_nombre == 'SVM' else 'SVM'
ax.bar(posicion - ancho/2, tabla_plot[otro], ancho, color=GRIS, label=otro)
ax.bar(posicion + ancho/2, tabla_plot[mejor_nombre], ancho, color=ACENTO, label=mejor_nombre)
ax.axhline(azar, color=TINTA, ls='--', lw=1); ax.text(len(orden)-0.5, azar+0.01, 'azar 1/13', ha='right', color=TINTA, fontsize=8)
ax.set_xticks(posicion); ax.set_xticklabels(orden, rotation=45, ha='right', color=TINTA)
ax.set_ylabel('F1 por clase (prueba)', color=TINTA); ax.legend(frameon=False)
for lado in ['top', 'right']:
    ax.spines[lado].set_visible(False)
plt.tight_layout(); plt.savefig('figuras/fase4_f1_por_clase.png', dpi=150); plt.show()

# %% [code]
# Guardar cifras clave
nb, sv = resultados.set_index('modelo').loc['NB'], resultados.set_index('modelo').loc['SVM']
salida = {
    'tamanos': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    'azar': azar, 'clase_mayoritaria': clase_mayoritaria,
    'base_mayoritaria_accuracy': base_acc, 'base_mayoritaria_f1_macro': base_f1,
    'hiperparametros': {'NB': {'alpha': params_nb['alpha'], 'ngramas': list(params_nb['ngramas']), 'f1_val': float(mejor_nb['f1_val'])},
                        'SVM': {'C': params_svm['C'], 'ngramas': list(params_svm['ngramas']),
                                'class_weight': params_svm['pesos'], 'f1_val': float(mejor_svm['f1_val'])}},
    'resultados': resultados.round(5).to_dict('records'),
    'mejor_nombre': mejor_nombre,
    'f1_por_clase': f1_por_clase.round(4).to_dict('index'),
    'top_confusiones': {n: t.round(2).to_dict('records') for n, t in confusiones.items()}}
with open('results/fase4.json', 'w') as f:
    json.dump(salida, f, indent=2, default=str)

# %% [conclusion]
# CONCLUSION_PLACEHOLDER
