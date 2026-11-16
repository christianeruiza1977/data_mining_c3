# %% [que]
# Se repite la comparación NB, SVM y RNN con 3, 5, 7, 9 y 11 emociones sorteadas al azar (3 semillas: 42, 43 y 44, promediando) para ver cómo cae el F1 macro al crecer el número de clases, frente a la línea de azar 1/k, y se recomienda una granularidad de etiquetas.

# %% [md]
# **Diseño del experimento.** Para cada semilla se sortea una permutación de las 13 emociones con `np.random.RandomState(semilla)` y se toman las primeras k (así, dentro de una semilla, los subconjuntos son anidados: k=5 contiene a k=3). Se filtra la misma partición train/val/test de la Fase 4. NB y SVM usan los hiperparámetros elegidos en la Fase 4; la RNN usa una versión más pequeña y rápida (embedding 32, GRU bidireccional de 32, máximo 4 épocas, paciencia 1) para que el experimento completo sea razonable en 4 CPU. Aquí la prueba es la evaluación final de cada combinación (una vez).

# %% [code]
import time, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

KS = [3, 5, 7, 9, 11]
SEMILLAS = [42, 43, 44]
conteo_total = df['sentiment'].value_counts()

def sortear_clases(semilla, k):
    rng = np.random.RandomState(semilla)
    return list(rng.permutation(ETIQUETAS)[:k])

def filtrar(X, y, clases):
    mascara = y.isin(clases)
    return X[mascara], y[mascara]

def evaluar(pred, y_real, clases):
    return (f1_score(y_real, pred, average='macro', labels=clases, zero_division=0),
            accuracy_score(y_real, pred))

registros = []
t_inicio = time.perf_counter()
for semilla in SEMILLAS:
    for k in KS:
        clases = sortear_clases(semilla, k)
        Xtr, ytr = filtrar(X_train, y_train, clases)
        Xva, yva = filtrar(X_val, y_val, clases)
        Xte, yte = filtrar(X_test, y_test, clases)
        info = {'k': k, 'semilla': semilla, 'clases': ', '.join(sorted(clases)),
                'tamanos': ', '.join(f"{c}={int((yte == c).sum())}" for c in sorted(clases)),
                'n_test': len(yte), 'pct_mayoritaria': (yte.value_counts().iloc[0] / len(yte)) * 100,
                'azar': 1 / k}

        # NB y SVM con los hiperparámetros de la Fase 4
        for nombre, pipe in [('NB', crear_nb(params_nb['alpha'], params_nb['ngramas'])),
                             ('SVM', crear_svm(params_svm['C'], params_svm['ngramas'], params_svm['pesos']))]:
            t0 = time.perf_counter()
            pipe.fit(Xtr, ytr)
            tiempo = time.perf_counter() - t0
            f1, acc = evaluar(pipe.predict(Xte), yte, clases)
            registros.append({**info, 'modelo': nombre, 'f1_macro': f1, 'accuracy': acc, 'tiempo_s': tiempo})

        # RNN pequeña (función de la Fase 5)
        r = entrenar_rnn(Xtr, ytr, Xva, yva, Xte, yte, sorted(clases), max_tokens=15000, dim_emb=32,
                         unidades=32, max_epocas=4, paciencia=1)
        registros.append({**info, 'modelo': 'RNN', 'f1_macro': r['f1_macro'], 'accuracy': r['accuracy'],
                          'tiempo_s': r['tiempo_s']})
        print(f"semilla {semilla}, k={k:2d}: " + ' | '.join(
            f"{x['modelo']} F1={x['f1_macro']:.3f}" for x in registros[-3:]) + f"  [{time.perf_counter()-t_inicio:.0f} s]")

tabla_larga = pd.DataFrame(registros)
tiempo_total_f6 = time.perf_counter() - t_inicio
print(f"\nTiempo total del experimento: {tiempo_total_f6/60:.1f} min")

# %% [md]
# **Resumen por k y modelo.** Promedio y desviación estándar sobre las 3 semillas de F1 macro y accuracy.

# %% [code]
resumen = (tabla_larga.groupby(['k', 'modelo'])[['f1_macro', 'accuracy']].agg(['mean', 'std']).round(4))
resumen.columns = ['f1_media', 'f1_desv', 'acc_media', 'acc_desv']
resumen = resumen.reset_index()
resumen['azar'] = (1 / resumen['k']).round(4)
resumen['ganancia_vs_azar'] = (resumen['f1_media'] / resumen['azar']).round(2)   # veces sobre el azar
print(resumen.to_string(index=False))

# %% [md]
# **Qué se analiza:** F1 macro (promedio de 3 semillas, con banda de ±1 desviación estándar) en función del número de clases para NB, SVM y RNN, junto a la línea de azar 1/k. Muestra cuánto se pierde al aumentar la granularidad y cuánto se gana sobre adivinar.

# %% [code]
TINTA, ACENTO, GRIS = "#14323F", "#6E5BD6", "#B7C3C9"
colores = {'NB': GRIS, 'SVM': ACENTO, 'RNN': TINTA}
estilos = {'NB': '-', 'SVM': '-', 'RNN': '--'}
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for nombre in ['NB', 'SVM', 'RNN']:
    s = resumen[resumen.modelo == nombre]
    ax.plot(s['k'], s['f1_media'], marker='o', color=colores[nombre], ls=estilos[nombre], label=nombre, lw=2)
    ax.fill_between(s['k'], s['f1_media'] - s['f1_desv'], s['f1_media'] + s['f1_desv'],
                    color=colores[nombre], alpha=0.15, lw=0)
ks_fino = np.linspace(3, 11, 50)
ax.plot(ks_fino, 1 / ks_fino, color=TINTA, ls=':', lw=1.2, label='azar (1/k)')
ax.set_xticks(KS); ax.set_xlabel('Número de emociones (k)', color=TINTA)
ax.set_ylabel('F1 macro en prueba', color=TINTA); ax.set_ylim(0, None)
ax.legend(frameon=False)
for lado in ['top', 'right']:
    ax.spines[lado].set_visible(False)
plt.tight_layout(); plt.savefig('figuras/fase6_f1_vs_k.png', dpi=150); plt.show()

# %% [md]
# **Qué se analiza:** accuracy frente a F1 macro para cada combinación (k, semilla, modelo). Cuando la accuracy es alta pero el F1 macro es bajo, el modelo acierta sobre todo las clases grandes e ignora las pequeñas; se señalan los casos con mayor brecha.

# %% [code]
tabla_larga['brecha'] = tabla_larga['accuracy'] - tabla_larga['f1_macro']
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot([0, 1], [0, 1], color=GRIS, lw=1)
for nombre in ['NB', 'SVM', 'RNN']:
    s = tabla_larga[tabla_larga.modelo == nombre]
    ax.scatter(s['accuracy'], s['f1_macro'], color=colores[nombre], label=nombre, s=40, alpha=0.9)
peores = tabla_larga.sort_values('brecha', ascending=False).head(3)
for _, fila in peores.iterrows():
    ax.annotate(f"k={fila['k']}, s={fila['semilla']}", (fila['accuracy'], fila['f1_macro']),
                textcoords='offset points', xytext=(6, -10), fontsize=8, color=TINTA)
ax.set_xlabel('Accuracy', color=TINTA); ax.set_ylabel('F1 macro', color=TINTA)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False)
for lado in ['top', 'right']:
    ax.spines[lado].set_visible(False)
plt.tight_layout(); plt.savefig('figuras/fase6_accuracy_vs_f1.png', dpi=150); plt.show()

print("Combinaciones con mayor brecha accuracy - F1 macro, y las clases sorteadas (tamaños en prueba):")
for _, fila in peores.iterrows():
    print(f"\n{fila['modelo']} k={fila['k']} semilla={fila['semilla']}: accuracy={fila['accuracy']:.3f}, "
          f"F1 macro={fila['f1_macro']:.3f}, mayoritaria={fila['pct_mayoritaria']:.0f}% de la prueba")
    print("  clases:", fila['tamanos'])

# %% [md]
# **Clases sorteadas y tamaños.** Qué emociones salieron en cada semilla y k (tamaño en el conjunto de prueba), para interpretar las diferencias entre semillas.

# %% [code]
sorteos = tabla_larga.drop_duplicates(['k', 'semilla'])[['k', 'semilla', 'n_test', 'pct_mayoritaria', 'clases']]
sorteos['pct_mayoritaria'] = sorteos['pct_mayoritaria'].round(1)
print(sorteos.to_string(index=False))

# Dispersión entre semillas del F1 macro del SVM
print("\nF1 macro del SVM por k y semilla:")
print(tabla_larga[tabla_larga.modelo == 'SVM'].pivot(index='k', columns='semilla', values='f1_macro').round(3).to_string())

# %% [md]
# **Criterio de granularidad.** Se define de forma explícita un F1 macro mínimo aceptable de **0.30** (la mitad de las clases se reconoce con precisión y recall moderados) y una ganancia mínima de **2 veces** sobre el azar 1/k. Se busca el mayor k cuyo mejor modelo cumpla ambos.

# %% [code]
UMBRAL_F1, UMBRAL_GANANCIA = 0.30, 2.0
mejor_por_k = resumen.sort_values('f1_media').groupby('k').tail(1).sort_values('k')
mejor_por_k['cumple'] = (mejor_por_k['f1_media'] >= UMBRAL_F1) & (mejor_por_k['ganancia_vs_azar'] >= UMBRAL_GANANCIA)
print(mejor_por_k[['k', 'modelo', 'f1_media', 'f1_desv', 'acc_media', 'azar', 'ganancia_vs_azar', 'cumple']].to_string(index=False))
k_recomendado = int(mejor_por_k[mejor_por_k.cumple]['k'].max()) if mejor_por_k.cumple.any() else None
print("\nMayor k que cumple el criterio:", k_recomendado)

# Rendimiento por clase del SVM de la Fase 4 (13 clases) para decidir qué agrupar o conservar
print("\nF1 por clase del SVM con 13 clases (Fase 4):")
print(f1_por_clase['SVM'].sort_values(ascending=False).round(3).to_string())

# %% [code]
salida6 = {'semillas': SEMILLAS, 'ks': KS, 'tiempo_total_s': round(tiempo_total_f6, 1),
           'config_rnn': {'max_tokens': 15000, 'dim_emb': 32, 'unidades': 32, 'max_epocas': 4, 'paciencia': 1},
           'criterio': {'f1_macro_minimo': UMBRAL_F1, 'ganancia_minima_vs_azar': UMBRAL_GANANCIA},
           'k_recomendado': k_recomendado,
           'resumen': resumen.to_dict('records'),
           'tabla_larga': tabla_larga.round(5).to_dict('records'),
           'sorteos': sorteos.to_dict('records')}
with open('results/fase6.json', 'w') as archivo:
    json.dump(salida6, archivo, indent=2, default=str)
tabla_larga.to_csv('results/fase6_tabla_larga.csv', index=False)

# %% [conclusion]
# **Conclusión de negocio (Fase 6).** El F1 macro cae de forma sistemática al aumentar el número de emociones, y el SVM es el mejor modelo en todos los k. Promedio de 3 semillas (42, 43, 44) del SVM: 0.543 con k=3, 0.393 con k=5, 0.316 con k=7, 0.258 con k=9 y 0.229 con k=11 (NB: 0.480, 0.304, 0.237, 0.190, 0.169; RNN pequeña: 0.494, 0.329, 0.259, 0.194, 0.169). La línea de azar 1/k también baja (0.333, 0.200, 0.143, 0.111, 0.091), así que la ganancia relativa sobre el azar del SVM sube de 1.6 veces (k=3) a 2.5 veces (k=11): al tener más clases el problema se vuelve más difícil en términos absolutos, pero el modelo sigue aprendiendo algo; en cifras absolutas, la distancia sobre el azar es mayor con pocas clases (+0.21 con k=3, +0.14 con k=11). La dispersión entre semillas es grande con pocas clases (desviación del SVM de 0.095 con k=3, 0.009 con k=11) porque depende de cuáles emociones salgan: por ejemplo, con k=3 el SVM da 0.448 (anger, relief, surprise), 0.544 (hate, neutral, worry) y 0.637 (happiness, love, sadness). Accuracy y F1 macro divergen cuando salen clases muy desiguales: NB con la semilla 42 y k=7 (anger=16, boredom=27, empty=121 junto a neutral=1280 y happiness=776) tiene accuracy 0.492 y F1 macro 0.205, porque la clase neutral es 46 % de la prueba y las raras casi no se reconocen. Criterio fijado de antemano: F1 macro mínimo de 0.30 y ganancia de al menos 2 veces sobre el azar; el mayor k que lo cumple (con el mejor modelo, SVM) es 7 (F1 0.316, 2.2 veces el azar); k=5 lo supera en F1 pero queda en 1.96 veces el azar, y k=9 y 11 no llegan a 0.30. **Recomendación:** trabajar con unas 5 a 7 categorías, no con 13. Conservar las emociones grandes y con señal propia (neutral, worry, sadness, happiness, love, hate) y agrupar las pequeñas o solapadas (fun, enthusiasm y relief con happiness; anger y boredom con hate o sadness; empty y surprise en «otras»), ya que en 13 clases anger, boredom, empty y enthusiasm tienen F1 de 0.08 o menos con el SVM. Advertencia: el experimento sortea subconjuntos al azar de las emociones originales, no prueba agrupaciones; la agrupación sugerida es una hipótesis que habría que medir. Tiempo del experimento: 7.4 min con RNN reducida (embedding 32, GRU de 32, máx. 4 épocas, paciencia 1), por lo que la RNN de este experimento es más débil que la de la Fase 5.
