# %% [que]
# Se carga el corpus de 40 000 tweets con 13 emociones, se explora ANTES de limpiar (duplicados, longitud, desbalance de clases y ruido de Twitter: menciones, URLs y entidades HTML) y luego se limpia documentando cada decisión con su impacto medido. También se examina qué significa la etiqueta `empty`.

# %% [code]
import re, html, time, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import spacy
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
SEMILLA = 42
np.random.seed(SEMILLA)
Path("results").mkdir(exist_ok=True); Path("figuras").mkdir(exist_ok=True)
t_ini_fase1 = time.time()

# --- Carga: Kaggle (kagglehub) si hay acceso; si no, copia local data/tweet_emotions.csv ---
def cargar_csv():
    try:
        import kagglehub
        ruta = kagglehub.dataset_download('pashupatigupta/emotion-detection-from-text')
        return pd.read_csv(f'{ruta}/tweet_emotions.csv')
    except Exception:
        return pd.read_csv('data/tweet_emotions.csv')
df_crudo = cargar_csv()
print("Forma del corpus crudo:", df_crudo.shape)
df_crudo.head()

# %% [md]
# **Qué se analiza:** duplicados del corpus crudo (por `tweet_id`, por texto y por texto con etiquetas contradictorias). Un texto repetido con emociones distintas es ruido de etiquetado y puede contaminar el entrenamiento y la evaluación.

# %% [code]
# --- Exploración sobre el corpus CRUDO (antes de limpiar) ---
n_total = len(df_crudo)
dup_id = int(df_crudo['tweet_id'].duplicated().sum())             # mismo id repetido
dup_texto = int(df_crudo['content'].duplicated().sum())           # filas repetidas en texto (sin contar la 1.ª copia)
# textos que aparecen con más de una emoción distinta
conflicto_crudo = df_crudo.groupby('content')['sentiment'].transform('nunique') > 1
filas_conflicto = int(conflicto_crudo.sum())
textos_conflicto = int(df_crudo.loc[conflicto_crudo, 'content'].nunique())
print(f"Filas: {n_total}")
print(f"Duplicados por tweet_id: {dup_id}")
print(f"Duplicados por texto (excluyendo 1.ª copia): {dup_texto}")
print(f"Textos con etiquetas contradictorias: {textos_conflicto} textos ({filas_conflicto} filas)")
print("Nulos por columna:\n", df_crudo.isna().sum().to_string())
df_crudo.loc[conflicto_crudo].sort_values('content').head(6)[['content', 'sentiment']]

# %% [md]
# **Qué se analiza:** longitud de los tweets (caracteres y palabras) con estadísticos y su histograma, para conocer el tamaño típico del texto que verán los modelos.

# %% [code]
# Longitud en caracteres y en palabras (sobre el texto crudo)
largo_car = df_crudo['content'].str.len()
largo_pal = df_crudo['content'].str.split().str.len()
resumen_largo = pd.DataFrame({
    'caracteres': [largo_car.mean(), largo_car.median(), largo_car.min(), largo_car.max()],
    'palabras':   [largo_pal.mean(), largo_pal.median(), largo_pal.min(), largo_pal.max()],
}, index=['media', 'mediana', 'min', 'max']).round(1)
print(resumen_largo)

TINTA, ACENTO, GRIS = "#14323F", "#6E5BD6", "#B7C3C9"
def estilo(ax):
    # sin rejilla y sin marco superior/derecho
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)

fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
ejes[0].hist(largo_car, bins=40, color=GRIS)
ejes[0].axvline(largo_car.mean(), color=ACENTO, lw=2, label=f"media = {largo_car.mean():.0f}")
ejes[0].set_xlabel("Caracteres por tweet"); ejes[0].set_ylabel("Tweets"); ejes[0].legend(frameon=False)
ejes[1].hist(largo_pal, bins=range(0, 40), color=GRIS)
ejes[1].axvline(largo_pal.mean(), color=ACENTO, lw=2, label=f"media = {largo_pal.mean():.1f}")
ejes[1].set_xlabel("Palabras por tweet"); ejes[1].legend(frameon=False)
for ax in ejes: estilo(ax)
plt.tight_layout()
plt.savefig("figuras/fase1_longitud.png", dpi=150)
plt.show()

# %% [md]
# **Qué se analiza:** distribución de las 13 emociones (barras horizontales ordenadas) y la razón entre la clase más grande y la más pequeña, que mide el desbalance.

# %% [code]
conteo = df_crudo['sentiment'].value_counts()
razon_desbalance = conteo.max() / conteo.min()
print(conteo.to_frame('tweets').assign(porcentaje=(conteo / n_total * 100).round(1)))
print(f"\nRazón mayor/menor clase: {conteo.idxmax()} ({conteo.max()}) / {conteo.idxmin()} ({conteo.min()}) = {razon_desbalance:.1f}x")

orden = conteo.sort_values()   # de menor a mayor para que la mayor quede arriba
colores = [ACENTO if e == 'empty' else GRIS for e in orden.index]   # ACENTO solo para el dato clave: empty
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(orden.index, orden.values, color=colores)
for i, v in enumerate(orden.values):
    ax.text(v + 80, i, f"{v:,}", va='center', color=TINTA, fontsize=9)
ax.set_xlabel("Número de tweets")
estilo(ax)
plt.tight_layout()
plt.savefig("figuras/fase1_distribucion_emociones.png", dpi=150)
plt.show()

# %% [md]
# **Qué se analiza:** qué significa la etiqueta `empty`. Se comparan sus ejemplos y su longitud contra el resto del corpus para decidir si es un dato faltante o una categoría con contenido.

# %% [code]
vacia = df_crudo[df_crudo['sentiment'] == 'empty']
print("Ejemplos de tweets con etiqueta 'empty':")
for t in vacia['content'].sample(12, random_state=SEMILLA):
    print("  -", t)
# ¿tiene el texto vacío o nulo? (si fuera dato faltante, el texto estaría vacío)
print("\n'empty' con texto nulo o vacío:", int((vacia['content'].isna() | (vacia['content'].str.strip() == '')).sum()))
print("Longitud media 'empty':", round(vacia['content'].str.len().mean(), 1),
      "| resto:", round(df_crudo.loc[df_crudo['sentiment'] != 'empty', 'content'].str.len().mean(), 1))
# ¿es empty simplemente texto de otra clase? Comparación con neutral
print("Menciones (@) en 'empty':", round(vacia['content'].str.contains('@').mean() * 100, 1), "% | en neutral:",
      round(df_crudo.loc[df_crudo['sentiment'] == 'neutral', 'content'].str.contains('@').mean() * 100, 1), "%")

# %% [md]
# **Qué se analiza:** cuánto ruido propio de Twitter hay en el texto crudo (menciones, URLs, entidades HTML, hashtags) para justificar las decisiones de limpieza.

# %% [code]
# Conteos de ruido sobre el corpus crudo (tweets que contienen cada elemento)
patron_mencion = r'@\w+'
patron_url = r'https?://\S+|www\.\S+'
patron_html = r'&(?:#\d+|\w+);'
n_mencion = int(df_crudo['content'].str.contains(patron_mencion).sum())
n_url = int(df_crudo['content'].str.contains(patron_url).sum())
n_html = int(df_crudo['content'].str.contains(patron_html).sum())
n_hashtag = int(df_crudo['content'].str.contains('#').sum())
print("Tweets con menciones:", n_mencion, f"({n_mencion / n_total * 100:.1f}%)")
print("Tweets con URLs:", n_url, f"({n_url / n_total * 100:.1f}%)")
print("Tweets con entidades HTML:", n_html, f"({n_html / n_total * 100:.1f}%)")
print("Tweets con '#':", n_hashtag, f"({n_hashtag / n_total * 100:.1f}%)")
print("Entidades HTML más frecuentes:\n", df_crudo['content'].str.findall(patron_html).explode().value_counts().head(5).to_string())

# %% [code]
# --- Limpieza (decisiones documentadas en la tabla de más abajo) ---
def limpiar(t):
    t = html.unescape(t)                       # &amp; -> &, &quot; -> "
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)   # URLs
    t = re.sub(r'@\w+', ' ', t)                # menciones @usuario
    t = t.replace('#', '')                     # hashtag: conserva la palabra
    return re.sub(r'\s+', ' ', t).strip()      # espacios múltiples

df = df_crudo.copy()
# duplicados: textos con etiquetas contradictorias se eliminan todos; el resto, se conserva la 1.ª copia
conflicto = df.groupby('content')['sentiment'].transform('nunique') > 1
df = df[~conflicto].drop_duplicates('content').copy()
df['texto_limpio'] = df['content'].map(limpiar)
df = df[df['texto_limpio'].str.len() > 0].reset_index(drop=True)   # tweets que quedaron vacíos (solo menciones/URL)
ETIQUETAS = sorted(df['sentiment'].unique())

# %% [md]
# **Qué se analiza:** impacto de cada decisión de limpieza (cuántos tweets afecta) y el tamaño final del corpus.

# %% [code]
# Medición del impacto de cada decisión
sin_conflicto = df_crudo[~conflicto]
dup_exactos_restantes = int(sin_conflicto['content'].duplicated().sum())
n_cambia_mencion = n_mencion
quedan_vacios = n_total - filas_conflicto - dup_exactos_restantes - len(df)  # vacíos entre los que sobrevivían a duplicados
tabla_decisiones = pd.DataFrame([
    ["Eliminar textos con etiquetas contradictorias", "El mismo texto con 2+ emociones no tiene respuesta correcta única; se descartan todas sus copias", filas_conflicto],
    ["Conservar solo la 1.ª copia de textos duplicados", "Copias idénticas inflan métricas y pueden quedar en entrenamiento y prueba a la vez (fuga de datos)", dup_exactos_restantes],
    ["Decodificar entidades HTML (&quot;, &amp;, &lt;, &gt;)", "Son artefactos de la exportación; el modelo debe ver '\"' y '&', no códigos", n_html],
    ["Quitar URLs", "Direcciones únicas sin valor emocional; solo añaden vocabulario raro", n_url],
    ["Quitar menciones @usuario", "Nombres de usuario no expresan emoción y fragmentan el vocabulario", n_mencion],
    ["Quitar '#' y conservar la palabra", "El hashtag suele llevar el tema o emoción (p. ej. #fail)", n_hashtag],
    ["Colapsar espacios múltiples", "Normaliza el texto tras las eliminaciones", int(df_crudo['content'].str.contains(r'\s{2,}').sum())],
    ["Descartar tweets que quedan vacíos tras limpiar", "Si solo había menciones/URLs no queda texto que clasificar", quedan_vacios],
    ["NO pasar a minúsculas ni quitar emoticonos/puntuación aquí", "Se deja para cada modelo (vectorizadores y spaCy lo gestionan; emoticonos como :( aportan señal)", 0],
    ["Conservar la clase `empty` como 13.ª emoción", "La guía pide matriz de 13 clases; es una etiqueta ruidosa, no dato faltante", int((df_crudo['sentiment'] == 'empty').sum())],
], columns=['decisión', 'por qué', 'tweets afectados'])
pd.set_option('display.max_colwidth', 110)
print(tabla_decisiones.to_string(index=False))
n_final = len(df)
perdidas = n_total - n_final
print(f"\nCorpus: {n_total} -> {n_final} tweets tras limpiar ({perdidas} eliminados, {perdidas / n_total * 100:.2f}%)")
print("Clases tras limpiar:", len(ETIQUETAS), "| 'empty' conservada:", int((df['sentiment'] == 'empty').sum()))
print("Longitud media tras limpiar: %.1f caracteres, %.1f palabras" % (df['texto_limpio'].str.len().mean(), df['texto_limpio'].str.split().str.len().mean()))
cambia = df['content'] != df['texto_limpio']
print("Tweets modificados por la limpieza:", int(cambia.sum()), f"({cambia.mean() * 100:.1f}%)")
print(df.loc[cambia, ['content', 'texto_limpio']].head(4).to_string(index=False))

# %% [code]
# Guardar cifras clave de la fase
conteo_limpio = df['sentiment'].value_counts()
resultados_fase1 = {
    "filas_crudas": int(n_total), "filas_limpias": int(n_final),
    "dup_tweet_id": dup_id, "dup_texto": dup_texto,
    "textos_contradictorios": textos_conflicto, "filas_contradictorias": filas_conflicto,
    "dup_exactos_tras_conflicto": dup_exactos_restantes, "vacios_tras_limpiar": int(quedan_vacios),
    "longitud_caracteres": {k: float(v) for k, v in zip(['media', 'mediana', 'min', 'max'], resumen_largo['caracteres'])},
    "longitud_palabras": {k: float(v) for k, v in zip(['media', 'mediana', 'min', 'max'], resumen_largo['palabras'])},
    "distribucion_cruda": {k: int(v) for k, v in conteo.items()},
    "distribucion_limpia": {k: int(v) for k, v in conteo_limpio.items()},
    "razon_desbalance_cruda": float(razon_desbalance),
    "razon_desbalance_limpia": float(conteo_limpio.max() / conteo_limpio.min()),
    "pct_neutral_worry": float((conteo['neutral'] + conteo['worry']) / n_total * 100),
    "tweets_con_menciones": n_mencion, "tweets_con_urls": n_url, "tweets_con_html": n_html, "tweets_con_hashtag": n_hashtag,
    "empty_n": int(len(vacia)), "empty_pct": float(len(vacia) / n_total * 100),
    "tabla_decisiones": tabla_decisiones.to_dict(orient='records'),
    "tiempo_s": round(time.time() - t_ini_fase1, 1),
}
with open("results/fase1.json", "w") as f:
    json.dump(resultados_fase1, f, ensure_ascii=False, indent=2)
print("Guardado results/fase1.json")

# %% [conclusion]
# **Conclusión de negocio (Fase 1).**
# - El corpus tiene 40 000 tweets sin nulos ni `tweet_id` repetidos, pero 173 textos duplicados y 62 textos (207 filas) con emociones contradictorias: se descartan esas 207 filas y las copias repetidas (28), y 77 tweets más quedan vacíos tras limpiar. Quedan 39 688 tweets (0,78 % menos).
# - El tweet típico es corto: media de 73,4 caracteres y 13,2 palabras (mediana 69 y 12; máximo 167 caracteres), por lo que los modelos trabajan con muy poco contexto por ejemplo.
# - El desbalance es fuerte: `neutral` (8 638) y `worry` (8 459) suman 42,7 % del corpus, mientras `anger` (110) y `boredom` (179) apenas llegan a 0,3 % y 0,4 %. La razón mayor/menor es 78,5 veces: un modelo que solo prediga `neutral` ya acierta 21,6 %, así que la métrica correcta es F1 macro y no la exactitud.
# - `empty` (827 tweets, 2,1 %) NO es dato faltante: ningún texto está vacío; son tweets con texto normal a los que el anotador no asignó una emoción identificable (algunos incluso expresan aburrimiento o enfado). Es una etiqueta ruidosa; se conserva como 13.ª clase y se espera que sea de las más difíciles de predecir.
# - El texto crudo es ruidoso: 47,4 % de los tweets tiene menciones, 5,8 % entidades HTML y 4,8 % URLs. La limpieza modifica el 66,3 % de los tweets y reduce la longitud media a 65,6 caracteres (12,7 palabras), sin perder ninguna de las 13 clases.
