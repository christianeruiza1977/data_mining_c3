# %% [que]
# Se replica la clasificación de emociones con un Pipeline de Spark MLlib (tokenizar, quitar stopwords, TF-IDF y Naive Bayes) sobre la MISMA partición 70/15/15 de sklearn, y se hace un conteo de palabras estilo MapReduce con RDD. Además se lee el plan con explain(), se consulta con Spark SQL y una función de ventana (top-5 palabras por emoción), se comparan tiempos con sklearn y se estima cuándo un clúster (EMR + S3) se justifica de verdad.

# %% [md]
# **Spark en modo local y la misma partición que sklearn.** Se crea la sesión de Spark (4 GB para el driver) y se pasan a Spark exactamente los mismos tweets de entrenamiento, validación y prueba de la Fase 4 (de pandas a Spark). Así la comparación con sklearn es justa: mismo dato, misma partición, misma semilla. En producción este paso sería `spark.read.csv('s3a://...')` o HDFS, no `createDataFrame` desde pandas.

# %% [code]
import os, sys, io, re, time, json, contextlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)   # los workers de Spark usan el mismo Python del notebook
t0 = time.time()
spark = (SparkSession.builder.appName('emociones').master('local[*]')
         .config('spark.driver.memory', '4g').getOrCreate())  # 4 GB: el valor por defecto (1 GB) se queda corto con TF-IDF
spark.sparkContext.setLogLevel('ERROR')                      # menos ruido en la salida
sdf = spark.createDataFrame(df[['texto_limpio', 'sentiment']])  # corpus completo (para MapReduce, SQL y ventana)

def a_spark(textos, etiquetas):
    """Pasa una partición de sklearn (Series) a un DataFrame de Spark con las mismas columnas del corpus."""
    return spark.createDataFrame(pd.DataFrame({'texto_limpio': textos.values, 'sentiment': etiquetas.values}))

sdf_train, sdf_val, sdf_test = a_spark(X_train, y_train), a_spark(X_val, y_val), a_spark(X_test, y_test)
t_sesion = time.time() - t0
print(f"Spark {spark.version} listo en {t_sesion:.1f} s | núcleos locales: {spark.sparkContext.defaultParallelism}")
print(f"Filas -> corpus: {sdf.count()}, train: {sdf_train.count()}, val: {sdf_val.count()}, test: {sdf_test.count()}")
print("Mismas cifras que sklearn:", (sdf_train.count(), sdf_val.count(), sdf_test.count()) == (len(X_train), len(X_val), len(X_test)))

# %% [md]
# **Pipeline de MLlib.** Cada etapa transforma una columna del DataFrame y la siguiente la consume: `StringIndexer` (emoción → número, en orden alfabético igual que `ETIQUETAS`), `RegexTokenizer` (texto → palabras en minúscula), `StopWordsRemover` (quita «the», «and»…), `HashingTF` (cuenta palabras en un vector de 2^15 posiciones) e `IDF` (pondera: palabras raras pesan más). El clasificador es `NaiveBayes` multinomial. Se ajusta un solo hiperparámetro (`smoothing`) con la validación y la prueba se usa UNA vez.

# %% [code]
# Etapas de características (comunes a todas las pruebas de smoothing)
indexador = StringIndexer(inputCol='sentiment', outputCol='label', stringOrderType='alphabetAsc')  # orden alfabético = ETIQUETAS
tokenizador = RegexTokenizer(inputCol='texto_limpio', outputCol='palabras', pattern=r"[^a-zA-Z']+", toLowercase=True)
sin_stop = StopWordsRemover(inputCol='palabras', outputCol='sin_stop')   # lista de stopwords en inglés por defecto
tf = HashingTF(inputCol='sin_stop', outputCol='tf', numFeatures=2**15)
idf = IDF(inputCol='tf', outputCol='features')
etapas_car = [indexador, tokenizador, sin_stop, tf, idf]

# Se ajustan las características UNA vez sobre train y se reutilizan para probar varios smoothing en validación
t0 = time.time()
modelo_car = SparkPipeline(stages=etapas_car).fit(sdf_train)
train_f = modelo_car.transform(sdf_train).select('features', 'label').cache()
val_f = modelo_car.transform(sdf_val).select('features', 'label').cache()
t_car = time.time() - t0
print(f"Características listas en {t_car:.1f} s; el orden de clases coincide con ETIQUETAS:",
      modelo_car.stages[0].labels == list(ETIQUETAS))

def f1_macro_spark(pred_sdf):
    """F1 macro calculado con sklearn sobre las predicciones recogidas (MLlib 'f1' es PONDERADO, no macro)."""
    p = pred_sdf.select('label', 'prediction').toPandas()
    return f1_score(p['label'], p['prediction'], average='macro'), accuracy_score(p['label'], p['prediction'])

t0 = time.time()
filas = []
for suavizado in [0.1, 0.3, 1.0]:                      # smoothing de Laplace del Naive Bayes
    nb = NaiveBayes(smoothing=suavizado, modelType='multinomial').fit(train_f)
    f1_v, acc_v = f1_macro_spark(nb.transform(val_f))
    filas.append({'smoothing': suavizado, 'f1_macro_val': round(f1_v, 4), 'accuracy_val': round(acc_v, 4)})
tabla_val = pd.DataFrame(filas)
t_ajuste = time.time() - t0   # tiempo de probar los 3 valores de smoothing
print(tabla_val.to_string(index=False))
mejor_smoothing = float(tabla_val.sort_values('f1_macro_val', ascending=False).iloc[0]['smoothing'])
print("Mejor smoothing en validación:", mejor_smoothing)

# %% [code]
# Modelo final: Pipeline COMPLETO (características + NB) ajustado sobre train; se mide su tiempo de entrenamiento
nb_final = NaiveBayes(smoothing=mejor_smoothing, modelType='multinomial')
t0 = time.time()
pipeline_spark = SparkPipeline(stages=etapas_car + [nb_final]).fit(sdf_train)
tiempo_spark = time.time() - t0

# Evaluación en PRUEBA, una sola vez
t0 = time.time()
pred_spark = pipeline_spark.transform(sdf_test)
f1_spark, acc_spark = f1_macro_spark(pred_spark)
f1_pond_spark = MulticlassClassificationEvaluator(labelCol='label', metricName='f1').evaluate(pred_spark)  # ponderado, solo informativo
t_pred_spark = time.time() - t0
print(f"Spark MLlib (TF-IDF + NB, smoothing={mejor_smoothing}) en prueba:")
print(f"  F1 macro = {f1_spark:.4f} (calculado con sklearn sobre las predicciones)  |  accuracy = {acc_spark:.4f}")
print(f"  F1 ponderado de MLlib = {f1_pond_spark:.4f} (NO es el macro; solo se informa)")
print(f"  Entrenamiento del Pipeline: {tiempo_spark:.1f} s  |  predicción + evaluación: {t_pred_spark:.1f} s")

# Se añade la fila a la tabla global de resultados (fase 7)
nombre_spark = f'Spark MLlib (TF-IDF + NB, smoothing={mejor_smoothing})'
resultados = pd.concat([resultados, pd.DataFrame([{'modelo': nombre_spark, 'fase': 7, 'f1_macro': round(f1_spark, 4),
                                                   'accuracy': round(acc_spark, 4), 'tiempo_s': round(tiempo_spark, 2)}])],
                       ignore_index=True)
print(resultados.to_string(index=False))

# %% [md]
# **Conteo de palabras estilo MapReduce con RDD.** Fase *map*: cada tweet se convierte en pares (palabra, 1) con `flatMap` + `map`. *Shuffle*: Spark agrupa los pares con la misma palabra en la misma partición (en Hadoop esto se escribe a disco). Fase *reduce*: `reduceByKey` suma los 1 de cada palabra (primero dentro de cada partición, como un *combiner*, y luego entre particiones). Se filtran stopwords y palabras de menos de 3 letras para que el top sea interpretable. Después se repite con DataFrame para comparar.

# %% [code]
parar = set(StopWordsRemover.loadDefaultStopWords('english'))   # mismas stopwords que el pipeline
parar_bc = spark.sparkContext.broadcast(parar)                   # se envía una copia a cada ejecutor (como la caché distribuida de Hadoop)

def palabras_de(texto):
    """Fase MAP (parte 1): un tweet -> lista de palabras útiles."""
    return [p for p in re.findall(r"[a-z']+", texto.lower()) if len(p) > 2 and p not in parar_bc.value]

# --- MapReduce con RDD ---
t0 = time.time()
rdd = sdf.rdd.map(lambda fila: fila['texto_limpio'])             # RDD de textos
conteo_rdd = (rdd.flatMap(palabras_de)                           # MAP: tweet -> palabras
                 .map(lambda w: (w, 1))                          # MAP: palabra -> (palabra, 1)
                 .reduceByKey(lambda a, b: a + b))               # SHUFFLE + REDUCE: suma por palabra
top15_rdd = conteo_rdd.takeOrdered(15, key=lambda kv: (-kv[1], kv[0]))   # top 15 (desempate alfabético)
t_rdd = time.time() - t0

# --- El mismo conteo con DataFrame (Catalyst lo optimiza) ---
t0 = time.time()
palabras_df = (sdf.select(F.explode(F.split(F.lower('texto_limpio'), r"[^a-z']+")).alias('palabra'))
                  .filter((F.length('palabra') > 2) & (~F.col('palabra').isin(list(parar)))))
top15_df = palabras_df.groupBy('palabra').count().orderBy(F.desc('count'), 'palabra').limit(15).collect()
t_df = time.time() - t0

print(f"RDD (MapReduce): {t_rdd:.1f} s | DataFrame: {t_df:.1f} s")
print("Top 15 con RDD:", top15_rdd)
print("Mismo top 15 con DataFrame:", [(r['palabra'], r['count']) for r in top15_df] == top15_rdd)
print(f"Palabras distintas: {conteo_rdd.count()} | particiones del RDD: {rdd.getNumPartitions()}")

# %% [md]
# **Qué se analiza:** las 15 palabras más frecuentes del corpus (sin stopwords), obtenidas con el conteo MapReduce. Sirve para ver que el resultado del RDD es el vocabulario dominante y para comprobar visualmente que el filtro dejó palabras con significado.

# %% [code]
palabras, cuentas = zip(*top15_rdd[::-1])        # invertido para que la más frecuente quede arriba
fig, ax = plt.subplots(figsize=(7, 5))
colores = [ACENTO if i == len(palabras) - 1 else GRIS for i in range(len(palabras))]   # ACENTO solo para la más frecuente
ax.barh(palabras, cuentas, color=colores)
for i, v in enumerate(cuentas):
    ax.text(v, i, f" {v:,}", va='center', fontsize=8, color=TINTA)
ax.set_xlabel('Apariciones en el corpus', color=TINTA)
ax.set_title('Top 15 palabras (conteo MapReduce con RDD)', color=TINTA, loc='left')
ax.spines[['top', 'right']].set_visible(False); ax.grid(False)
plt.tight_layout(); plt.savefig('figuras/fase7_top_palabras_mapreduce.png', dpi=150); plt.show()

# %% [md]
# **Plan de ejecución con `explain()`.** Cada consulta pasa por cuatro planes: *Parsed* (lo que se escribió), *Analyzed* (con columnas y tipos resueltos), *Optimized* (tras las reglas de Catalyst: poda de columnas, filtros adelantados) y *Physical* (los operadores reales). Se busca `Exchange`: es el **shuffle**, el equivalente a la fase shuffle de MapReduce, donde los datos viajan entre particiones y se paga en red/disco.

# %% [code]
def texto_plan(consulta, modo):
    """Captura lo que imprime explain() para poder buscar operadores dentro del plan."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        consulta.explain(mode=modo)
    return buf.getvalue()

agregacion = sdf.groupBy('sentiment').agg(F.count('*').alias('n_tweets'), F.avg(F.length('texto_limpio')).alias('long_media'))
plan_ext = texto_plan(agregacion, 'extended')
print(plan_ext)
plan_fmt = texto_plan(agregacion, 'formatted')
print(plan_fmt)
def n_shuffles(plan_texto):
    """Cuenta los operadores Exchange del plan físico (en 'extended' solo aparecen en el árbol, una vez cada uno)."""
    return len(re.findall(r'Exchange (?:hash|range)?[Pp]artitioning|Exchange SinglePartition', plan_texto.split('== Physical Plan ==')[-1]))
n_exchange = n_shuffles(plan_ext)
print(f"Operadores Exchange (shuffle) en el plan físico: {n_exchange}")
print("HashAggregate presente 2 veces (parcial antes del shuffle y final después):", plan_fmt.count('HashAggregate') >= 2)

# %% [md]
# **Cómo leer el plan anterior.** (1) *Parsed/Analyzed*: `Aggregate [sentiment], [sentiment, count(1), avg(length(texto_limpio))]` sobre la relación local creada desde pandas. (2) *Optimized*: aquí es idéntico al Analyzed porque la consulta ya es mínima; con filtros o `select` Catalyst los empuja hacia la lectura y poda columnas (con Parquet se vería en `PushedFilters`/`ReadSchema`). (3) *Physical*: `HashAggregate (partial)` → `Exchange hashpartitioning(sentiment, ...)` → `HashAggregate (final)`. Ese `Exchange` es el shuffle: primero cada partición agrega lo suyo (como el *combiner* de MapReduce) y solo viajan los totales parciales por emoción. Lo mismo que en el conteo con `reduceByKey`, pero con el plan elegido por el optimizador.

# %% [md]
# **Spark SQL.** La misma agregación escrita en SQL: se registra el DataFrame como vista temporal con `createOrReplaceTempView` y se consulta con `spark.sql`. SQL y DataFrame generan el mismo plan lógico, así que el rendimiento es igual; cambia solo la comodidad (SQL para analistas, DataFrame para código).

# %% [code]
sdf.createOrReplaceTempView('tweets')
consulta_sql = """
    SELECT sentiment, COUNT(*) AS n_tweets, ROUND(AVG(LENGTH(texto_limpio)), 1) AS long_media
    FROM tweets
    GROUP BY sentiment
    ORDER BY n_tweets DESC
"""
res_sql = spark.sql(consulta_sql).toPandas()
print(res_sql.to_string(index=False))

# Verificación contra pandas (debe dar lo mismo)
ref = df.groupby('sentiment')['texto_limpio'].agg(n_tweets='size', long_media=lambda s: s.str.len().mean()).round(1)
comp = res_sql.set_index('sentiment').join(ref, rsuffix='_pandas')
print("Spark SQL = pandas (conteos):", (comp['n_tweets'] == comp['n_tweets_pandas']).all(),
      "| longitudes (tolerancia 0.1):", np.allclose(comp['long_media'], comp['long_media_pandas'], atol=0.1))
plan_sql = texto_plan(spark.sql(consulta_sql), 'extended')
print("Exchange en el plan de la consulta SQL:", n_shuffles(plan_sql), "(uno para agrupar por emoción y otro para ordenar globalmente por n_tweets)")

# %% [md]
# **Función de ventana: top-5 palabras por emoción.** Primero se cuentan palabras por emoción (`explode(split(...))` + `groupBy`) y luego `row_number()` numera las palabras dentro de cada emoción (`Window.partitionBy(sentiment).orderBy(desc(conteo))`); se conservan las posiciones 1 a 5. Se filtran stopwords y palabras de menos de 3 letras para que la lista sea interpretable. La misma consulta se escribe también en SQL para mostrar que es equivalente.

# %% [code]
conteo_emo = (sdf.select('sentiment', F.explode(F.split(F.lower('texto_limpio'), r"[^a-z']+")).alias('palabra'))
                                   .filter((F.length('palabra') > 2) & (~F.col('palabra').isin(list(parar))))
                                   .groupBy('sentiment', 'palabra').count())
ventana = Window.partitionBy('sentiment').orderBy(F.desc('count'), F.asc('palabra'))   # desempate alfabético = resultado repetible
top5 = conteo_emo.withColumn('posicion', F.row_number().over(ventana)).filter(F.col('posicion') <= 5)
top5_pd = top5.orderBy('sentiment', 'posicion').toPandas()

# Una fila por emoción: "palabra (conteo), ..."
resumen_top5 = (top5_pd.assign(par=lambda d: d['palabra'] + ' (' + d['count'].astype(str) + ')')
                       .groupby('sentiment')['par'].apply(lambda s: ', '.join(s)))
for emo, fila in resumen_top5.items():
    print(f"{emo:<11}{fila}")

# Versión SQL equivalente
conteo_emo.createOrReplaceTempView('conteo_emocion')
top5_sql = spark.sql("""
    SELECT sentiment, palabra, count, posicion FROM (
        SELECT sentiment, palabra, count,
               ROW_NUMBER() OVER (PARTITION BY sentiment ORDER BY count DESC, palabra ASC) AS posicion
        FROM conteo_emocion) t
    WHERE posicion <= 5
    ORDER BY sentiment, posicion""").toPandas()
print("SQL con ROW_NUMBER() = versión DataFrame:", top5_sql.equals(top5_pd[['sentiment', 'palabra', 'count', 'posicion']].astype(top5_sql.dtypes.to_dict())))
plan_ventana = texto_plan(top5, 'extended')
print("Operador Window en el plan:", 'Window' in plan_ventana, "| Exchange:", n_shuffles(plan_ventana),
      "(shuffle por sentiment para reunir cada emoción en una partición)")

# %% [md]
# **Qué se analiza:** cuánto tardan sklearn y Spark en entrenar sobre el MISMO dato (unos 40 000 tweets; ≈ 2.6 MB de texto limpio, el archivo CSV original pesa ≈ 3.8 MB) y qué F1 macro obtiene cada uno. Se espera que Spark local sea MÁS lento: el costo fijo de la JVM, la planificación de tareas y la serialización no se amortizan con un conjunto tan pequeño.

# %% [code]
# Comparación del MISMO algoritmo (TF-IDF + Naive Bayes, alpha=1) en sklearn, para que sea justa con Spark
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
t0 = time.time()
nb_sk = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nb', MultinomialNB(alpha=1.0))]).fit(X_train, y_train)
tiempo_sk_nb = time.time() - t0
t0 = time.time()
pred_sk_nb = nb_sk.predict(X_test)
t_pred_sk_nb = time.time() - t0
f1_sk_nb = f1_score(y_test, pred_sk_nb, average='macro')
acc_sk_nb = accuracy_score(y_test, pred_sk_nb)
fila_nb = pd.DataFrame([{'modelo': 'sklearn TF-IDF + NB (mismo algoritmo)', 'fase': 4, 'f1_macro': round(f1_sk_nb, 4),
                         'accuracy': round(acc_sk_nb, 4), 'tiempo_s': round(tiempo_sk_nb, 2)}])
sk = pd.concat([resultados[resultados['fase'].isin([4, 7])], fila_nb], ignore_index=True)   # modelos clásicos de sklearn y Spark
sk['motor'] = np.where(sk['fase'] == 7, 'Spark', 'sklearn')
sk = sk.drop_duplicates('modelo', keep='first').reset_index(drop=True)

# Tiempo de punta a punta de Spark (sesión + características + ajuste de smoothing + entrenamiento final + predicción)
t_total_spark = t_sesion + t_car + t_ajuste + tiempo_spark + t_pred_spark
print(f"sklearn NB: entrenar {tiempo_sk_nb:.2f} s + predecir {t_pred_sk_nb:.2f} s | Spark: entrenar el Pipeline {tiempo_spark:.1f} s, "
      f"y de punta a punta (sesión, características, ajuste, entrenamiento, predicción) {t_total_spark:.1f} s")
print(f"Spark local tarda {t_total_spark / (tiempo_sk_nb + t_pred_sk_nb):.0f} veces más que sklearn NB en el flujo completo")
print(sk[['modelo', 'motor', 'f1_macro', 'accuracy', 'tiempo_s']].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.8))
nombres = [m if len(m) < 34 else m[:31] + '...' for m in sk['modelo']]
colores = [ACENTO if mo == 'Spark' else GRIS for mo in sk['motor']]   # ACENTO = Spark
ax.barh(nombres, sk['tiempo_s'], color=colores)
for i, (t, f) in enumerate(zip(sk['tiempo_s'], sk['f1_macro'])):
    ax.text(t, i, f"  {t:.1f} s | F1 macro {f:.3f}", va='center', fontsize=8, color=TINTA)
ax.set_xlim(0, sk['tiempo_s'].max() * 1.6)
ax.set_xlabel('Tiempo de entrenamiento (s)', color=TINTA)
ax.set_title('Mismo dato, mismo split: sklearn vs Spark MLlib (local)', color=TINTA, loc='left')
ax.spines[['top', 'right']].set_visible(False); ax.grid(False)
plt.tight_layout(); plt.savefig('figuras/fase7_tiempos_sklearn_vs_spark.png', dpi=150); plt.show()

# %% [md]
# **Cuándo basta un solo nodo y cuándo se justifica Spark.** Se mide el tamaño real por tweet y cuánta memoria ocupa la matriz TF-IDF en sklearn respecto al texto original; con ese factor se estiman tres escenarios (10 mil, 1 millón y 100 millones de tweets al día) para un año de historia. Los supuestos van explícitos en el código: el registro completo (texto + usuario + fecha + metadatos) se asume de 500 bytes y la máquina de referencia es un servidor de 64 GB de RAM donde se acepta usar hasta la mitad para el modelo.

# %% [code]
# --- Medidas reales del corpus ---
bytes_texto = df['texto_limpio'].str.encode('utf-8').str.len()
bytes_medio = float(bytes_texto.mean())
mb_corpus = float(bytes_texto.sum()) / 1e6
mb_crudo = float(df['content'].str.encode('utf-8').str.len().sum()) / 1e6   # texto original sin limpiar
tfidf_ref = TfidfVectorizer(min_df=2, sublinear_tf=True)
M = tfidf_ref.fit_transform(X_train)
bytes_matriz = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
factor_ram = bytes_matriz / X_train.str.encode('utf-8').str.len().sum()   # RAM de la matriz TF-IDF por byte de texto
print(f"Texto por tweet: {bytes_medio:.0f} bytes en promedio | corpus completo: {mb_corpus:.2f} MB de texto limpio ({mb_crudo:.2f} MB de texto original)")
print(f"Matriz TF-IDF (train): {bytes_matriz / 1e6:.1f} MB = {factor_ram:.1f}x el tamaño del texto original")

# --- Supuestos explícitos (NO medidos) ---
BYTES_REGISTRO = 500          # supuesto: texto + usuario + fecha + metadatos de un tweet (bytes)
DIAS_HISTORIA = 365           # supuesto: se quiere reentrenar con un año de historia
RAM_NODO_GB = 64              # supuesto: un servidor de 64 GB de RAM
USO_MAX_RAM = 0.5             # supuesto: se acepta usar hasta el 50 % de la RAM para datos + matriz
escenarios = []
for nombre, tweets_dia in [('10 mil/día', 10_000), ('1 millón/día', 1_000_000), ('100 millones/día', 100_000_000)]:
    gb_dia = tweets_dia * BYTES_REGISTRO / 1e9
    gb_anio = gb_dia * DIAS_HISTORIA
    gb_ram = gb_anio * (1 + factor_ram)                    # datos crudos + matriz TF-IDF en memoria
    cabe = gb_ram <= RAM_NODO_GB * USO_MAX_RAM
    gb_ram_30d = gb_dia * 30 * (1 + factor_ram)            # alternativa: reentrenar solo con una ventana de 30 días
    escenarios.append({'escenario': nombre, 'tweets_dia': tweets_dia, 'GB_dia': round(gb_dia, 3), 'GB_anio': round(gb_anio, 1),
                       'RAM_estimada_GB': round(gb_ram, 1), 'cabe_en_un_nodo': bool(cabe),
                       'RAM_30d_GB': round(gb_ram_30d, 1), 'cabe_30d': bool(gb_ram_30d <= RAM_NODO_GB * USO_MAX_RAM)})
tabla_esc = pd.DataFrame(escenarios)
print(tabla_esc.to_string(index=False))
print(f"Referencia: el corpus del ejercicio pesa {mb_corpus:.1f} MB de texto; "
      f"un año a 10 mil/día son {tabla_esc.loc[0, 'GB_anio']} GB.")

# %% [md]
# **Qué se analiza:** cuánta memoria necesitaría cada escenario (un año de historia) frente al límite que se acepta usar en un nodo de 64 GB. Donde la barra supera la línea punteada, un solo nodo ya no alcanza y se justifica distribuir con Spark.

# %% [code]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
colores = [GRIS if c else ACENTO for c in tabla_esc['cabe_en_un_nodo']]   # ACENTO = ya no cabe en un nodo
ax.bar(tabla_esc['escenario'], tabla_esc['RAM_estimada_GB'], color=colores)
ax.set_yscale('log')
ax.axhline(RAM_NODO_GB * USO_MAX_RAM, color=TINTA, linestyle='--', linewidth=1)
ax.text(2.45, RAM_NODO_GB * USO_MAX_RAM * 1.15, f"límite: {RAM_NODO_GB * USO_MAX_RAM:.0f} GB (50 % de 64 GB)", ha='right', fontsize=8, color=TINTA)
for i, v in enumerate(tabla_esc['RAM_estimada_GB']):
    ax.text(i, v * 1.12, f"{v:,.1f} GB", ha='center', fontsize=8, color=TINTA)
ax.set_ylabel('RAM estimada (GB, escala log)', color=TINTA)
ax.set_title('Un año de historia: ¿cabe en un solo nodo?', color=TINTA, loc='left')
ax.spines[['top', 'right']].set_visible(False); ax.grid(False)
plt.tight_layout(); plt.savefig('figuras/fase7_volumen_un_nodo_vs_cluster.png', dpi=150); plt.show()

# %% [md]
# **De Hadoop a Spark, paso por paso.**
# - **Almacenamiento: HDFS ↔ S3.** HDFS parte cada archivo en bloques de 128 MB, los replica 3 veces y los reparte en DataNodes; el NameNode guarda el mapa de bloques. En AWS se usa S3 (objetos, `s3a://bucket/ruta`): almacenamiento separado del cómputo, más barato, durable y sin NameNode que administrar. Por eso `spark.createDataFrame(df)` de este notebook sería `spark.read.csv('s3a://bucket/tweets/')` o, mejor, `spark.read.parquet(...)`.
# - **Cómputo: MapReduce ↔ Spark.** MapReduce escribe a disco el resultado de cada fase (map → disco → shuffle → reduce → disco): robusto, pero lento en trabajos iterativos. Spark construye un DAG de transformaciones perezosas y mantiene datos en memoria entre etapas (`cache()`), por eso un Pipeline de MLlib con varias pasadas sobre los datos es mucho más rápido en clúster. El `reduceByKey` de arriba es el mismo patrón map/shuffle/reduce.
# - **Recursos: YARN ↔ EMR.** YARN reparte CPU y memoria entre aplicaciones del clúster. En AWS, **EMR** lo levanta ya configurado: un nodo maestro (coordina y corre el driver), nodos *core* (cómputo + almacenamiento HDFS temporal) y nodos *task* (solo cómputo, ideales para instancias *spot*, que cuestan bastante menos pero pueden interrumpirse). **EMR Serverless** evita administrar nodos: se envía el trabajo y AWS ajusta los recursos.
# - **Despliegue típico de este trabajo:** (1) subir `tweet_emotions.csv` (o mejor Parquet) a S3; (2) empaquetar el script con el Pipeline; (3) `spark-submit --deploy-mode cluster --master yarn s3://bucket/codigo/entrenar.py` en un paso (*step*) de EMR; (4) guardar el modelo y las predicciones en S3 en Parquet; (5) permisos con roles IAM (el rol de EMR lee/escribe solo ese bucket); (6) apagar el clúster al terminar (clústeres transitorios) y usar spot en nodos task para bajar el costo. Para datos personales, además, cifrado en S3 y región/retención acordes con la LOPDP.

# %% [md]
# **Spark 4 y Spark Connect.** Hasta Spark 3 el notebook (el *driver*) corría dentro de la misma JVM que Spark: el cliente tenía acceso directo a `SparkContext` y a los RDD. Con **Spark Connect** (nuevo en Spark 3.4 y modo recomendado en Spark 4) el notebook pasa a ser un **cliente ligero** de un servidor Spark remoto: se conecta por gRPC con `SparkSession.builder.remote('sc://host:15002')`, y lo único que viaja son **planes lógicos de DataFrame/SQL** (en formato Protocol Buffers); el servidor los optimiza y ejecuta en el clúster y devuelve los resultados (en formato Arrow). Ventajas: el cliente es liviano (no necesita Java), varios usuarios comparten un clúster, se actualiza el servidor sin romper los notebooks y un fallo del cliente no tumba el driver.
#
# **Por qué en ese modo no hay RDD ni `SparkContext`:** un RDD se define con funciones Python arbitrarias (`lambda`) aplicadas sobre particiones y con acceso directo a la JVM, y el protocolo solo transporta planes de DataFrame/SQL, no funciones arbitrarias sobre particiones. Por eso, en Connect, `sdf.rdd`, `spark.sparkContext` y `sc.parallelize` no existen y el conteo MapReduce de arriba se escribe con `explode/split/groupBy` (que sí viaja como plan) o con UDF/`mapInPandas`. MLlib sí funciona en Connect para los algoritmos soportados.

# %% [code]
# --- Spark Connect (Spark 4): el notebook como CLIENTE de un clúster remoto. NO se ejecuta aquí. ---
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.remote('sc://mi-servidor-spark:15002').getOrCreate()  # gRPC; solo viajan planes DataFrame/SQL
# sdf = spark.read.parquet('s3a://mi-bucket/tweets/')       # el servidor (EMR) lee de S3, el notebook solo recibe el resultado
# sdf.groupBy('sentiment').count().show()                    # funciona: es un plan de DataFrame
# sdf.rdd.flatMap(...)                                       # ERROR en Connect: no hay RDD ni SparkContext en el cliente
# # Equivalente local sin clúster: SparkSession.builder.config('spark.api.mode', 'connect').master('local[*]').getOrCreate()
print("Spark Connect: snippet comentado (no ejecutado). Sesión clásica 'spark' sigue activa para la Fase 9:", spark.version)

# %% [code]
# Cifras clave a JSON (results/fase7.json)
resumen7 = {
    'spark_version': spark.version,
    'filas': {'corpus': int(len(df)), 'train': int(len(X_train)), 'val': int(len(X_val)), 'test': int(len(X_test))},
    'validacion_smoothing': tabla_val.to_dict('records'),
    'mejor_smoothing': mejor_smoothing,
    'mllib_test': {'f1_macro_sklearn': round(float(f1_spark), 4), 'accuracy': round(float(acc_spark), 4),
                   'f1_ponderado_mllib': round(float(f1_pond_spark), 4),
                   'tiempo_entrenamiento_s': round(tiempo_spark, 2), 'tiempo_prediccion_eval_s': round(t_pred_spark, 2)},
    'tiempo_sesion_spark_s': round(t_sesion, 1),
    'tiempos_spark_s': {'caracteristicas': round(t_car, 1), 'ajuste_smoothing': round(t_ajuste, 1), 'entrenamiento_final': round(tiempo_spark, 2),
                        'prediccion_eval': round(t_pred_spark, 2), 'total_punta_a_punta': round(t_total_spark, 1)},
    'sklearn_nb_mismo_algoritmo': {'f1_macro': round(float(f1_sk_nb), 4), 'accuracy': round(float(acc_sk_nb), 4),
                                   'entrenar_s': round(tiempo_sk_nb, 2), 'predecir_s': round(t_pred_sk_nb, 2)},
    'comparacion_tiempos': sk[['modelo', 'motor', 'f1_macro', 'accuracy', 'tiempo_s']].to_dict('records'),
    'mapreduce': {'tiempo_rdd_s': round(t_rdd, 2), 'tiempo_dataframe_s': round(t_df, 2),
                  'top15': [[w, int(c)] for w, c in top15_rdd], 'palabras_distintas': int(conteo_rdd.count())},
    'plan_explain': {'exchange_agregacion': int(n_exchange), 'exchange_sql': int(n_shuffles(plan_sql)),
                     'exchange_ventana': int(n_shuffles(plan_ventana))},
    'sql_por_emocion': res_sql.to_dict('records'),
    'top5_por_emocion': resumen_top5.to_dict(),
    'volumen': {'bytes_texto_por_tweet': round(bytes_medio, 1), 'mb_corpus_texto': round(mb_corpus, 2), 'mb_texto_original': round(mb_crudo, 2),
                'factor_ram_tfidf': round(float(factor_ram), 2), 'supuestos': {'bytes_registro': BYTES_REGISTRO, 'dias_historia': DIAS_HISTORIA,
                'ram_nodo_gb': RAM_NODO_GB, 'uso_max_ram': USO_MAX_RAM}, 'escenarios': tabla_esc.to_dict('records')},
}
with open('results/fase7.json', 'w', encoding='utf-8') as f:
    json.dump(resumen7, f, ensure_ascii=False, indent=2, default=str)
print("Guardado results/fase7.json")

# %% [conclusion]
# **Conclusión de negocio (Fase 7).**
# 1. **MLlib replica la clasificación, pero no la mejora.** Con la misma partición que sklearn (27 781 train / 5 953 val / 5 954 prueba), el Pipeline de Spark (tokenizar, stopwords, HashingTF 2^15, IDF, Naive Bayes con smoothing = 1.0 elegido en validación) obtiene en prueba F1 macro = 0.142 y accuracy = 0.236 (el F1 ponderado que reporta MLlib, 0.236, no es el macro: aquí se usa el macro calculado con sklearn sobre las predicciones). El mismo algoritmo en sklearn (TF-IDF + NB sin ajustar) da F1 macro = 0.0965 y accuracy = 0.292; la mejor comparación contra el mejor modelo de la Fase 4 está en la tabla `resultados`. La dificultad (13 clases solapadas) la fija el dato, no el motor.
# 2. **Spark local es MÁS lento para este volumen, y es lo esperable.** En esta corrida, sklearn entrenó el NB en 0.32 s y predijo en 0.05 s; Spark tardó 1.3 s solo en entrenar el Pipeline final y 15.7 s de punta a punta (arrancar la JVM ≈ 6.5 s, características ≈ 3.2 s, probar 3 valores de smoothing ≈ 3.8 s, entrenamiento final y predicción), unas 43 veces más. Los tiempos varían entre corridas con la carga de la máquina, pero el orden de magnitud no. Con ≈ 2.6 MB de texto no se amortiza el costo fijo de Spark (JVM, planificación, serialización).
# 3. **MapReduce y planes.** El conteo con RDD (flatMap → map → reduceByKey) y el de DataFrame dan exactamente el mismo top 15 (day 3 075, good 2 290, get 2 008, like 1 939, today 1 647…) sobre 29 425 palabras distintas; el RDD tardó ≈ 0.6 s y el DataFrame ≈ 2.0 s (el DataFrame paga la planificación de Catalyst y el explode en la JVM; en un volumen mayor Catalyst suele ganar). `explain()` muestra HashAggregate parcial → `Exchange hashpartitioning(sentiment)` → HashAggregate final: un solo shuffle para agrupar; la consulta SQL (tweets y longitud media por emoción) tiene 2 (agrupar y ordenar) y la ventana top-5 también 2 (agrupar y repartir por emoción para `row_number()`). Los cuellos de botella de un clúster son justo esos Exchange.
# 4. **Hallazgo de negocio con la ventana.** Las palabras dominantes separan emociones donde hay señal léxica clara (hate: «hate» 210, «sucks»; sadness: «sad» 385, «miss» 340; fun: «lol» 192, «fun» 142; love: «love» 861), pero en neutral, worry, relief o surprise dominan palabras genéricas («day», «get», «good», «like»). Eso anticipa el techo de rendimiento de los modelos bag-of-words.
# 5. **Recomendación con números.** El texto limpio mide ≈ 66 bytes por tweet; con un registro completo supuesto de 500 bytes y un factor medido de 2.0x para la matriz TF-IDF, un año de historia ocupa: 10 mil tweets/día = 1.8 GB (≈ 5.5 GB de RAM: **un solo nodo basta**, Spark no se justifica); 1 millón/día = 182.5 GB (≈ 550 GB de RAM: **no cabe en un nodo de 64 GB**; incluso con solo 30 días de historia, ≈ 45 GB, queda por encima del 50 % de RAM aceptado, por lo que ahí sí se justifica Spark en EMR con datos en S3 en Parquet); 100 millones/día = 18 250 GB al año (≈ 55 000 GB de RAM): Spark/EMR es obligatorio, además con almacenamiento en S3 y nodos task spot para abaratar. Regla práctica: Spark vale la pena cuando el dato supera cientos de GB o la ventana histórica no cabe en la RAM de una máquina; para este ejercicio (2.6 MB) basta sklearn. Los supuestos (500 bytes por registro, 64 GB de RAM, 50 % de uso, 365 días) están declarados en el código y son estimaciones, no mediciones; en cambio, los 66 bytes por tweet y el factor 2.0x sí salen de los datos.
# 6. **Spark Connect (Spark 4):** el notebook pasaría a ser un cliente que envía planes de DataFrame/SQL por gRPC a un clúster remoto; por eso el conteo con RDD de esta fase no correría en ese modo y habría que escribirlo con DataFrame (como la versión de la ventana). El snippet quedó comentado y no se ejecutó.
