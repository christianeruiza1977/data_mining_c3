# %% [que]
# Se responde el desafío con la evidencia de las fases 1 a 7 y 9 y se arma la tabla de evaluación de alternativas (spaCy, LDA, NB, SVM, RNN y scikit-learn frente a Spark), con ventajas, limitaciones, aplicabilidad y fuente.

# %% [md]
# **Qué se analiza:** primero se juntan en una sola tabla las cifras que sostienen las decisiones (modelos, tiempos y experimento de clases). Todo sale de las variables y archivos de las fases anteriores, no se escribe a mano.

# %% [code]
import json
# Cifras del experimento de clases (fase 6): promedio de 3 semillas por modelo y número de clases
f6 = json.load(open("results/fase6.json"))
tabla_k = pd.DataFrame(f6["resumen"])
print("Modelos con la partición de la fase 4 (13 clases, prueba):")
display(resultados.round(4))
print("\nExperimento de clases (fase 6), resumen:")
display(tabla_k.round(3))
print("\nCriterio de granularidad fijado en la fase 6:", f6["criterio"], "| k recomendado:", f6["k_recomendado"])

# %% [md]
# **Qué se analiza:** la tabla de evaluación de alternativas pedida por la guía. Cada fila indica ventajas, limitaciones y en qué escenario se usaría la técnica.

# %% [code]
# Tabla de alternativas (las fuentes son lecturas del Tema 3 y documentación oficial)
alternativas = pd.DataFrame([
    ["spaCy (POS, NER, dependencias)",
     "Pipeline listo, rápido con nlp.pipe; da lemas, categorías gramaticales y entidades legibles.",
     "En tweets el NER falla con jerga y emoticones; las proporciones POS casi no separan emociones (fase 2).",
     "Extraer entidades y limpiar texto en un pipeline profesional; no como único insumo para detectar emoción.",
     "Doc. spaCy: spacy.io/usage/linguistic-features; Tema 3, L1 Minería de textos"],
    ["LDA (scikit-learn)",
     "No supervisado, barato e interpretable con palabras clave; sirve para explorar el corpus.",
     "Hay que fijar K; con textos cortos la perplejidad de validación no se estabiliza con K; los temas son mezclas poco nítidas.",
     "Exploración inicial de un corpus largo y sin etiquetas (reclamos, informes).",
     "Doc. scikit-learn (LatentDirichletAllocation); Tema 3, L1"],
    ["Naive Bayes multinomial",
     "Muy rápido (menos de 1 s), simple y buena línea base; accuracy más alta en 13 clases.",
     "Supone independencia entre palabras y favorece a las clases grandes; F1 macro bajo (fase 4).",
     "Prototipo y línea base cuando se necesita una respuesta casi inmediata.",
     "Doc. scikit-learn (naive_bayes); Tema 3, L1"],
    ["SVM lineal (TF-IDF)",
     "Mejor F1 macro de NB y SVM en la fase 4 y en todos los k de la fase 6; entrena en segundos; barato de operar.",
     "No entiende el orden ni el contexto; sigue fallando en las clases raras (anger, boredom).",
     "Opción por defecto para clasificar texto con muchos datos etiquetados y poco presupuesto.",
     "Doc. scikit-learn (svm, LinearSVC); Tema 3, L1"],
    ["RNN (BiGRU en Keras)",
     "Captura el orden de las palabras y puede mejorar con más datos y embeddings preentrenados.",
     "Unas 34 veces más lenta que el SVM, con más parámetros, y no lo superó en este dato (fase 5).",
     "Cuando hay mucho texto etiquetado, GPU y el orden de las palabras importa de verdad.",
     "Doc. Keras (GRU, TextVectorization); Tema 3, L1"],
    ["scikit-learn (un solo nodo)",
     "API simple; todo el flujo cabe en memoria y corre en segundos para 40 000 tweets.",
     "Limitado por la RAM de la máquina; no escala solo a cientos de GB.",
     "Volúmenes de hasta unos pocos GB (nuestro caso hoy).",
     "Doc. scikit-learn; Tema 3, L2 Big Data"],
    ["Spark MLlib (PySpark)",
     "Escala a muchos nodos, lee de HDFS/S3 y guarda en Parquet; el mismo código sirve de 1 a N máquinas.",
     "En un solo equipo fue mucho más lento que sklearn por la sobrecarga de la JVM (fase 7); más difícil de operar.",
     "Cuando el histórico no cabe en una máquina (cientos de GB) o ya existe un clúster EMR/S3.",
     "Doc. Spark (ml-pipeline, spark-connect-overview); Tema 3, L2 Big Data"],
    ["BERTopic / embeddings / LLM (fase 9)",
     "Entienden el significado y no dependen de que las palabras coincidan; BERTopic da temas más específicos que LDA (25 frente a 5) y los embeddings lograron la accuracy más alta (0,356).",
     "Más costo y latencia, y enviar texto a un servicio externo implica riesgo con la LOPDP. Medido en la fase 9: BERTopic dejó 47,2 % de los tweets sin tema (LDA, 5,0 %); embeddings MiniLM + LinearSVC lograron F1 macro 0,159 frente a 0,198 de TF-IDF + SVM (94 s de codificación + 49 s de entrenamiento frente a 1,7 s); un LLM zero-shot de 0,5B en CPU obtuvo F1 macro 0,030 en 100 tweets, 0,33 s por tweet y 87 % de respuestas fuera de la lista (no representa a un LLM grande).",
     "Si se justifica por calidad, con datos que puedan salir de la empresa o con modelos locales.",
     "Grootendorst (2022) BERTopic; lectura de la semana ¿LLM o embeddings?; doc. spacy-llm"],
], columns=["tecnica", "ventajas", "limitaciones", "aplicabilidad", "fuente"])
pd.set_option("display.max_colwidth", None)
display(alternativas)
alternativas.to_csv("results/fase8_alternativas.csv", index=False)

# %% [conclusion]
# **Respuesta al desafío (con la evidencia de este notebook).**
#
# 1. **Granularidad:** no recomendamos usar las 13 emociones con confianza. Con 13 clases el mejor F1 macro en prueba fue 0,198 (SVM; azar = 0,077). En el experimento de la fase 6 el SVM promedió 0,543 con 3 clases, 0,393 con 5, 0,316 con 7, 0,258 con 9 y 0,229 con 11. Con el criterio fijado de antemano (F1 macro de al menos 0,30 y al menos el doble del azar), el máximo es 7 clases. Recomendamos 5 a 7 categorías (por ejemplo neutral, worry, sadness, happiness, love y hate, y agrupar el resto). El experimento sortea clases y no prueba agrupaciones, así que esta es una hipótesis que hay que validar.
# 2. **Modelo:** SVM lineal con TF-IDF. Tuvo el mejor F1 macro (0,198 frente a 0,144 de NB y 0,149 de la RNN) y entrena en unos 2 segundos (1,7 s), frente a unos 57 s de la RNN. NB es el más rápido (menos de 1 s) y tiene la accuracy más alta entre los modelos TF-IDF, pero no detecta bien las clases pequeñas. La RNN no justifica su costo con este dato. La fase 9 confirma la elección con modelos más modernos: los embeddings MiniLM con un clasificador lineal obtuvieron F1 macro de 0,159 (accuracy 0,356, la más alta de todas, pero con 94 s de codificación y 49 s de entrenamiento frente a 1,7 s del SVM) y un LLM zero-shot pequeño (Qwen2.5-0.5B en CPU, no representa a un LLM grande) obtuvo 0,030 en 100 tweets, con 0,33 s por tweet y 87 % de respuestas fuera de la lista. En F1 macro ganó TF-IDF + SVM.
# 3. **Spark:** hoy no se justifica. Con 40 000 tweets (3,8 MB en CSV) Spark local fue mucho más lento que scikit-learn por la sobrecarga de arranque (15,7 s frente a 0,37 s del flujo completo del NB en sklearn, unas 43 veces). En la estimación de volumen de la fase 7, un millón de tweets al día suma unos 182,5 GB al año (con el supuesto de 500 bytes por registro) y sí superaría una sola máquina al conservar el histórico; con 10 mil al día basta un nodo. Recomendamos empezar con scikit-learn y migrar a Spark sobre EMR con datos en S3 (Parquet) cuando el histórico ya no quepa en memoria.
# 4. **Limitación seria del dataset:** las etiquetas son ruidosas y muy desbalanceadas (anger tiene 110 tweets y neutral 8 638; la razón es 78,5), la clase `empty` mezcla tweets con emoción evidente y los textos son tweets en inglés de una época concreta. No representan los mensajes en español de clientes en Ecuador. Además, el PSI de 0,35 de la fase 9 (tramo antiguo frente a tramo reciente ordenado por `tweet_id`; cambio fuerte, con el F1 macro bajando de 0,166 a 0,142) muestra que el modelo ya pierde estabilidad entre tramos de tiempo.
# 5. **Implicación ética:** analizar la emoción de personas es un tratamiento de datos personales (LOPDP). No debe usarse para decisiones automáticas con efecto sobre el cliente sin una persona que revise, y enviar mensajes a una API externa exige base legal y minimización (en la fase 9 los modelos de Hugging Face corrieron en local: solo se descargaron los pesos y ningún tweet salió del equipo). Con un F1 macro cercano a 0,2, el modelo se equivocaría en la mayoría de las clases raras, por lo que debe servir solo para priorizar la atención y no para etiquetar a las personas.
# 6. **Temas (BERTopic frente a LDA, fase 9):** para explorar de qué hablan los tweets, BERTopic dio temas más específicos que LDA (25 frente a 5; por ejemplo dolor y salud, transporte, Día de la Madre), pero dejó 47,2 % de los tweets sin tema (LDA, 5,0 %). Sirve para explorar el corpus, no para predecir la emoción (pureza emocional de 0,296 frente a 0,229 de LDA, con una emoción más común de 0,222).
