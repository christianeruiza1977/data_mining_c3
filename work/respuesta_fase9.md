**Fase 9: respuesta a la pregunta de la guía** (BERTopic vs. LDA; LLM zero-shot y embeddings vs. el mejor modelo; PSI de drift; CSV vs. Parquet)

Todo se ejecutó con datos reales en un entorno con acceso a Hugging Face (no en Colab). Los tweets no salieron del equipo: solo se descargaron los pesos de los modelos.

**BERTopic vs. LDA** (misma muestra de 5 000 tweets). BERTopic encontró 25 temas (`min_topic_size=30`) contra 5 de LDA, pero dejó 47,2 % de los tweets sin tema (-1) contra 5,0 % de LDA. La pureza emocional sube de 0,229 a 0,296, con una emoción más común de 0,222 en la muestra: los temas son sobre todo de asunto, no de emoción. Ejemplos: tema 4 (`day, happy, mothers, mom, flowers`) concentra `love` 36 % y `happiness` 21 %; tema 5 (`hurts, pain, headache, sick, hospital`) concentra `worry` 51 %; tema 0 (`car, bus, london, traffic`) no tiene emoción dominante (`worry` 24 %, `neutral` 20 %).

**LLM zero-shot** (100 tweets de prueba). F1 macro de 0,030 contra 0,176 del SVM en los mismos 100 tweets; 0,33 s por tweet contra 0,04 ms; 87 % de respuestas fuera de la lista de emociones. Es un modelo pequeño (Qwen2.5-0.5B-Instruct, 0,5B) en CPU: el resultado no representa a un LLM grande.

**Embeddings** (MiniLM + LinearSVC). F1 macro 0,159 contra 0,198 de TF-IDF + SVM; accuracy 0,356 contra 0,270 (la más alta de todos los modelos). Codificar tardó 94 s y entrenar 49 s, contra 1,7 s del SVM. TF-IDF + SVM ganó en F1 macro. El clasificador sobre embeddings no usó `class_weight='balanced'`, así que parte de la brecha podría deberse a eso (no se probó).

**Drift (PSI).** Ordenando por `tweet_id` y entrenando con el 70 % antiguo, el PSI de las predicciones es 0,350 (mayor que 0,25: cambio fuerte). El modelo predice más «worry» (21,2 % a 37,1 %) y «sadness» (6,8 % a 14,8 %) y menos «happiness» (20,0 % a 8,0 %). El F1 macro baja de 0,166 a 0,142.

**Parquet.** El corpus completo pesa 3,39 MB en CSV y 2,25 MB en Parquet con snappy (1,51 veces). La tabla de resultados (6 filas) es tan pequeña que Parquet pesa más (3,8 KB contra 0,43 KB) por sus metadatos. Spark leyó el Parquet sin problema.

**Cierre (costo, latencia, LOPDP).** TF-IDF + SVM entrena en 1,7 s, predice 1 000 tweets en 0,019 s y no envía datos fuera. Un LLM por API cobra tokens × tarifa (≈ 69 000 tokens de entrada por 1 000 tweets con este prompt) y mandar mensajes de clientes a un tercero es tratamiento y posible transferencia internacional de datos personales (LOPDP). Embeddings o LLM se justifican si importa el significado, hay pocos datos etiquetados o el volumen es pequeño, y el dato puede salir o el modelo corre en local; con esta evidencia se mantiene TF-IDF + SVM y se monitorea el drift con PSI. Fuentes: *¿LLM o embeddings? El dilema del costo en la minería de textos* (lectura de la semana); Grootendorst (2022), BERTopic, arXiv:2203.05794; documentación de spacy-llm.

| Medida | Valor |
|---|---|
| BERTopic vs. LDA: temas | 25 vs. 5 |
| Tweets sin tema | 47,2 % vs. 5,0 % |
| Pureza de emoción | 0,296 vs. 0,229 |
| LLM zero-shot (100 tweets): F1 macro | 0,030 (SVM en los mismos 100: 0,176) |
| LLM: tiempo por tweet / fuera de la lista | 0,33 s (SVM 0,04 ms) / 87 % |
| Embeddings: F1 macro / accuracy | 0,159 / 0,356 (TF-IDF + SVM: 0,198 / 0,270) |
| Embeddings: codificar + entrenar | 94 s + 49 s (SVM: 1,7 s) |
| PSI de predicciones (referencia vs. reciente) | 0,350 (cambio fuerte) |
| F1 macro referencia / reciente | 0,166 / 0,142 |
| Corpus CSV / Parquet snappy | 3,39 MB / 2,25 MB (1,51 veces) |

Figuras para la guía: `figuras/fase9_bertopic_tema_emocion.png`, `figuras/fase9_drift_distribuciones.png`, `figuras/fase9_tamano_csv_parquet.png`.
