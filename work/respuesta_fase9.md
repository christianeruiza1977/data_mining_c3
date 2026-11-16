**Fase 9: respuesta a la pregunta de la guía** (BERTopic vs. LDA; LLM zero-shot y embeddings vs. el mejor modelo; PSI de drift; CSV vs. Parquet)

**Pendiente de ejecutar en Colab (a, b y c).** Hugging Face está bloqueado en este entorno (error 403). Por eso BERTopic (con `min_topic_size=30` sobre los 5 000 tweets de la Fase 3), el LLM zero-shot (100 tweets de prueba con `preguntar_llm()`) y los embeddings `all-MiniLM-L6-v2` están escritos completos pero se saltan con el aviso «No ejecutado en este entorno». No se reportan cifras de estos tres bloques. Como expectativa (no resultado): BERTopic dejará tweets en el tema -1, mientras LDA asigna siempre un tema; el LLM será mucho más lento que el SVM y puede responder fuera de la lista; los embeddings podrían acercarse o superar algo a TF-IDF.

**Drift (ejecutado).** Se ordenó por `tweet_id`, se entrenó con el 70 % antiguo (TF-IDF + LinearSVC) y se comparó la distribución de clases predichas entre la referencia y el 30 % reciente. El PSI es 0.350, mayor que 0.25 (cambio fuerte): el modelo predice más «worry» (21.2 % a 37.1 %) y «sadness» (6.8 % a 14.8 %) y menos «happiness» (20.0 % a 8.0 %). El F1 macro baja de 0.166 a 0.142.

**Parquet (ejecutado).** El corpus completo pesa 3.39 MB en CSV y 2.25 MB en Parquet con snappy (razón 1.51x). La tabla de resultados es tan pequeña que Parquet pesa más que el CSV por sus metadatos. Spark leyó el Parquet sin problema.

**Cierre.** Un LLM o embeddings se justifican si importa el significado más allá de las palabras, no hay datos etiquetados o el volumen es pequeño, y si el dato puede salir de la organización o el modelo corre en local (LOPDP). Con volumen alto, latencia baja y datos personales, TF-IDF + SVM sigue siendo la mejor opción. Fuentes: *¿LLM o embeddings? El dilema del costo en la minería de textos* (lectura de la semana); Grootendorst (2022), BERTopic, arXiv:2203.05794; documentación de spacy-llm.

| Medida (ejecutada) | Valor |
|---|---|
| PSI de predicciones (referencia vs. reciente) | 0.350 (cambio fuerte) |
| F1 macro referencia / reciente | 0.166 / 0.142 |
| Corpus CSV / Parquet snappy | 3.39 MB / 2.25 MB (1.51x) |
| BERTopic, LLM zero-shot, embeddings | pendiente (Colab) |

Figuras para la guía: `figuras/fase9_drift_distribuciones.png`, `figuras/fase9_tamano_csv_parquet.png` (y `figuras/fase9_bertopic_tema_emocion.png`, que solo se genera en Colab).
