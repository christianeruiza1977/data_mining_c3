**Fase 7: respuesta a la pregunta de la guía** (plan de `explain()`, Spark SQL con ventana, MapReduce y despliegue en EMR con S3)

Se replicó la clasificación con un Pipeline de Spark MLlib (RegexTokenizer, StopWordsRemover, HashingTF, IDF y Naive Bayes) sobre la misma partición 70/15/15 de sklearn. Con `smoothing = 1.0` (elegido en validación) se obtuvo F1 macro 0.142 y accuracy 0.236 en prueba. El F1 que da MLlib es ponderado; el macro se calculó con sklearn sobre las predicciones.

El `explain()` de una agregación por emoción muestra HashAggregate parcial, luego `Exchange hashpartitioning(sentiment)` y HashAggregate final. Ese Exchange es el shuffle, igual que en MapReduce. La consulta Spark SQL (`createOrReplaceTempView`) da los mismos conteos que pandas. La ventana `row_number() over (partition by sentiment order by conteo desc)` entrega el top-5 de palabras por emoción (por ejemplo, hate: «hate», «like», «work»; sadness: «sad», «day», «miss»). El conteo con RDD (`flatMap`, `map`, `reduceByKey`) y el de DataFrame coinciden en el top 15 (day 3 075, good 2 290, get 2 008).

En AWS, los datos irían a S3 (`s3a://`, en lugar de HDFS con bloques de 128 MB y réplica 3). El trabajo correría con `spark-submit` en EMR (nodo maestro, nodos core y task spot, en lugar de YARN propio) y guardaría Parquet en S3. Con Spark Connect el notebook sería solo un cliente gRPC (`remote('sc://host:15002')`) y no habría RDD.

Spark local fue más lento: 17.0 s de punta a punta contra 0.45 s del NB de sklearn (0.36 s de entrenamiento y 0.09 s de predicción). Con 2.6 MB de texto no se justifica un clúster. Un año a 10 mil tweets/día (1.8 GB) cabe en un nodo; a 1 millón/día (182.5 GB) ya no, y a 100 millones/día es obligatorio.

| Medida | Valor |
|---|---|
| F1 macro Spark MLlib (prueba) | 0.142 |
| Accuracy Spark MLlib (prueba) | 0.236 |
| Entrenar sklearn NB / Pipeline Spark | 0.36 s / 1.5 s |
| Spark de punta a punta (local) | 17.0 s |
| Shuffles (Exchange): agregación / SQL / ventana | 1 / 2 / 2 |
| Un año a 10 mil / 1 millón / 100 millones al día | 1.8 / 182.5 / 18 250 GB |

Figuras para la guía: `figuras/fase7_top_palabras_mapreduce.png`, `figuras/fase7_tiempos_sklearn_vs_spark.png`, `figuras/fase7_volumen_un_nodo_vs_cluster.png`.
