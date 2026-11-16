# Especificación común (la mantiene el agente orquestador)

Proyecto: Ejercicio Práctico Semana 3 (Data Mining II, UIDE). Grupo 9. Dataset `tweet_emotions.csv` (40 000 tweets, 13 emociones).
Notebook original (plantilla): /root/.claude/uploads/619f4b5e-b391-58c8-894a-fd78c5af312b/4cb1c6fd-Ejercicio_Practico_Semana3.ipynb
Guía DOCX original: .../6023d6de-Ejercicio_Practico_Semana3_Guia.docx (ya leída por el orquestador).

## Entorno
- Python: /home/user/venv/bin/python (spark 4.0.1, spacy 3.8 + en_core_web_sm, sklearn 1.9, tensorflow 2.21/keras, pyarrow, matplotlib, bertopic). Java 21.
- Ejecutar SIEMPRE desde la raíz del repo: /home/user/data_mining_c3 (rutas relativas `data/`, `results/`, `figuras/`).
- Sin acceso a Hugging Face ni Kaggle (403). Solo corre aquí lo que no los necesita. El CSV está en data/tweet_emotions.csv.
- Máquina: 4 CPU, 15 GB RAM. Otros agentes corren en paralelo: durante el desarrollo usa subconjuntos/pocas épocas y limita hilos (p. ej. spark local[2], tf.config.threading). La corrida completa final la hace el orquestador.

## Contrato de código compartido
`work/common.py` define: SEMILLA=42, df (limpio; columnas tweet_id, sentiment, content, texto_limpio), ETIQUETAS (13), X_train/y_train, X_val/y_val, X_test/y_test (70/15/15 estratificado), muestra_temas (5 000 tweets).
En el notebook final este código estará repartido: carga+limpieza en Fase 1, partición en Fase 4, muestra_temas en Fase 3. Tu archivo de fase puede asumir que `common.py` ya se ejecutó (en tu prueba haz `exec(open('work/common.py').read())`). NO redefinas esas variables; no cambies common.py (si crees que hay un error, dilo en tu informe).
Variables que se pasan entre fases dentro del notebook (respétalas exactamente):
- Fase 3 deja: `K_LDA` (int), `palabras_lda` (lista de listas: 10 palabras por tema), `lda`, `vec_lda`, `tema_dom` (Series con el tema dominante de cada tweet de muestra_temas, índice = índice de df).
- Fase 4 deja: `resultados` (DataFrame con columnas modelo, fase, f1_macro, accuracy, tiempo_s), `mejor_nombre` ('SVM' o 'NB'), `mejor_modelo` (Pipeline sklearn ya ajustado), `pred_test_mejor` (predicciones sobre X_test).
- Fase 5 añade a `resultados` la fila de la RNN.
- Fase 7 añade a `resultados` la fila de Spark MLlib; deja `spark` activo.
- Fase 9 guarda `resultados` completo (+ filas de LLM/embeddings solo si se ejecutaron) en Parquet.
Para probar tu fase sola, crea stubs mínimos de lo que te falte (en tu script de prueba, NO en el archivo de fase).

## Formato del archivo de fase `work/faseN.py`
Una secuencia de celdas separadas por líneas marcador exactas:
- `# %% [que]`        → texto (comentado con `# `) que reemplaza «_completar antes del código_» en la celda de encabezado de la fase (1–3 frases: qué se analiza y con qué).
- `# %% [code]`       → celda de código (Python normal).
- `# %% [md]`         → celda markdown (líneas comentadas con `# `), p. ej. la línea «Qué se analiza» ANTES de cada gráfico.
- `# %% [conclusion]` → conclusión de negocio (markdown, comentada con `# `), al final de la fase, con CIFRAS REALES de tu corrida.
Reglas de estilo:
- Código en español, estilo analista junior: simple, legible, funciones cortas, variables con nombres claros, SIN sobreingeniería. Comenta las líneas relevantes (qué hace y por qué) con comentarios cortos en español.
- Antes de cada gráfico una celda [md] «Qué se analiza: …»; después del bloque, la conclusión de negocio.
- Gráficos: matplotlib con el estilo del notebook (variables TINTA, ACENTO, GRIS ya definidas: TINTA="#14323F", ACENTO="#6E5BD6", GRIS="#B7C3C9"; sin rejilla, sin marco superior/derecho, ACENTO solo para el dato clave). Guarda cada figura también en figuras/faseN_nombre.png (dpi 150) y muéstrala con plt.show().
- Cada cifra citada en el texto debe provenir de tu ejecución. NO inventes números. Si algo no se pudo ejecutar, dilo explícitamente.
- Reproducibilidad: usa SEMILLA (42) en todo lo aleatorio.
- Resultados: guarda un JSON con las cifras clave en results/faseN.json (desde el propio código del notebook).
- Además escribe `work/respuesta_faseN.md`: la respuesta a la «Pregunta de la guía» de la fase, en lenguaje claro y sencillo de maestrante de data science, 120–220 palabras + (si aplica) una tabla markdown pequeña con las cifras, y la lista de figuras (figuras/…png) que deben ir en la guía. Debe ser coherente con las cifras del notebook.
- Entrega: tu informe final debe listar archivos creados, cifras clave, tiempo de ejecución de tu archivo completo con datos completos, problemas/desviaciones.
- NO toques archivos que no sean los tuyos (work/faseN.py, work/respuesta_faseN.md, results/faseN*.json, figuras/faseN_*). NO hagas commit/push.
- El archivo de fase se ejecutará secuencialmente dentro de un solo notebook (kernel único): evita imports que ya hace el notebook solo si es seguro, pero repetir un import es válido.
