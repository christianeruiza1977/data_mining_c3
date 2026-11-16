# Respuesta Fase 4: Naive Bayes y SVM

Con la partición estratificada 70/15/15 (27 781 / 5 953 / 5 954 tweets, semilla 42) ajusté dos pipelines TF-IDF (unigramas y bigramas): Naive Bayes multinomial (alpha=0.03) y SVM lineal (C=0.1, class_weight balanceado). Los hiperparámetros se eligieron por F1 macro en validación y la prueba se usó una sola vez. El SVM es mejor en F1 macro (0.198 contra 0.144); Naive Bayes tiene más accuracy (0.299 contra 0.270) porque acierta sobre todo las clases grandes (neutral y worry) y casi ignora las raras (F1 de 0.00 en empty y anger). Ambos superan el azar (1/13 = 0.077) y la clase mayoritaria (siempre «neutral»: accuracy 0.215, F1 macro 0.027), pero siguen siendo modelos débiles.

Las emociones que más se confunden son neutral y worry entre sí (NB: 354 y 335 tweets; SVM: 144 y 138), sadness predicha como worry (NB 314; SVM 128) y happiness predicha como neutral (NB 238) o como love (SVM 129). Tiene sentido: worry y sadness comparten vocabulario de malestar, happiness y love de afecto, y neutral no tiene señal léxica propia.

| Modelo | F1 macro | Accuracy | Tiempo de ajuste (s) |
|---|---|---|---|
| Naive Bayes | 0.144 | 0.299 | 0.7 |
| SVM lineal | 0.198 | 0.270 | 1.7 |

Figuras para la guía: `figuras/fase4_matriz_nb.png`, `figuras/fase4_matriz_svm.png`, `figuras/fase4_f1_por_clase.png`.
