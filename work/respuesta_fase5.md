# Respuesta Fase 5: RNN frente a NB y SVM

Entrené en Keras una red TextVectorization (20 000 tokens, 40 palabras), Embedding de 64, GRU bidireccional de 64, Dropout y softmax de 13 clases, con la misma partición, early stopping sobre `val_loss` (paciencia 2) y pesos de clase balanceados. Paró en la época 5 y restauró los pesos de la época 3, porque después la pérdida de validación sube (sobreajuste). En prueba obtuvo F1 macro de 0.149 y accuracy de 0.176: no supera al SVM (0.198 y 0.270) y queda al nivel de Naive Bayes en F1 macro (0.144), con menor accuracy (0.299 en NB). Además es mucho más costosa: 95 s de entrenamiento contra 1.7 s del SVM y 0.7 s de NB, y 1.33 millones de parámetros contra 0.50 millones de coeficientes.

Para el desbalance usé `class_weight='balanced'`: neutral y worry pesan 0.36, boredom 17.1 y anger 27.8. Esto fuerza a la red a atender las clases raras, pero baja la accuracy porque sacrifica las grandes (F1 de neutral 0.10 contra 0.31 del SVM). La alternativa sería muestrear (sobremuestrear las raras o submuestrear neutral y worry), que no probé. Con unos 28 mil tweets cortos y etiquetas ruidosas, el modelo lineal es la mejor relación costo-beneficio.

| Modelo | F1 macro | Accuracy | Tiempo (s) | Parámetros |
|---|---|---|---|---|
| NB | 0.144 | 0.299 | 0.7 | 502 268 |
| SVM | 0.198 | 0.270 | 1.7 | 502 268 |
| RNN (BiGRU) | 0.149 | 0.176 | 95.4 | 1 331 597 |

Figuras para la guía: `figuras/fase5_curva_entrenamiento.png`, `figuras/fase5_f1_por_clase_rnn_svm.png`.
