# Respuesta Fase 6: número de clases

Repetí NB, SVM y una RNN reducida (embedding 32, GRU de 32, máx. 4 épocas) con k = 3, 5, 7, 9 y 11 emociones sorteadas con `RandomState(semilla)` para las semillas 42, 43 y 44, evaluando en prueba una vez por combinación. El F1 macro baja al crecer k en los tres modelos, y el SVM es el mejor en todos los k. Promedio de 3 semillas del SVM: 0.543 (k=3), 0.393 (5), 0.316 (7), 0.258 (9) y 0.229 (11). La línea de azar 1/k baja también (0.333 a 0.091), así que la ganancia relativa del SVM sube de 1.6 a 2.5 veces el azar, pero la ganancia absoluta cae. Con pocas clases hay mucha dispersión entre semillas (desviación 0.0945 en k=3) porque depende de qué emociones salgan. Accuracy y F1 macro divergen cuando salen clases muy desiguales: NB con semilla 42 y k=7 (anger=16, boredom=27 junto a neutral=1280) tiene accuracy 0.492 y F1 macro 0.205.

Granularidad recomendada: con el criterio F1 macro mínimo de 0.30 y al menos 2 veces el azar, el mayor k que se cumple es 7. Recomiendo 5 a 7 categorías: conservar neutral, worry, sadness, happiness, love y hate, y agrupar las raras o solapadas (fun, enthusiasm y relief con happiness; anger y boredom con hate o sadness; empty y surprise en «otras»). Es una hipótesis: el experimento sortea clases, no prueba agrupaciones.

| k | NB | SVM | RNN | Azar 1/k |
|---|---|---|---|---|
| 3 | 0.480 | 0.543 | 0.494 | 0.333 |
| 5 | 0.304 | 0.393 | 0.329 | 0.200 |
| 7 | 0.237 | 0.316 | 0.259 | 0.143 |
| 9 | 0.190 | 0.258 | 0.194 | 0.111 |
| 11 | 0.169 | 0.229 | 0.169 | 0.091 |

Figuras para la guía: `figuras/fase6_f1_vs_k.png`, `figuras/fase6_accuracy_vs_f1.png`.
