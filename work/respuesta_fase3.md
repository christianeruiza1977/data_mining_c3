**Pregunta de la guía:** Número de temas elegido (criterio), palabras clave de 3 temas y su cruce con emociones.

Se aplicó LDA sobre los 5 000 tweets de `muestra_temas` (4 748 con palabras útiles, vocabulario de 1 038 palabras). Se eligió K = 5 con este criterio: menor perplejidad en un 20 % de validación, confirmada por la coherencia NPMI de los temas. Ambas medidas empeoran en general al subir K (perplejidad de 2 843 con K = 5 a 10 703 con K = 20; coherencia de -0,193 a -0,401), porque los tweets son tan cortos que los temas extra se sobreajustan. Ningún K mayor mejora las medidas, así que K = 5 es una visión amplia.

Tres temas: Tema 4 `day, happy, fun, mothers, mother, mom` (celebraciones, Día de la Madre); Tema 3 `work, miss, need, wish, feel, feeling` (trabajo, añoranza y sentimientos); Tema 0 `home, today, night, working, going, house` (rutina diaria).

Cruce: el Tema 4 reúne 31,4 % de `love`, 25,2 % de `happiness` y 24,5 % de `fun`, frente a 10,2 % de `hate`. El Tema 3 reúne 24,4 % de `sadness` y de `enthusiasm`. Las demás celdas están cerca del 20 % esperado al azar, así que los temas describen de qué se habla, no la emoción.

| Tema | Emoción más asociada | % de la emoción en el tema |
|---|---|---|
| 4 (celebraciones) | love | 31,4 |
| 4 (celebraciones) | happiness | 25,2 |
| 3 (trabajo, añoranza) | sadness | 24,4 |

Figuras para la guía: `figuras/fase3_perplejidad.png`, `figuras/fase3_temas_vs_emociones.png`.
