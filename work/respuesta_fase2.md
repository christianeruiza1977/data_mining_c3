**Pregunta de la guía:** ¿Qué categorías gramaticales, entidades y dependencias distinguen a las emociones? ¿Aportan para clasificar?

Se analizó con spaCy una muestra estratificada de 2 489 tweets (200 por emoción; `anger` 110 y `boredom` 179). Las diferencias en categorías gramaticales son estadísticamente detectables pero pequeñas: los adjetivos van de 4,59 % (`neutral`) a 7,16 % (`surprise`) y los verbos de 11,37 % (`love`) a 14,40 % (`hate`). Las interjecciones no difieren de forma significativa (p = 0,66). En dependencias, la negación pesa 2,39 % de los tokens en `worry` y 0,62 % en `happiness`, y los modificadores adjetivales son más frecuentes en `happiness` (4,34 %) y `love` (4,22 %).

Las entidades aportan poco: las más frecuentes son marcas de tiempo (today 98, tonight 48, tomorrow 33) y el NER se equivoca con jerga (`lol` como ORG, `awww` como PERSON).

Para clasificar, la señal es débil. Una regresión logística solo con proporciones POS obtiene F1 macro 0,095, apenas sobre el azar (0,076) y muy por debajo de una bolsa de palabras TF-IDF (0,184). Con dependencias y entidades sube a 0,098. Sirven como complemento (sobre todo la negación), no como base.

| Modelo (CV 5 particiones) | F1 macro |
|---|---|
| Clase mayoritaria | 0,011 |
| Azar uniforme | 0,076 |
| LogReg solo POS | 0,095 |
| LogReg POS + dependencias + entidades | 0,098 |
| TF-IDF (referencia) | 0,184 |

Figuras para la guía: `figuras/fase2_pos_por_emocion.png`, `figuras/fase2_clasificacion_pos.png`, `figuras/fase2_dependencias_1.svg` (y 2, 3).
