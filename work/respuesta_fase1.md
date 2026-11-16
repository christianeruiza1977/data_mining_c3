**Pregunta de la guía:** Duplicados, longitud media y distribución de las 13 emociones. ¿Qué significa «empty» y qué hicieron con ella?

El corpus tiene 40 000 tweets, sin valores nulos ni `tweet_id` repetidos. Sí hay 173 textos duplicados y 62 textos (207 filas) que aparecen con emociones contradictorias. Se eliminaron las 207 filas contradictorias (no hay una respuesta correcta) y se dejó una sola copia de los demás duplicados (28 filas). Al limpiar menciones, URLs y entidades HTML, otros 77 tweets quedaron vacíos y se descartaron. El corpus final tiene 39 688 tweets (0,78 % menos).

El tweet medio mide 73,4 caracteres y 13,2 palabras (mediana 69 y 12; máximo 167 caracteres).

La distribución está muy desbalanceada: `neutral` (8 638) y `worry` (8 459) suman 42,7 %, mientras que `anger` tiene 110 y `boredom` 179. La razón mayor/menor es de 78,5 veces, así que se evalúa con F1 macro, no con exactitud.

`empty` (827 tweets, 2,1 %) no es un dato faltante: todos sus textos tienen contenido. Es la etiqueta de tweets en los que el anotador no identificó una emoción, y es ruidosa (algunos expresan aburrimiento o enfado). Se conservó como la 13.ª clase, porque la guía pide una matriz de 13 clases, con la advertencia de que será difícil de predecir.

| Indicador | Valor |
|---|---|
| Tweets crudos / limpios | 40 000 / 39 688 |
| Textos con etiquetas contradictorias | 62 (207 filas) |
| Longitud media | 73,4 caracteres, 13,2 palabras |
| Razón mayor/menor clase | 78,5 |
| Tweets con menciones / HTML / URLs | 47,4 % / 5,8 % / 4,8 % |

Figuras para la guía: `figuras/fase1_longitud.png`, `figuras/fase1_distribucion_emociones.png`.
