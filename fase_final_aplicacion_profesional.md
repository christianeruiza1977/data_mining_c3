# Aplicación profesional: Unidad de Afiliación y Control Técnico, IESS

**Textos que se generan.** En mi unidad se producen correos y oficios de empleadores y afiliados, reclamos y solicitudes de corrección de historia laboral, observaciones de los avisos de entrada y salida, informes de control técnico y mensajes recibidos por chat y redes sociales. Están en español de Ecuador, con regionalismos y abreviaturas.

**Información que extraería.** Primero, el tema de cada caso: mora patronal, afiliación voluntaria, aportes no registrados o cambio de datos. Segundo, la urgencia o la emoción del mensaje, para priorizar la atención. Tercero, las entidades: empresa, provincia, fechas y períodos de aportación.

**Técnica.** Usaría spaCy en español para la limpieza y el reconocimiento de entidades. Para clasificar tema y prioridad, TF-IDF con SVM lineal, porque en el ejercicio fue el modelo más barato y mejor que la RNN. Para explorar reclamos nuevos sin etiqueta, LDA o BERTopic. Habría que etiquetar mensajes propios en español: los tweets en inglés no sirven para producción.

**Procesamiento distribuido.** Para el trabajo diario bastaría un servidor. Spark con Parquet se justificaría solo para analizar el histórico nacional de varios años, que ya no cabe en memoria.

**Implicación ética.** Estos textos contienen datos personales y a veces sensibles, como cédula, salud o situación laboral. Según la LOPDP, debo minimizarlos, anonimizarlos y no enviarlos a servicios externos. El modelo solo debe ayudar a priorizar: nunca puede negar o demorar un derecho del afiliado sin que una persona revise el caso.
