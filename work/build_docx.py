"""Rellena la guía DOCX con las respuestas de work/respuesta_faseN.md (orquestador)."""
import re, copy, datetime
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

ORIG = "/tmp/claude-0/-home-user-data-mining-c3/619f4b5e-b391-58c8-894a-fd78c5af312b/scratchpad/guia/guia.docx"
SALIDA = "grupo9_Semana3_Guia.docx"
doc = Document(ORIG)
hijos = list(doc.element.body.iterchildren())
P = lambda i: Paragraph(hijos[i], doc)

def runs_md(par, texto):
    """Agrega texto con **negrita** y `código` a un párrafo."""
    for trozo in re.split(r"(\*\*.+?\*\*|`.+?`)", texto):
        if not trozo: continue
        if trozo.startswith("**"): par.add_run(trozo[2:-2]).bold = True
        elif trozo.startswith("`"):
            r = par.add_run(trozo[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        else: par.add_run(trozo)

def nuevo_parrafo_antes(ref_el, texto="", estilo="Normal"):
    p = copy.deepcopy(hijos[3]); [p.remove(r) for r in list(p) if r.tag != qn("w:pPr")]   # párrafo vacío con formato Normal
    ref_el.addprevious(p)
    par = Paragraph(p, doc)
    if texto: runs_md(par, texto)
    return par

def tabla_antes(ref_el, filas, ancho_cm=16.5, tam=9):
    t = doc.add_table(rows=len(filas), cols=len(filas[0]))
    try: t.style = "Table Grid"
    except Exception: pass
    for i, fila in enumerate(filas):
        for j, celda in enumerate(fila):
            c = t.cell(i, j); c.text = ""
            par = c.paragraphs[0]; runs_md(par, celda.strip())
            for r in par.runs:
                r.font.size = Pt(tam)
                if i == 0: r.bold = True
    ref_el.addprevious(t._tbl)
    return t

def imagen_antes(ref_el, ruta, ancho=15):
    par = nuevo_parrafo_antes(ref_el)
    par.add_run().add_picture(ruta, width=Cm(ancho))
    return par

def volcar_md(ref_el, ruta_md):
    """Inserta el contenido de un .md (párrafos, tablas, figuras) antes de ref_el."""
    lineas = Path(ruta_md).read_text(encoding="utf8").splitlines()
    figuras, bloque, tabla = [], [], []
    def cerrar_parrafo():
        nonlocal bloque
        if bloque: nuevo_parrafo_antes(ref_el, " ".join(bloque)); bloque = []
    def cerrar_tabla():
        nonlocal tabla
        if tabla:
            filas = [[c.strip() for c in l.strip().strip("|").split("|")] for l in tabla if not re.match(r"^\s*\|?\s*:?-{2,}", l)]
            tabla_antes(ref_el, filas); nuevo_parrafo_antes(ref_el); tabla = []
    for l in lineas:
        if l.startswith("**Pregunta de la guía"): continue
        if l.strip().startswith("|"): cerrar_parrafo(); tabla.append(l); continue
        cerrar_tabla()
        if l.lower().startswith(("figuras para la guía", "figura para la guía")):
            figuras += re.findall(r"figuras/[\w\-\.]+\.(?:png|svg)", l); continue
        if not l.strip(): cerrar_parrafo(); continue
        if re.match(r"^\s*[-*] ", l): cerrar_parrafo(); nuevo_parrafo_antes(ref_el, "•  " + re.sub(r"^\s*[-*] ", "", l)); continue
        bloque.append(l.strip())
    cerrar_parrafo(); cerrar_tabla()
    for f in figuras:
        if f.endswith(".png") and Path(f).exists():
            imagen_antes(ref_el, f)
            c = nuevo_parrafo_antes(ref_el, f"Figura: {Path(f).stem.replace('_', ' ')} (archivo {f})")
            for r in c.runs: r.italic = True; r.font.size = Pt(9)

# --- Fases: borrar los 3 renglones de guiones bajos y poner la respuesta ---
BLOQUES = {1: (36, 37, 38), 2: (42, 43, 44), 3: (48, 49, 50), 4: (54, 55, 56), 5: (60, 61, 62),
           6: (66, 67, 68), 7: (72, 73, 74), 8: (78, 79, 80), 9: (84, 85, 86)}
for n, (a, b, c) in BLOQUES.items():
    ref = hijos[c]                                   # los textos se insertan antes del último renglón y luego se borran los tres
    md = Path(f"work/respuesta_fase{n}.md")
    if not md.exists():
        print("falta", md); continue
    volcar_md(ref, md)
    if n == 8:                                       # tabla de alternativas (generada por el notebook)
        import pandas as pd
        alt = pd.read_csv("results/fase8_alternativas.csv")
        nuevo_parrafo_antes(ref, "**Tabla de evaluación de alternativas** (misma tabla del notebook, fase 8):")
        tabla_antes(ref, [list(alt.columns)] + alt.values.tolist(), tam=7.5)
        nuevo_parrafo_antes(ref)
    for i in (a, b, c): hijos[i].getparent().remove(hijos[i])

# --- Portada ---
def reemplazar(par, viejo, nuevo):
    for r in par.runs:
        if viejo in r.text: r.text = r.text.replace(viejo, nuevo); return
    texto = par.text.replace(viejo, nuevo)
    for r in par.runs[1:]: r.text = ""
    par.runs[0].text = texto
reemplazar(P(6), "[N.º de grupo]", "9")
reemplazar(P(7), "[Nombre — rol]  ·  [Nombre — rol]  ·  [Nombre — rol]  ·  [Nombre — rol]",
           "Kevin Ambuludi — [rol]  ·  Rodrigo Soria — [rol]  ·  Christian Ruiz — [rol]")
reemplazar(P(8), "[dd/mm/aaaa]", datetime.date.today().strftime("%d/%m/%Y"))
for i, nombre in ((90, "Kevin Ambuludi"), (95, "Rodrigo Soria"), (100, "Christian Ruiz")):
    reemplazar(P(i), "[nombre]", nombre)
for i in range(105, 110): hijos[i].getparent().remove(hijos[i])      # el grupo es de 3 integrantes
doc.save(SALIDA)
print("guardado", SALIDA)
