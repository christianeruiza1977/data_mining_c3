"""Ensambla el notebook final a partir de la plantilla y de work/faseN.py (orquestador)."""
import re, sys, copy, nbformat
from pathlib import Path
ORIG = "/root/.claude/uploads/619f4b5e-b391-58c8-894a-fd78c5af312b/4cb1c6fd-Ejercicio_Practico_Semana3.ipynb"
SALIDA = "grupo9_Semana3_Ejercicio.ipynb"
# ids de la plantilla por fase: (encabezado, código, conclusión)
IDS = {1:("ac56e3a6","0dbaedd4","aa9e1ceb"), 2:("721dddcc","51340b7e","74c34dfd"), 3:("f7dc0676","e8aba557","1cb93dbd"),
       4:("9baec58e","49615ffa","7331377e"), 5:("3b1c0140","4252fe06","da115839"), 6:("965b46e4","fe432bcc","80c65e10"),
       7:("840ff023","77f154ed","37f8d8a7"), 8:("07488fe3","a88b035b","f562db96"), 9:("9af4e78c","785ad184","bc85e6e2")}

VERSIONES = """# ===== Reproducibilidad: semilla y versiones de las librerías =====
import os, sys, platform
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow
import numpy, pandas, sklearn, spacy, pyspark
SEMILLA = 42   # semilla única de todo el notebook
print("Semilla:", SEMILLA)
print("Python", sys.version.split()[0], "|", platform.platform())
for lib in (numpy, pandas, sklearn, spacy, pyspark):
    print(f"{lib.__name__:<10}", lib.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)"""

def leer_fase(ruta):
    """Devuelve lista de (tipo, texto) según los marcadores '# %% [tipo]'."""
    celdas, tipo, buf = [], None, []
    for linea in Path(ruta).read_text(encoding="utf8").splitlines():
        m = re.match(r"# %% \[(\w+)\]\s*$", linea)
        if m:
            if tipo: celdas.append((tipo, buf))
            tipo, buf = m.group(1), []
        else:
            buf.append(linea)
    if tipo: celdas.append((tipo, buf))
    salida = []
    for tipo, lineas in celdas:
        while lineas and not lineas[-1].strip(): lineas.pop()
        while lineas and not lineas[0].strip(): lineas.pop(0)
        if tipo in ("que", "md", "conclusion"):      # markdown: quitar el '# ' inicial
            lineas = [re.sub(r"^# ?", "", l) for l in lineas]
        salida.append((tipo, "\n".join(lineas)))
    return salida

def main():
    nb = nbformat.read(ORIG, as_version=4)
    pos = {c.get("id"): i for i, c in enumerate(nb.cells)}
    def celda(id_): return nb.cells[pos[id_]]
    nuevos = {}   # id -> lista de celdas que la sustituyen
    for n, (h, c, k) in IDS.items():
        f = Path(f"work/fase{n}.py")
        if not f.exists():
            print(f"[aviso] falta work/fase{n}.py: fase {n} queda como la plantilla"); continue
        cel = leer_fase(f)
        que = [t for ti, t in cel if ti == "que"]
        if que:
            celda(h).source = celda(h).source.replace("_completar antes del código_", que[0])
        concl = [t for ti, t in cel if ti == "conclusion"]
        if concl:
            celda(k).source = "**Conclusión de negocio:** " + concl[0] if not concl[0].startswith("**") else concl[0]
        cuerpo = []
        for ti, t in cel:
            if ti == "code": cuerpo.append(nbformat.v4.new_code_cell(t))
            elif ti == "md": cuerpo.append(nbformat.v4.new_markdown_cell(t))
        if cuerpo: nuevos[c] = cuerpo

    # --- datos del grupo, versiones y limpieza de la plantilla (orquestador) ---
    t = celda("1531455e")
    t.source = t.source.replace("**Grupo N.º:** ___", "**Grupo N.º:** 9")
    t.source = t.source.replace("| 1 | | |", "| 1 | Kevin Ambuludi | |").replace("| 2 | | |", "| 2 | Rodrigo Soria | |").replace("| 3 | | |", "| 3 | Christian Ruiz | |").replace("| 4 | | |\n", "").replace("\n| 4 | | |", "")
    for id_, n, nombre in (("a9b870ee",1,"Kevin Ambuludi"),("a6b87332",2,"Rodrigo Soria"),("eaaae2aa",3,"Christian Ruiz")):
        celda(id_).source = celda(id_).source.replace("_nombre_", nombre)
    nb.cells = [c for c in nb.cells if c.get("id") != "150243c6"]   # el grupo tiene 3 integrantes
    pos = {c.get("id"): i for i, c in enumerate(nb.cells)}
    versiones = nbformat.v4.new_code_cell(VERSIONES)
    nb.cells.insert(pos["bf3c2a0c"] + 1, versiones)
    cells = []
    for c in nb.cells:
        if c.get("id") in nuevos: cells.extend(nuevos[c["id"]])
        else: cells.append(c)
    nb.cells = cells
    for i, c in enumerate(nb.cells):          # ids únicos y sin salidas viejas
        c["id"] = c.get("id") or f"c{i:03d}"
    nbformat.write(nb, SALIDA)
    print("escrito", SALIDA, len(nb.cells), "celdas")

if __name__ == "__main__":
    main()
