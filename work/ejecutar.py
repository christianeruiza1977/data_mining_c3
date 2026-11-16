import nbformat, sys, time
from nbclient import NotebookClient
ruta="grupo9_Semana3_Ejercicio.ipynb"
nb=nbformat.read(ruta,as_version=4)
t=time.time()
c=NotebookClient(nb,timeout=3600,kernel_name="python3",resources={"metadata":{"path":"."}},allow_errors=True)
c.execute()
nbformat.write(nb,ruta)
err=[(i,o.ename,o.evalue[:200]) for i,cl in enumerate(nb.cells) if cl.cell_type=="code" for o in cl.get("outputs",[]) if o.output_type=="error"]
print("min:",round((time.time()-t)/60,1),"errores:",err)
