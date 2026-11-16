import sys, time, matplotlib
matplotlib.use('Agg')
TINTA="#14323F"; ACENTO="#6E5BD6"; GRIS="#B7C3C9"
t0=time.time()
exec(open('work/common.py').read())
print("common", time.time()-t0)
for _fase in sys.argv[1:]:
    t=time.time(); exec(open(f"work/{_fase}.py").read()); print(f"== {_fase}: {time.time()-t:.1f}s")
