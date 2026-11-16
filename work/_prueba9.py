import sys, time, matplotlib
matplotlib.use('Agg')
TINTA="#14323F"; ACENTO="#6E5BD6"; GRIS="#B7C3C9"
exec(open('work/common.py').read())
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import f1_score, accuracy_score
t=time.time()
mejor_modelo = Pipeline([('tfidf', TfidfVectorizer(min_df=2, sublinear_tf=True)), ('clf', LinearSVC(random_state=SEMILLA))]).fit(X_train, y_train)
tt=time.time()-t
mejor_nombre='SVM'; pred_test_mejor = mejor_modelo.predict(X_test)
resultados = pd.DataFrame([{'modelo':'TF-IDF + SVM (stub)','fase':4,'f1_macro':round(f1_score(y_test,pred_test_mejor,average='macro'),4),
    'accuracy':round(accuracy_score(y_test,pred_test_mejor),4),'tiempo_s':round(tt,2)}])
# stubs fase 3
K_LDA=6
vec_lda=CountVectorizer(stop_words='english',min_df=5,max_df=0.5); M=vec_lda.fit_transform(muestra_temas['texto_limpio'])
lda=LatentDirichletAllocation(K_LDA,random_state=SEMILLA,max_iter=5).fit(M)
voc=np.array(vec_lda.get_feature_names_out())
palabras_lda=[list(voc[np.argsort(c)[::-1][:10]]) for c in lda.components_]
tema_dom=pd.Series(lda.transform(M).argmax(1),index=muestra_temas.index)
OLLAMA=None; MODELO_OLLAMA=None; MODELO_HF='stub'
MODO = sys.argv[1]
if MODO=='falso':   # simula Colab: HF accesible + LLM con respuestas simuladas + modulos falsos (SOLO prueba del codigo)
    import types, urllib.request
    urllib.request.urlopen = lambda *a, **k: True
    OLLAMA='http://stub'; MODELO_OLLAMA='stub'
    rng=np.random.RandomState(0)
    def preguntar_llm(prompt, max_tokens=200):
        return rng.choice(ETIQUETAS+['maybe.','Sadness!'])
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    class SentenceTransformer:
        def __init__(self,n): self.v=TfidfVectorizer(min_df=2).fit(df['texto_limpio']); self.s=TruncatedSVD(32,random_state=0).fit(self.v.transform(df['texto_limpio'][:8000]))
        def encode(self,t,**k): return self.s.transform(self.v.transform(t))
    class BERTopic:
        def __init__(self,**k): pass
        def fit_transform(self,docs,embeddings=None):
            self.km=KMeans(8,random_state=0,n_init=3).fit(embeddings); y=self.km.labels_.copy(); y[::9]=-1
            self.cv=CountVectorizer(stop_words='english',min_df=3).fit(docs); self.docs=np.array(docs); self.y=y; return list(y),None
        def get_topic_info(self):
            u,c=np.unique(self.y,return_counts=True); return pd.DataFrame({'Topic':u,'Count':c})
        def get_topic(self,t):
            m=self.cv.transform(self.docs[self.y==t]).sum(0).A1; v=self.cv.get_feature_names_out(); i=np.argsort(m)[::-1][:10]; return [(v[j],m[j]) for j in i]
    sys.modules['sentence_transformers']=types.SimpleNamespace(SentenceTransformer=SentenceTransformer)
    sys.modules['bertopic']=types.SimpleNamespace(BERTopic=BERTopic)
    sys.modules['umap']=types.SimpleNamespace(UMAP=lambda **k: None)
if MODO=='real':
    exec(open('work/fase7.py').read())
t0=time.time()
exec(open('work/fase9.py').read())
print(f"== fase9 ({MODO}): {time.time()-t0:.1f}s")
