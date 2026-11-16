import sys, time, matplotlib
matplotlib.use('Agg')
TINTA="#14323F"; ACENTO="#6E5BD6"; GRIS="#B7C3C9"
exec(open('work/common.py').read())
# --- STUBS de la Fase 4 (no forman parte de la fase) ---
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
t=time.time()
mejor_modelo = Pipeline([('tfidf', TfidfVectorizer(min_df=2, sublinear_tf=True)), ('clf', LinearSVC(random_state=SEMILLA))]).fit(X_train, y_train)
tt=time.time()-t
mejor_nombre='SVM'; pred_test_mejor = mejor_modelo.predict(X_test)
resultados = pd.DataFrame([{'modelo':'TF-IDF + SVM (stub)','fase':4,'f1_macro':round(f1_score(y_test,pred_test_mejor,average='macro'),4),
    'accuracy':round(accuracy_score(y_test,pred_test_mejor),4),'tiempo_s':round(tt,2)}])
t0=time.time()
exec(open(f'work/{sys.argv[1]}.py').read())
print(f"== {sys.argv[1]}: {time.time()-t0:.1f}s")
