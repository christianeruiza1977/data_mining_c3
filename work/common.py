# ===== CONTRATO COMÚN (código idéntico en el notebook final) =====
import re, html, time, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
SEMILLA = 42
np.random.seed(SEMILLA)
Path("results").mkdir(exist_ok=True); Path("figuras").mkdir(exist_ok=True)

# --- Carga: Kaggle (kagglehub) si hay acceso; si no, copia local data/tweet_emotions.csv ---
def cargar_csv():
    try:
        import kagglehub
        ruta = kagglehub.dataset_download('pashupatigupta/emotion-detection-from-text')
        return pd.read_csv(f'{ruta}/tweet_emotions.csv')
    except Exception:
        return pd.read_csv('data/tweet_emotions.csv')
df_crudo = cargar_csv()

# --- Limpieza (decisiones documentadas en Fase 1) ---
def limpiar(t):
    t = html.unescape(t)                       # &amp; -> &, &quot; -> "
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)   # URLs
    t = re.sub(r'@\w+', ' ', t)                # menciones @usuario
    t = t.replace('#', '')                     # hashtag: conserva la palabra
    return re.sub(r'\s+', ' ', t).strip()      # espacios múltiples

df = df_crudo.copy()
# duplicados: textos con etiquetas contradictorias se eliminan todos; el resto, se conserva la 1.ª copia
conflicto = df.groupby('content')['sentiment'].transform('nunique') > 1
df = df[~conflicto].drop_duplicates('content').copy()
df['texto_limpio'] = df['content'].map(limpiar)
df = df[df['texto_limpio'].str.len() > 0].reset_index(drop=True)   # tweets que quedaron vacíos (solo menciones/URL)
ETIQUETAS = sorted(df['sentiment'].unique())

# --- Partición estratificada 70/15/15 (se define en la Fase 4; la usan las fases 5-9) ---
idx_ent, idx_tmp = train_test_split(df.index, test_size=0.30, stratify=df['sentiment'], random_state=SEMILLA)
idx_val, idx_pru = train_test_split(idx_tmp, test_size=0.50, stratify=df.loc[idx_tmp, 'sentiment'], random_state=SEMILLA)
X_train, y_train = df.loc[idx_ent, 'texto_limpio'], df.loc[idx_ent, 'sentiment']
X_val,   y_val   = df.loc[idx_val, 'texto_limpio'], df.loc[idx_val, 'sentiment']
X_test,  y_test  = df.loc[idx_pru, 'texto_limpio'], df.loc[idx_pru, 'sentiment']

# --- Muestra de 5 000 tweets compartida por LDA (Fase 3) y BERTopic (Fase 9a) ---
muestra_temas = df.sample(5000, random_state=SEMILLA)
