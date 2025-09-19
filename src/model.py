import os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
VEC_PATH  = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLF_PATH  = os.path.join(MODEL_DIR, "classifier.pkl")

def train_baseline(texts, labels):
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X, labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(clf, CLF_PATH)
