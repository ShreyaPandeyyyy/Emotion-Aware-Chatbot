import os, joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
VEC_PATH  = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLF_PATH  = os.path.join(MODEL_DIR, "classifier.pkl")

# Load saved model + vectorizer
vectorizer = joblib.load(VEC_PATH)
clf = joblib.load(CLF_PATH)

def predict(text):
    X = vectorizer.transform([text])
    return clf.predict(X)[0]

# Demo
while True:
    text = input("\nEnter a sentence (or 'quit' to exit): ")
    if text.lower() == "quit":
        break
    label = predict(text)
    print(f"➡ Emotion: {label}")
