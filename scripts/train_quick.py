import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import train_baseline

# Small synthetic dataset for quick training
joy = ["I am so happy", "This made my day", "Feeling wonderful", "I love this", "What a great time"]
sad = ["I feel sad", "This is depressing", "I am unhappy", "It’s a bad day", "I miss my friends"]
anger = ["I am angry", "This makes me mad", "So frustrating", "I hate this", "I am furious"]
fear = ["I am scared", "This worries me", "I feel anxious", "I’m afraid", "This is terrifying"]
surprise = ["Wow I didn’t expect that", "That’s surprising", "No way!", "Unbelievable", "I’m shocked"]
neutral = ["Okay", "Thanks", "I will check", "Not sure", "It is fine"]

texts = joy + sad + anger + fear + surprise + neutral
labels = (
    ["joy"] * len(joy)
    + ["sadness"] * len(sad)
    + ["anger"] * len(anger)
    + ["fear"] * len(fear)
    + ["surprise"] * len(surprise)
    + ["neutral"] * len(neutral)
)

# Train and save baseline model
train_baseline(texts, labels)
print("✅ Model trained and saved in ./models/")
