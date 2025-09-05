import numpy as np
import re
import math
import random
import csv
import pickle
import os

# ---------------- CONFIG ----------------
data_path = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\english\english_2500.csv"
model_tag = "2500"
model_dir = r"D:\BITS Acad\4-1\nlp_assn_1\Models"
# ----------------------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def load_dataset(path):
    print("[Stage 1] Loading dataset...")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre = row["genre"].strip()
            desc = preprocess(row["description"])
            data.append((genre, desc))
    print(f"[Stage 1 Complete] Loaded {len(data)} samples.")
    return data

def build_ngram_model(data, n=2):
    print(f"[Stage 2] Building {n}-gram model...")
    model = {}
    for genre, tokens in data:
        if genre not in model:
            model[genre] = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            model[genre][ngram] = model[genre].get(ngram, 0) + 1
    print(f"[Stage 2 Complete] {n}-gram model built.")
    return model

def score_desc(tokens, model, vocab_size, n=2):
    scores = {}
    for genre in model:
        log_prob = 0
        total_count = sum(model[genre].values())
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            count = model[genre].get(ngram, 0)
            prob = (count + 1) / (total_count + vocab_size)
            log_prob += math.log(prob)
        scores[genre] = log_prob
    return scores

def predict(desc, bigram_model, trigram_model, vocab, alpha=0.5):
    tokens = preprocess(desc)
    bigram_scores = score_desc(tokens, bigram_model, len(vocab), n=2)
    trigram_scores = score_desc(tokens, trigram_model, len(vocab), n=3)
    combined_scores = {}
    for genre in bigram_scores:
        combined_scores[genre] = alpha * bigram_scores[genre] + (1 - alpha) * trigram_scores[genre]
    return max(combined_scores, key=combined_scores.get)

def accuracy(data, bigram_model, trigram_model, vocab, alpha):
    correct = 0
    for genre, tokens in data:
        desc = " ".join(tokens)
        pred = predict(desc, bigram_model, trigram_model, vocab, alpha)
        if pred == genre:
            correct += 1
    return correct / len(data)

# ---------------- MAIN WORKFLOW -----------------
print("[Start] Training workflow initiated.")

data = load_dataset(data_path)
random.shuffle(data)

split_idx = int(0.8 * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]

vocab = set([word for _, tokens in train_data for word in tokens])

bigram_model = build_ngram_model(train_data, n=2)
trigram_model = build_ngram_model(train_data, n=3)

print("[Stage 3] Hyperparameter tuning (alpha between 0 and 1)...")
best_alpha, best_acc = 0, 0
for alpha in np.linspace(0, 1, 11):  # 0.0, 0.1, ..., 1.0
    acc = accuracy(val_data, bigram_model, trigram_model, vocab, alpha)
    print(f"  alpha={alpha:.1f} -> Validation Accuracy={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha

print(f"[Stage 3 Complete] Best alpha = {best_alpha}, Validation Accuracy = {best_acc:.3f}")

# Save the trained model
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"ngram_model_{model_tag}.pkl")

with open(model_path, "wb") as f:
    pickle.dump((bigram_model, trigram_model, vocab, best_alpha), f)

print("[Training Complete] Model saved to:", model_path)
