import re
import math
import csv
import pickle
import numpy as np
import os

# ---------------- CONFIG ----------------
model_path = r"D:\BITS Acad\4-1\nlp_assn_1\Models\ngram_model_2500.pkl"
test_data_path = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\english\english_test.csv"
result_tag = "2500English"
result_dir = r"D:\BITS Acad\4-1\nlp_assn_1\Results"
# ----------------------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre = row["genre"].strip()
            desc = preprocess(row["description"])
            data.append((genre, desc))
    return data

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
    return max(combined_scores, key=combined_scores.get), tokens

# Intrinsic evaluation: Perplexity
def perplexity(dataset, bigram_model, trigram_model, vocab, alpha):
    total_log_prob = 0
    total_len = 0
    for _, tokens in dataset:
        bigram_scores = score_desc(tokens, bigram_model, len(vocab), n=2)
        trigram_scores = score_desc(tokens, trigram_model, len(vocab), n=3)
        best_genre = max(bigram_scores, key=bigram_scores.get)
        log_prob = alpha * bigram_scores[best_genre] + (1 - alpha) * trigram_scores[best_genre]
        total_log_prob += log_prob
        total_len += len(tokens)
    return math.exp(-total_log_prob / total_len)

# Extrinsic evaluation: Classification metrics
def classification_metrics(true_labels, pred_labels):
    # Include all genres from both truth and predictions
    genres = sorted(list(set(true_labels) | set(pred_labels)))
    tp, fp, fn = {}, {}, {}
    for g in genres:
        tp[g], fp[g], fn[g] = 0, 0, 0

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    precision_list, recall_list, f1_list = [], [], []
    for g in genres:
        prec = tp[g] / (tp[g] + fp[g]) if (tp[g] + fp[g]) > 0 else 0
        rec = tp[g] / (tp[g] + fn[g]) if (tp[g] + fn[g]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    accuracy = sum(tp.values()) / len(true_labels)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    return accuracy, precision, recall, f1

# ---------------- MAIN -----------------
print("[Stage 1] Loading model and test dataset...")
with open(model_path, "rb") as f:
    bigram_model, trigram_model, vocab, alpha = pickle.load(f)

test_data = load_dataset(test_data_path)
print(f"[Stage 1 Complete] Loaded {len(test_data)} test samples.")

print("[Stage 2] Running predictions...")
true_labels, pred_labels = [], []
for genre, tokens in test_data:
    desc = " ".join(tokens)
    pred, _ = predict(desc, bigram_model, trigram_model, vocab, alpha)
    true_labels.append(genre)
    pred_labels.append(pred)

print("[Stage 3] Computing metrics...")
ppl = perplexity(test_data, bigram_model, trigram_model, vocab, alpha)
acc, prec, rec, f1 = classification_metrics(true_labels, pred_labels)

# Save results
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"results_{result_tag}.txt")

with open(result_path, "w", encoding="utf-8") as f:
    f.write("===== Evaluation Results =====\n")
    f.write(f"Intrinsic Evaluation (Perplexity): {ppl:.3f}\n")
    f.write("Extrinsic Evaluation (Classification Metrics):\n")
    f.write(f"  Accuracy  : {acc:.3f}\n")
    f.write(f"  Precision : {prec:.3f}\n")
    f.write(f"  Recall    : {rec:.3f}\n")
    f.write(f"  F1-score  : {f1:.3f}\n")
    f.write("=================================\n")

print("\n===== Evaluation Results =====")
print(f"Intrinsic Evaluation (Perplexity): {ppl:.3f}")
print("Extrinsic Evaluation (Classification Metrics):")
print(f"  Accuracy  : {acc:.3f}")
print(f"  Precision : {prec:.3f}")
print(f"  Recall    : {rec:.3f}")
print(f"  F1-score  : {f1:.3f}")
print("=================================")

print("[Evaluation Complete] Results saved to:", result_path)
