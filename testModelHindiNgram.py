import csv
import pickle
import math
import os
import numpy as np
from collections import defaultdict, Counter

# Load trained model
with open(r"D:\BITS Acad\4-1\nlp_assn_1\Models\hindi_2500_prob.pkl", "rb") as f:
    model = pickle.load(f)

def generate_ngrams(tokens):
    unigrams = tokens
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return unigrams + bigrams

# Classification function
def classify(text, model):
    tokens = text.split()
    ngrams = generate_ngrams(tokens)

    scores = {}
    for label in model["all_categories"]:
        log_prob = 0.0
        for ngram in ngrams:
            if ngram in model["vocab"]:
                log_prob += math.log(model["category_probs"][label][ngram])
            else:
                total = model["total_counts"][label] + model["alpha"] * len(model["vocab"])
                log_prob += math.log(model["alpha"] / total)
        scores[label] = log_prob

    return max(scores, key=scores.get), scores

# Step 1: Evaluate on test set
test_file = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\hindi\hindi_test.csv"

true_labels = []
pred_labels = []
log_likelihood = 0
total_ngrams = 0

with open(test_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["text"]
        labels = row["labels"].split(",")
        true_label = labels[0]  # use first label if multiple

        pred, scores = classify(text, model)

        true_labels.append(true_label)
        pred_labels.append(pred)

        # Compute log likelihood for perplexity
        tokens = text.split()
        ngrams = generate_ngrams(tokens)
        for ngram in ngrams:
            if ngram in model["vocab"]:
                log_likelihood += math.log(model["category_probs"][pred][ngram])
            else:
                total = model["total_counts"][pred] + model["alpha"] * len(model["vocab"])
                log_likelihood += math.log(model["alpha"] / total)
        total_ngrams += len(ngrams)

# Convert to numpy arrays for easy calculation
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Step 2: Compute metrics manually
unique_labels = list(set(true_labels) | set(pred_labels))

# Accuracy
accuracy = np.mean(true_labels == pred_labels)

# Precision, Recall, F1 (macro-averaged)
precisions, recalls, f1s = [], [], []

for label in unique_labels:
    tp = np.sum((pred_labels == label) & (true_labels == label))
    fp = np.sum((pred_labels == label) & (true_labels != label))
    fn = np.sum((pred_labels != label) & (true_labels == label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

precision_macro = np.mean(precisions)
recall_macro = np.mean(recalls)
f1_macro = np.mean(f1s)

# Intrinsic metric: Perplexity
perplexity = math.exp(-log_likelihood / total_ngrams)

# Step 3: Prepare results string
results = []
results.append("===== Evaluation Results =====")
results.append(f"Intrinsic Evaluation (Perplexity): {perplexity:.3f}")
results.append("Extrinsic Evaluation (Classification Metrics):")
results.append(f"  Accuracy  : {accuracy:.3f}")
results.append(f"  Precision : {precision_macro:.3f}")
results.append(f"  Recall    : {recall_macro:.3f}")
results.append(f"  F1-score  : {f1_macro:.3f}")
results.append("=================================")

results_str = "\n".join(results)

# Print to console
print(results_str)

# Step 4: Save to file
output_dir = r"D:\BITS Acad\4-1\nlp_assn_1\Results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "hindi_test_eval.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(results_str)

print(f"\nResults saved to: {output_file}")
