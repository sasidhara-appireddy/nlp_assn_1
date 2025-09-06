import csv
import pickle
from collections import Counter
import math
import os

# Custom string for output file
custom_string = "2500"
output_dir = r"D:\BITS Acad\4-1\nlp_assn_1\Results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"hindi_result_{custom_string}.txt")

# Helper: generate unigrams + bigrams
def generate_ngrams(tokens, n=2):
    unigrams = tokens
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return unigrams + bigrams

print("Loading trained model...")
with open(r"D:\BITS Acad\4-1\nlp_assn_1\Models\hindi_2500.pkl", "rb") as f:
    model = pickle.load(f)

category_counts = model["category_counts"]
all_categories = model["all_categories"]
print(f"Loaded model with {len(all_categories)} categories.")

# Load test set
print("Loading test dataset...")
test_file = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\hindi\hindi_test.csv"
texts, labels = [], []

with open(test_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row["text"])
        labels.append(row["labels"].split(","))

print(f"Loaded {len(texts)} test samples.")

# Evaluation loop
correct, total = 0, 0
tp, fp, fn = 0, 0, 0
log_prob_sum, token_count = 0, 0

print("Evaluating...")
for i, text in enumerate(texts):
    tokens = text.split()
    ngrams = generate_ngrams(tokens)

    scores = {}
    for cat in all_categories:
        cat_counter = category_counts[cat]
        scores[cat] = sum(cat_counter[ng] for ng in ngrams)

    predicted = sorted(scores, key=scores.get, reverse=True)[:1]

    # Accuracy check (only valid if gold label exists in training categories)
    gold = [g for g in labels[i] if g in all_categories]

    if not gold:
        # test label not in training → impossible to predict → count as FN
        fn += 1
    else:
        if any(p in gold for p in predicted):
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1
    total += 1

    # Perplexity calculation
    for ng in ngrams:
        count = sum(category_counts[c][ng] for c in all_categories)
        prob = (count + 1) / (sum(sum(cat_counter.values()) for cat_counter in category_counts.values()) + len(category_counts))
        log_prob_sum -= math.log(prob)
        token_count += 1

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(texts)} samples...")

# Final metrics
accuracy = correct / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
perplexity = math.exp(log_prob_sum / token_count) if token_count > 0 else float("inf")

# Write results to file
print("Writing results to file...")
with open(output_file, "w", encoding="utf-8") as out:
    out.write("Intrinsic Evaluation (Perplexity): {:.3f}\n".format(perplexity))
    out.write("Extrinsic Evaluation (Classification Metrics):\n")
    out.write("  Accuracy  : {:.3f}\n".format(accuracy))
    out.write("  Precision : {:.3f}\n".format(precision))
    out.write("  Recall    : {:.3f}\n".format(recall))
    out.write("  F1-score  : {:.3f}\n".format(f1))

print(f"✅ Evaluation complete. Results saved to {output_file}")
