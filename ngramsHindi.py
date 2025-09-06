import csv
import pickle
from collections import defaultdict, Counter
import math

data_file = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\hindi\hindi_2500.csv"

def generate_ngrams(tokens):
    unigrams = tokens
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return unigrams + bigrams

# Step 1: Count n-grams
category_counts = defaultdict(Counter)
total_counts = Counter()
all_categories = set()
vocab = set()

with open(data_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["text"]
        labels = row["labels"].split(",")

        tokens = text.split()
        ngrams = generate_ngrams(tokens)
        vocab.update(ngrams)

        for label in labels:
            category_counts[label].update(ngrams)
            total_counts[label] += len(ngrams)
            all_categories.add(label)

# Step 2: Convert counts → probabilities with Laplace smoothing
alpha = 1.0  # smoothing constant
vocab_size = len(vocab)

category_probs = defaultdict(dict)

for label in all_categories:
    total = total_counts[label] + alpha * vocab_size
    for ngram in vocab:
        count = category_counts[label][ngram]
        prob = (count + alpha) / total
        category_probs[label][ngram] = prob

model = {
    "category_probs": category_probs,
    "total_counts": total_counts,
    "all_categories": sorted(all_categories),
    "vocab": vocab,
    "alpha": alpha
}

with open(r"D:\BITS Acad\4-1\nlp_assn_1\Models\hindi_2500_prob.pkl", "wb") as f:
    pickle.dump(model, f)

print("Probabilistic n-gram model trained and saved.")


# Step 3: Classify new text
def classify(text, model):
    tokens = text.split()
    ngrams = generate_ngrams(tokens)

    scores = {}
    for label in model["all_categories"]:
        # log-probability to avoid underflow
        log_prob = 0.0
        for ngram in ngrams:
            if ngram in model["vocab"]:
                log_prob += math.log(model["category_probs"][label][ngram])
            else:
                # unseen word → Laplace smoothed probability
                total = model["total_counts"][label] + model["alpha"] * len(model["vocab"])
                log_prob += math.log(model["alpha"] / total)
        scores[label] = log_prob

    return max(scores, key=scores.get), scores
