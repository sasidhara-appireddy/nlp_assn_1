import csv
import pickle
from collections import defaultdict, Counter

data_file = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\hindi\hindi_2500.csv"

# Helper: generate unigrams + bigrams
def generate_ngrams(tokens, n=2):
    unigrams = tokens
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return unigrams + bigrams

category_counts = defaultdict(Counter)
all_categories = set()

with open(data_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["text"]
        labels = row["labels"].split(",")

        tokens = text.split()
        ngrams = generate_ngrams(tokens)

        for label in labels:
            category_counts[label].update(ngrams)
            all_categories.add(label)

model = {
    "category_counts": category_counts,
    "all_categories": sorted(all_categories)
}

with open(r"D:\BITS Acad\4-1\nlp_assn_1\Models\hindi_2500.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
