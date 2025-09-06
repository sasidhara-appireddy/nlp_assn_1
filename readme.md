hello to the assignment..

INSTALL GIT YOU MINGWITS

## N-Grams

Testing:-
Change model_path → point to the model you want (e.g., ngram_model_500.pkl, ngram_model_2500.pkl).
Change test_data_path → point to your test dataset CSV.
Run the script.
It prints perplexity + classification metrics.

## Hindi

Training phase

The training dataset is read line by line.

Each text is split into tokens.

Both unigrams (single words) and bigrams (pairs of consecutive words) are extracted.

For each label (category) the text belongs to, a frequency table of n-grams is updated.

Then, counts are normalized into probabilities with Laplace smoothing:

𝑃
(
ngram
∣
label
)
=
count(ngram,label)

- 𝛼
  total(label)
- 𝛼
  ⋅
  ∣
  𝑉
  ∣
  P(ngram∣label)=
  total(label)+α⋅∣V∣
  count(ngram,label)+α
  ​

where
∣
𝑉
∣
∣V∣ = vocabulary size,
𝛼
=
1.0
α=1.0.

Classification phase

When a new text is given, it is tokenized into unigrams+bigrams.

For each category, the model computes the log probability of seeing those n-grams under that category:

log
⁡
𝑃
(
text
∣
label
)
=
∑
ngram in text
log
⁡
𝑃
(
ngram
∣
label
)
logP(text∣label)=
ngram in text
∑
​

logP(ngram∣label)

The category with the highest total log probability is chosen as the prediction.

Evaluation phase

Intrinsic evaluation (Perplexity):
Measures how “surprised” the model is by the test data.
Lower perplexity = better language modeling.

# Perplexity

𝑒
−
1
𝑁
∑
log
⁡
𝑃
(
ngram
)
Perplexity=e
−
N
1
​

∑logP(ngram)

Extrinsic evaluation (Classification):
Compares predicted labels with true labels to compute:

Accuracy (fraction correct)

Precision (correct positive predictions / all positive predictions)

Recall (correct positive predictions / all true positives)

F1 (harmonic mean of precision & recall)
