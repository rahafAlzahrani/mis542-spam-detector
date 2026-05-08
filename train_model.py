"""
MIS 542: Spam Detection Project
train_model.py — Full pipeline: preprocessing, training, evaluation, SMOTE, save model

Run this ONCE locally before deploying the Streamlit app:
    python train_model.py

Outputs:
    model.pkl       ← saved vectorizer + best model (load in app.py)
    results.pkl     ← all metrics for display in app.py
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score)

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("MIS 542  —  SPAM DETECTION PIPELINE")
print("=" * 60)

df = pd.read_csv('Project_file.csv', usecols=['class', 'message'])
df.columns = ['label', 'message']
df = df.dropna(subset=['message'])
df['label'] = df['label'].str.strip()

print(f"\n[1] DATA LOADED")
print(f"    Total   : {len(df)}")
print(f"    Valid   : {(df['label'] == 'valid').sum()}")
print(f"    Spam    : {(df['label'] == 'spam').sum()}")


# ─────────────────────────────────────────────
# 2. PREPROCESSING  (all 6 required steps)
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Text cleaning pipeline — applied to every message.
      i.   Lowercase
      ii.  Remove special characters
      iii. Remove stop words
      iv.  Remove hyperlinks
      v.   Remove numbers
      vi.  Remove extra whitespace
    """
    text = str(text)

    # i. Lowercase
    text = text.lower()

    # iv. Remove hyperlinks (before special-char removal so slashes are still there)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # v. Remove numbers
    text = re.sub(r'\d+', '', text)

    # ii. Remove special characters — keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # vi. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # iii. Remove stop words (sklearn's built-in English stop-word list)
    tokens = [t for t in text.split()
              if t not in ENGLISH_STOP_WORDS and len(t) > 1]

    return ' '.join(tokens)


df['clean'] = df['message'].apply(preprocess)

print(f"\n[2] PREPROCESSING COMPLETE")
print(f"    Original : {df['message'].iloc[2][:70]}")
print(f"    Cleaned  : {df['clean'].iloc[2][:70]}")


# ─────────────────────────────────────────────
# 3. WORD CLOUDS  (top-60 words per class)
# ─────────────────────────────────────────────
def draw_word_cloud(series, title, color, ax):
    freq = Counter(' '.join(series).split()).most_common(60)
    if not freq:
        return
    words, counts = zip(*freq)
    max_c = max(counts)
    np.random.seed(42)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off'); ax.set_facecolor('#f8f8f8')
    ax.set_title(title, fontsize=14, fontweight='bold')
    for word, count in freq:
        fs = int(8 + (count / max_c) * 28)
        al = 0.6 + 0.4 * (count / max_c)
        ax.text(np.random.uniform(0.05, 0.95),
                np.random.uniform(0.05, 0.95),
                word, fontsize=fs, ha='center', va='center',
                color=color, alpha=al,
                fontweight='bold' if count > max_c * 0.5 else 'normal')

print(f"\n[3] WORD CLOUDS...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Word Clouds — Valid vs Spam', fontsize=16, fontweight='bold')
draw_word_cloud(df[df['label'] == 'valid']['clean'], 'Valid Emails', '#2196F3', axes[0])
draw_word_cloud(df[df['label'] == 'spam']['clean'],  'Spam Emails',  '#E53935', axes[1])
plt.tight_layout()
plt.savefig('wordclouds.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: wordclouds.png")


# ─────────────────────────────────────────────
# 4. TF-IDF TRANSFORMATION
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean'])
y = (df['label'] == 'spam').astype(int)   # 1 = spam, 0 = valid

print(f"\n[4] TF-IDF  →  feature matrix: {X.shape}")


# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n[5] SPLIT  →  train: {X_train.shape[0]}  |  test: {X_test.shape[0]}")


# ─────────────────────────────────────────────
# 6. TRAIN 4 MODELS  +  CONFUSION MATRICES
# ─────────────────────────────────────────────
MODELS = {
    'KNN  (K-Nearest Neighbors)':  KNeighborsClassifier(n_neighbors=5),
    'NB   (Naive Bayes)':          MultinomialNB(),
    'LR   (Logistic Regression)':  LogisticRegression(max_iter=1000, random_state=42),
    'DT   (Decision Tree)':        DecisionTreeClassifier(max_depth=20, random_state=42),
}

results_before = {}
print(f"\n[6] TRAINING & EVALUATION (before SMOTE)")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Confusion Matrices — Before SMOTE', fontsize=15, fontweight='bold')

for idx, (name, clf) in enumerate(MODELS.items()):
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=['Valid', 'Spam'],
                                    output_dict=True)
    results_before[name] = {
        'acc': acc, 'cm': cm, 'report': report, 'clf': clf
    }

    print(f"\n  {name}")
    print(f"    Accuracy          : {acc:.4f}")
    print(f"    Spam Precision    : {report['Spam']['precision']:.4f}")
    print(f"    Spam Recall       : {report['Spam']['recall']:.4f}")
    print(f"    Spam F1           : {report['Spam']['f1-score']:.4f}")

    # Plot
    ax = axes.flatten()[idx]
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Valid', 'Spam']); ax.set_yticklabels(['Valid', 'Spam'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_before_smote.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    Saved: confusion_matrices_before_smote.png")


# ─────────────────────────────────────────────
# 7. SMOTE  (Synthetic Minority Over-sampling)
# ─────────────────────────────────────────────
print(f"\n[7] SMOTE")

X_train_dense = X_train.toarray()
X_test_dense  = X_test.toarray()

np.random.seed(42)
minority_idx  = np.where(y_train == 1)[0]
n_majority    = int((y_train == 0).sum())
n_minority    = int((y_train == 1).sum())
n_needed      = n_majority - n_minority

print(f"    Before → Valid: {n_majority}  |  Spam: {n_minority}")

X_min = X_train_dense[minority_idx]
synthetic = []
for _ in range(n_needed):
    a, b  = np.random.randint(0, len(X_min), 2)
    lam   = np.random.random()
    synthetic.append(X_min[a] + lam * (X_min[b] - X_min[a]))

X_smote = np.vstack([X_train_dense, np.array(synthetic)])
y_smote = np.concatenate([y_train.values, np.ones(n_needed, dtype=int)])

print(f"    After  → Valid: {int((y_smote==0).sum())}  |  Spam: {int((y_smote==1).sum())}")


# ─────────────────────────────────────────────
# 8. RETRAIN 4 MODELS WITH SMOTE  +  CONFUSION MATRICES
# ─────────────────────────────────────────────
results_after = {}
print(f"\n[8] TRAINING & EVALUATION (after SMOTE)")
print("-" * 60)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Confusion Matrices — After SMOTE', fontsize=15, fontweight='bold')

model_classes = [
    KNeighborsClassifier(n_neighbors=5),
    MultinomialNB(),
    LogisticRegression(max_iter=1000, random_state=42),
    DecisionTreeClassifier(max_depth=20, random_state=42),
]

for idx, (name, clf_fresh) in enumerate(zip(MODELS.keys(), model_classes)):
    # NB requires non-negative input
    X_tr = np.abs(X_smote)      if 'NB' in name else X_smote
    X_te = np.abs(X_test_dense) if 'NB' in name else X_test_dense

    clf_fresh.fit(X_tr, y_smote)
    y_pred_s = clf_fresh.predict(X_te)
    acc_s    = accuracy_score(y_test, y_pred_s)
    cm_s     = confusion_matrix(y_test, y_pred_s)
    report_s = classification_report(y_test, y_pred_s,
                                     target_names=['Valid', 'Spam'],
                                     output_dict=True)
    results_after[name] = {
        'acc': acc_s, 'cm': cm_s, 'report': report_s, 'clf': clf_fresh
    }

    print(f"\n  {name}  [SMOTE]")
    print(f"    Accuracy          : {acc_s:.4f}")
    print(f"    Spam Precision    : {report_s['Spam']['precision']:.4f}")
    print(f"    Spam Recall       : {report_s['Spam']['recall']:.4f}")
    print(f"    Spam F1           : {report_s['Spam']['f1-score']:.4f}")

    ax = axes2.flatten()[idx]
    ax.imshow(cm_s, interpolation='nearest', cmap='Oranges')
    ax.set_title(f"{name}  [SMOTE]", fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Valid', 'Spam']); ax.set_yticklabels(['Valid', 'Spam'])
    thresh = cm_s.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_s[i, j]), ha='center', va='center',
                    color='white' if cm_s[i, j] > thresh else 'black',
                    fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_after_smote.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    Saved: confusion_matrices_after_smote.png")


# ─────────────────────────────────────────────
# 9. BEFORE vs AFTER SMOTE COMPARISON CHART
# ─────────────────────────────────────────────
short_names = ['KNN', 'NB', 'LR', 'DT']
acc_b = [results_before[n]['acc'] for n in MODELS]
acc_a = [results_after[n]['acc']  for n in MODELS]
rec_b = [results_before[n]['report']['Spam']['recall'] for n in MODELS]
rec_a = [results_after[n]['report']['Spam']['recall']  for n in MODELS]

x, w = np.arange(4), 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Before vs After SMOTE', fontsize=14, fontweight='bold')

ax1.bar(x - w/2, acc_b, w, label='Before SMOTE', color='#2196F3', alpha=0.85)
ax1.bar(x + w/2, acc_a, w, label='After SMOTE',  color='#FF9800', alpha=0.85)
ax1.set_title('Accuracy'); ax1.set_xticks(x); ax1.set_xticklabels(short_names)
ax1.set_ylim(0.8, 1.02); ax1.legend(); ax1.set_ylabel('Accuracy')
for i, (b, a) in enumerate(zip(acc_b, acc_a)):
    ax1.text(i - w/2, b + 0.003, f'{b:.3f}', ha='center', fontsize=8)
    ax1.text(i + w/2, a + 0.003, f'{a:.3f}', ha='center', fontsize=8)

ax2.bar(x - w/2, rec_b, w, label='Before SMOTE', color='#E53935', alpha=0.85)
ax2.bar(x + w/2, rec_a, w, label='After SMOTE',  color='#4CAF50', alpha=0.85)
ax2.set_title('Spam Recall'); ax2.set_xticks(x); ax2.set_xticklabels(short_names)
ax2.set_ylim(0, 1.1); ax2.legend(); ax2.set_ylabel('Recall')
for i, (b, a) in enumerate(zip(rec_b, rec_a)):
    ax2.text(i - w/2, b + 0.02, f'{b:.3f}', ha='center', fontsize=8)
    ax2.text(i + w/2, a + 0.02, f'{a:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[9] Saved: comparison_chart.png")


# ─────────────────────────────────────────────
# 10. SAVE MODEL + RESULTS  →  model.pkl / results.pkl
# ─────────────────────────────────────────────

# Best model = highest accuracy after SMOTE (LR wins at 98.1%)
best_name  = max(results_after, key=lambda n: results_after[n]['acc'])
best_clf   = results_after[best_name]['clf']

# For NB we used abs(X) — store a flag so app.py knows
use_abs = 'NB' in best_name

with open('model.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'model':      best_clf,
        'use_abs':    use_abs,
        'best_name':  best_name,
    }, f)

# Save metrics summary for the Streamlit sidebar
summary = {
    'before': {n: {'acc':    results_before[n]['acc'],
                   'prec':   results_before[n]['report']['Spam']['precision'],
                   'recall': results_before[n]['report']['Spam']['recall'],
                   'f1':     results_before[n]['report']['Spam']['f1-score'],
                   'cm':     results_before[n]['cm'].tolist()}
               for n in MODELS},
    'after':  {n: {'acc':    results_after[n]['acc'],
                   'prec':   results_after[n]['report']['Spam']['precision'],
                   'recall': results_after[n]['report']['Spam']['recall'],
                   'f1':     results_after[n]['report']['Spam']['f1-score'],
                   'cm':     results_after[n]['cm'].tolist()}
               for n in MODELS},
}

with open('results.pkl', 'wb') as f:
    pickle.dump(summary, f)

print(f"\n[10] Best model  : {best_name}")
print(f"     Accuracy     : {results_after[best_name]['acc']:.4f}")
print(f"     Saved        : model.pkl  |  results.pkl")
print("\n" + "=" * 60)
print("DONE — now run:  streamlit run app.py")
print("=" * 60)
