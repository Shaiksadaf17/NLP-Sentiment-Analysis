#!/usr/bin/env python
# coding: utf-8

"""
NLP Sentiment Analysis (Optimized Version)
- Lowercasing, punctuation removal, stopwords removal, lemmatization
- Features: unigrams, bigrams, word count
- SVM classifier, cross-validation, error analysis
"""

import csv
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Global Variables
# --------------------------
raw_data = []
train_data = []
test_data = []
global_feature_dict = {}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --------------------------
# Preprocessing
# --------------------------
def pre_process(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

# --------------------------
# Feature Extraction
# --------------------------
def to_feature_vector(tokens, use_bigrams=True):
    feature_dict = {}
    for word in tokens:
        feature_dict[word] = 1
        global_feature_dict[word] = global_feature_dict.get(word, 0) + 1
    if use_bigrams:
        for bigram in ngrams(tokens, 2):
            bigram_str = "_".join(bigram)
            feature_dict[bigram_str] = 1
            global_feature_dict[bigram_str] = global_feature_dict.get(bigram_str, 0) + 1
    feature_dict["num_words"] = len(tokens)
    return feature_dict

# --------------------------
# Load Data
# --------------------------
def load_data(path):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            label, text = line[1], line[2]
            raw_data.append((text, label))

# --------------------------
# Split & Preprocess
# --------------------------
def split_and_preprocess_data(percentage=0.8):
    num_training_samples = int(len(raw_data) * percentage)
    for (text, label) in raw_data[:num_training_samples]:
        train_data.append((to_feature_vector(pre_process(text)), label))
    for (text, label) in raw_data[num_training_samples:]:
        test_data.append((to_feature_vector(pre_process(text)), label))

# --------------------------
# Train Classifier
# --------------------------
def train_classifier(data):
    pipeline = Pipeline([('svc', LinearSVC(max_iter=5000))])
    return SklearnClassifier(pipeline).train(data)

# --------------------------
# Predict Labels
# --------------------------
def predict_labels(samples, classifier):
    return classifier.classify_many(samples)

# --------------------------
# Confusion Matrix Heatmap
# --------------------------
def confusion_matrix_heatmap(y_test, preds, labels):
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="w")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    b, t = plt.ylim()
    plt.ylim(b+0.5, t-0.5)
    plt.show()

# --------------------------
# Cross-validation
# --------------------------
def cross_validate(dataset, folds=10):
    results = []
    fold_size = len(dataset) // folds
    for fold in range(folds):
        start = fold * fold_size
        end = start + fold_size if fold != folds - 1 else len(dataset)
        validation = dataset[start:end]
        training = dataset[:start] + dataset[end:]
        clf = train_classifier(training)
        X_val, y_val = zip(*validation)
        preds = predict_labels(X_val, clf)
        report = classification_report(y_val, preds, output_dict=True)
        results.append(report)
    avg_precision = np.mean([r['weighted avg']['precision'] for r in results])
    avg_recall = np.mean([r['weighted avg']['recall'] for r in results])
    avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print("Cross-validation (avg):", {"precision": avg_precision, "recall": avg_recall, "f1-score": avg_f1, "accuracy": avg_accuracy})

# --------------------------
# Error Analysis
# --------------------------
def error_analysis(X_test, y_test, clf):
    preds = predict_labels(X_test, clf)
    confusion_matrix_heatmap(y_test, preds, labels=clf.labels())
    print(classification_report(y_test, preds))
    
    false_positives = []
    false_negatives = []
    for text, true_label, pred_label in zip(X_test, y_test, preds):
        if pred_label == 'positive' and true_label == 'negative':
            false_positives.append(text)
        elif pred_label == 'negative' and true_label == 'positive':
            false_negatives.append(text)
    print("\nFalse Positives (Predicted Positive but Actually Negative):")
    for fp in false_positives[:5]:
        print("-", fp)
    print("\nFalse Negatives (Predicted Negative but Actually Positive):")
    for fn in false_negatives[:5]:
        print("-", fn)

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    data_file_path = 'sentiment-dataset.tsv'
    load_data(data_file_path)
    split_and_preprocess_data(0.8)

    fold_size = len(train_data) // 10
    train_split = train_data[fold_size:]
    test_split = train_data[:fold_size]

    clf = train_classifier(train_split)
    X_test, y_test = zip(*test_split)

    cross_validate(train_data, folds=10)
    error_analysis(X_test, y_test, clf)
