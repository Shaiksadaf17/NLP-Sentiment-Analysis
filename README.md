# NLP Sentiment Analysis

This repository contains **two versions** of a sentiment analysis project on tweets using Python and SVM. The project was implemented as part of an NLP assignment and demonstrates text preprocessing, feature extraction, SVM classification, cross-validation, and error analysis.

---

## Project Overview

- **Goal:** Predict the sentiment (`positive` or `negative`) of tweets.
- **Dataset:** ~27,000 tweets in `sentiment-dataset.tsv` (tab-separated, labels in column 2, text in column 3).  
- **Tech Stack:** Python, NLTK, scikit-learn, matplotlib.

---

## Repository Structure

1. **Basic Version (`nlp_sentiment.py`)**
   
   - **Preprocessing:** Tokenization using `split()`.
   - **Features:** Unigram presence only.
   - **Classifier:** LinearSVC via NLTK's `SklearnClassifier`.
   - **Evaluation:** Cross-validation and error analysis (confusion matrix, false positives/negatives).

2. **Optimized Version (`nlp_sentiment_optimized.py`)**
   - Improved preprocessing and feature extraction.
   - **Preprocessing:**
     - Lowercasing
     - Punctuation removal
     - Stopwords removal
     - Lemmatization
   - **Features:**
     - Unigrams + bigrams
     - Word count (`num_words`)
   - **Classifier & Evaluation:** Same SVM, cross-validation, and error analysis.

3. **Dataset**
   - `sentiment-dataset.tsv` – tab-separated file containing tweet texts and labels.  
   *(Add your dataset here or link to the source if public.)*

---

## How to Run

1. Install dependencies:

```bash
pip install nltk scikit-learn matplotlib
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
python nlp_sentiment_q1_q3.py
python nlp_sentiment_optimized.py


