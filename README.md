# NLP Sentiment Analysis

This repository contains **two versions** of a sentiment analysis project on tweets using Python and SVM. It demonstrates text preprocessing, feature extraction, SVM classification, cross-validation, and error analysis.

---

## Project Overview

- **Goal:** Predict the sentiment (`positive` or `negative`) of tweets.  
- **Dataset:** ~27,000 tweets in `sentiment-dataset.tsv` (tab-separated, labels in column 2, text in column 3).  
- **Tech Stack:** Python, NLTK, scikit-learn, matplotlib.

---

## Repository Structure

1. **Basic Version (`nlp_sentiment.py`)**
   
   - **Preprocessing:** Simple tokenization using `split()`.  
   - **Features:** Unigram presence only.  
   - **Classifier:** LinearSVC via NLTK's `SklearnClassifier`.  
   - **Evaluation:** Cross-validation and error analysis (confusion matrix, false positives/negatives).  

2. **Optimized Version (`nlp_sentiment_optimized.py`)**
   
   - **Preprocessing:**
     - Lowercasing  
     - Punctuation removal  
     - Stopwords removal  
     - Lemmatization  
   - **Features:**  
     - Unigrams + bigrams  
     - Word count (`num_words`)  
   - **Classifier & Evaluation:** Same SVM, cross-validation, and error analysis with improved performance.

3. **Dataset**
   - `sentiment-dataset.tsv`  
   

---

## How to Run

1. Install dependencies:

```bash
pip install nltk scikit-learn matplotlib
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
python nlp_sentiment.py
python nlp_sentiment_optimized.py


