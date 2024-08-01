from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def extract_features(texts, max_features=5000, ngram_range=(1, 2)):
    """
    Extract TF-IDF features from text data.

    Args:
    texts (list): List of preprocessed text documents.
    max_features (int): Maximum number of features to extract.
    ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted.

    Returns:
    scipy.sparse.csr.csr_matrix: TF-IDF features matrix.
    list: Feature names (vocabulary).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    features = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    return features, feature_names


def analyze_top_features(feature_matrix, feature_names, top_n=10):
    """
    Analyze and print the top N features (words or phrases) based on their TF-IDF scores.

    Args:
    feature_matrix (scipy.sparse.csr.csr_matrix): TF-IDF features matrix.
    feature_names (list): List of feature names corresponding to the columns in feature_matrix.
    top_n (int): Number of top features to display for each document.
    """
    for i in range(feature_matrix.shape[0]):
        feature_scores = pd.Series(feature_matrix[i].toarray().reshape(-1), index=feature_names)
        top_features = feature_scores.nlargest(top_n)
        print(f"Top {top_n} features for document {i + 1}:")
        print(top_features)
        print("\n")


# Usage example
data = pd.read_csv('data/processed/resumes.csv')
features, feature_names = extract_features(data['processed_text'])

print(f"Shape of feature matrix: {features.shape}")
print(f"Number of unique features: {len(feature_names)}")

# Analyze top features for the first 5 documents
analyze_top_features(features[:5], feature_names)