import spacy
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Get English stopwords
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocess the input text.

    Args:
    text (str): Input text to preprocess.

    Returns:
    str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Tokenize and process with spaCy
    doc = nlp(text)

    # Lemmatize, remove stopwords and non-alphabetic tokens
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]

    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# Example usage
raw_text = "The quick brown fox jumps over the lazy dog."
preprocessed_text = preprocess_text(raw_text)
print(f"Original: {raw_text}")
print(f"Preprocessed: {preprocessed_text}")