```python
"""Python code exported from IR_Assignment_1_Text_Preprocessing.ipynb."""

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 1. Stop Word Removal
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    filtered_text = [
        w for w in word_tokens
        if w.lower() not in stop_words
    ]

    return " ".join(filtered_text)


# 2. Stemming
def apply_stemming(text):
    ps = PorterStemmer()

    word_tokens = word_tokenize(text)

    stemmed_text = [
        ps.stem(w) for w in word_tokens
    ]

    return " ".join(stemmed_text)


# 3. Lemmatization
def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    word_tokens = word_tokenize(text)

    lemmatized_text = [
        lemmatizer.lemmatize(w.lower(), pos='v')
        for w in word_tokens
    ]

    return " ".join(lemmatized_text)


# Main Program
example_text = "This is an example sentence, showing off the stop word removal and stemming functionalities of NLTK. Running and learning new things are great!"

print(f"Original Text: {example_text}\n")


# Stop Word Removal
text_without_stopwords = remove_stopwords(example_text)

print(f"After Stop Word Removal: {text_without_stopwords}\n")


# Stemming
stemmed_text = apply_stemming(text_without_stopwords)

print(f"After Stemming: {stemmed_text}\n")


# Lemmatization
lemmatized_text = apply_lemmatization(text_without_stopwords)

print(f"After Lemmatization: {lemmatized_text}")
```