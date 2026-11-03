"""Python code exported from IR_Assignment_1_Text_Preprocessing.ipynb."""

# %% Cell 1
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Added to resolve LookupError

# %% Cell 3
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    return " ".join(filtered_text)

# %% Cell 5
def apply_stemming(text):
    ps = PorterStemmer()
    word_tokens = word_tokenize(text)
    stemmed_text = [ps.stem(w) for w in word_tokens]
    return " ".join(stemmed_text)

# %% Cell 7
example_text = "This is an example sentence, showing off the stop word removal and stemming functionalities of NLTK. Running and learning new things are great!"

print(f"Original Text: {example_text}\n")

# Stop word removal
text_without_stopwords = remove_stopwords(example_text)
print(f"After Stop Word Removal: {text_without_stopwords}\n")

# Stemming
stemmed_text = apply_stemming(text_without_stopwords)
print(f"After Stemming (on text without stopwords): {stemmed_text}")

# %% Cell 8

