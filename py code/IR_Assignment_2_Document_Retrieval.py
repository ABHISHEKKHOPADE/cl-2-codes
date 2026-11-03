"""Python code exported from IR_Assignment_2_Document_Retrieval.ipynb."""

# %% Cell 1
documents = {
    'doc1': "The quick brown fox jumps over the lazy dog. The dog is lazy.",
    'doc2': "A brown cat sat on the mat. The cat is quick.",
    'doc3': "Foxes are clever. Dogs are loyal. Cats are independent.",
    'doc4': "Quick dogs and lazy cats are good pets."
}

print("Sample Documents:")
for doc_id, content in documents.items():
    print(f"{doc_id}: {content}")

# %% Cell 2
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stop words if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and convert to lowercase
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

processed_documents = {}
for doc_id, content in documents.items():
    processed_documents[doc_id] = preprocess_text(content)

print("Processed Documents:")
for doc_id, tokens in processed_documents.items():
    print(f"{doc_id}: {tokens}")

# %% Cell 4
inverted_index = {}

for doc_id, tokens in processed_documents.items():
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = set()
        inverted_index[token].add(doc_id)

print("Inverted Index:")
for term, doc_ids in inverted_index.items():
    print(f"'{term}': {sorted(list(doc_ids))}")

# %% Cell 5
def search_inverted_index(query, inverted_index, documents):
    processed_query = preprocess_text(query)
    print(f"Processed query: {processed_query}")

    if not processed_query:
        return []

    # Start with the document set for the first term
    # If the term is not in the index, start with an empty set
    results = inverted_index.get(processed_query[0], set()).copy()

    # Intersect with results for subsequent terms (AND logic)
    for term in processed_query[1:]:
        if not results: # Optimization: if results are already empty, no need to check further
            break
        term_docs = inverted_index.get(term, set())
        results = results.intersection(term_docs)

    if not results:
        return []
    else:
        # Return the actual document content for found doc_ids
        found_docs_content = {doc_id: documents[doc_id] for doc_id in sorted(list(results))}
        return found_docs_content

# Test the search function
query1 = "quick brown fox"
retrieved_docs1 = search_inverted_index(query1, inverted_index, documents)
print(f"\nDocuments for query '{query1}':")
if retrieved_docs1:
    for doc_id, content in retrieved_docs1.items():
        print(f"  {doc_id}: {content}")
else:
    print("  No documents found.")

query2 = "lazy cats"
retrieved_docs2 = search_inverted_index(query2, inverted_index, documents)
print(f"\nDocuments for query '{query2}':")
if retrieved_docs2:
    for doc_id, content in retrieved_docs2.items():
        print(f"  {doc_id}: {content}")
else:
    print("  No documents found.")

query3 = "elephant"
retrieved_docs3 = search_inverted_index(query3, inverted_index, documents)
print(f"\nDocuments for query '{query3}':")
if retrieved_docs3:
    for doc_id, content in retrieved_docs3.items():
        print(f"  {doc_id}: {content}")
else:
    print("  No documents found.")
