import os
import string
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
# nltk.download()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize into words
    words = word_tokenize(text)
    # Remove stop words and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word not in stop_words]
    return cleaned_words

def chunk_with_overlap(tokens, window_size=150, overlap=50):
    chunks = []
    for i in range(0, len(tokens), window_size - overlap):
        chunk = tokens[i:i + window_size]
        if len(chunk) > 0:
            chunks.append(' '.join(chunk))
    return chunks


def process_documents(folder_path, window_size=75, overlap=50):
    cleaned_documents = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                # clean and tokenize the text
                cleaned_tokens = clean_text(text)
                # Chunk the tokens into fixed-size windows
                chunks = chunk_with_overlap(cleaned_tokens, window_size, overlap)
                # Store the cleaned chunks
                cleaned_documents[filename] = chunks

    return cleaned_documents

# Example usage
folder = "datasets"
cleaned_data = process_documents(folder)

# Optional preview
for doc, sentences in cleaned_data.items():
    print(f"\nDocument: {doc}")
    for s in sentences[:3]:
        print(" -", s)

# Save to file
with open("outputs/task1_cleaned_window_75.json", "w", encoding="utf-8") as out_file:
    json.dump(cleaned_data, out_file, indent=2, ensure_ascii=False)

print("✅ Cleaned data saved to outputs/task1_cleaned_window_75.json")
