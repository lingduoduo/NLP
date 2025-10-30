import re
import string
import nltk
from typing import List
import os
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.data import find

# Set NLTK data path to a writable location in Databricks
current_dir = Path.cwd()
nltk_data_dir = current_dir / 'nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)  
nltk.data.path.append(str(nltk_data_dir))

# Download required NLTK resources safely (only if missing)
NLTK_RESOURCES = [
    "corpora/stopwords",
    "corpora/wordnet",
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
]
for resource in NLTK_RESOURCES:
    resource_name = resource.split("/", 1)[1]
    if any(os.path.exists(os.path.join(path, resource)) 
           for path in nltk.data.path):
        continue
    nltk.download(resource_name, download_dir=str(nltk_data_dir), quiet=True)

class CustomTextSplitter:
    """A custom text splitter that extracts top keywords from each segment using NLTK."""

    def __init__(self, separator: str, **kwargs):
        """
        Initialize with a separator string and number of keywords to extract per segment.
        :param separator: The string to split the text on.
        """
        self._separator = separator
        self._stopwords = set(stopwords.words('english'))
        self._punct_table = str.maketrans('', '', string.punctuation)

    def normalize_text(self, text: str):
        # Replace line breaks (both types) with spaces
        text = re.sub(r'[\n\r]', ' ', text)

        # Replace special characters
        _special_chars = r"(\'|\"|\.|\,|\;|\<|\>|\{|\}|\[|\]|\"|\'|\=|\~|\*|\:|\#|\+|\^|\$|\@|\%|\!|\&|\)|\(|/|\-|\\)"
        text = re.sub(_special_chars, ' ', text)
        
        # Replace URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Replace HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Replace Email addresses
        text = re.sub(r'b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)

        # Replace user mentions
        text = re.sub(r'@(\w+)', ' ', text)

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Replace two or more subsequent white spaces with a single space
        text = re.sub(r'\ {2,}', ' ', text)

        # Trim white spaces at the beginning or end
        text = text.strip()
        return ' ' if text is None else text

    def split_text(self, text: str) -> List[str]:
        """
        Split the input text using the separator, extract keywords from each segment,
        and return a list of comma-separated keyword strings per segment.
        :param text: The full text to split and analyze.
        :return: List of keyword strings for each segment.
        """
        segments = text.split(self._separator)
        keyword_chunks: List[str] = []
        for segment in segments:
            tokens = word_tokenize(segment.lower())
            cleaned = [t.translate(self._punct_table) for t in tokens]
            filtered_tokens = [self.normalize_text(t) for t in cleaned if t.isalpha() and t not in self._stopwords]

            # Initialize the stemmer and lemmatizer
            # stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            # processed_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in filtered_tokens]
            processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
            keyword_chunks.append(",".join(processed_tokens))

        return keyword_chunks