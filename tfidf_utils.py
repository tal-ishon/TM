import json
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
import string
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# remove stopwords, punctuation, and normalize the corpus
STOP = set(stopwords.words('english'))
EXCLUDE = set(string.punctuation)
LEMMA = WordNetLemmatizer()



def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in STOP])
    punc_free = "".join(ch for ch in stop_free if ch not in EXCLUDE)
    normalized = " ".join(LEMMA.lemmatize(word) for word in punc_free.split())
    return normalized


def prepare_data():
    # Read the list of lists from the file
    with open('docs.json', 'r') as file:
        docs = json.load(file)
    clean_corpus = [clean(doc).split() for doc in docs]
    return clean_corpus


def get_filtered_vocabulary(corpus):
    total_doc = len(corpus)

    # Creating document-term matrix
    dictionary = corpora.Dictionary(corpus)

    min_freq = total_doc * 0.001
    max_freq = total_doc * 0.99
    dictionary.filter_extremes(no_below=min_freq, no_above=max_freq)
    dictionary.compactify()

    # Create a vocabulary set to collect unique words
    vocabulary = set()
    # Convert the tokenized corpus to a bag-of-words representation using the filtered dictionary
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in corpus]
    
    # Accumulate unique words into the vocabulary set
    for bow in bow_corpus:
        vocabulary.update(dictionary[word_id] for word_id, freq in bow)

    return vocabulary


def get_filtered_corpus(corpus):
    total_doc = len(corpus)

    # Creating document-term matrix
    dictionary = corpora.Dictionary(corpus)

    min_freq = total_doc * 0.0001
    max_freq = total_doc * 1
    dictionary.filter_extremes(no_below=min_freq, no_above=max_freq)
    dictionary.compactify()

    # Convert the tokenized corpus to a bag-of-words representation using the filtered dictionary
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in corpus]
    
    # Convert back to text for visualization (optional) ensuring no duplicates
    filtered_corpus = [
        ' '.join(dictionary[word_id] for word_id, freq in bow)
        for bow in bow_corpus
    ]
    
    word_to_ix = dictionary.token2id

    return word_to_ix, filtered_corpus


def get_corpus(data):
    dictionary = corpora.Dictionary(data)

    # Convert the tokenized corpus to a bag-of-words representation using the filtered dictionary
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in data]
    
    # Convert back to text for visualization (optional) ensuring no duplicates
    corpus = [
        ' '.join(sorted(dictionary[word_id] for word_id, freq in bow))
        for bow in bow_corpus
    ]
    
    return corpus


def run_SN_model():
    spectralNet = torch.load("SpectralNet_filtered_corpus")
    X = torch.load("data_to_fit_sn")
    spectralNet.fit()


# Generate the TF-IDF matrix
def custom_tokenizer(text):
    return text.split()  # Split on whitespace, keeping all tokens


def generate_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        preprocessor=lambda x: x,  # Don't preprocess
        token_pattern=None  # Don't use token pattern as we have a custom tokenizer
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def load_file_txt(path):
    with open(f'{path}.txt', 'r') as f:
        loaded_list = f.read().splitlines()
    
    return loaded_list


def save_file_txt(path, list):
    with open(f'{path}.txt', 'w') as f:
        for item in list:
            f.write(f"{item}\n")