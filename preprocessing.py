import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, RegexpStemmer
from gensim.utils import simple_preprocess
import json
import nltk
import string
from gensim import corpora
import torch
import csv

# nltk.download('punkt')
# nltk.download('stopwords')
# Define stopwords and lemmatizer
STOP = set(stopwords.words('english'))
EXCLUDE = set(string.punctuation)
LEMMA = WordNetLemmatizer()
# STEMMER = PorterStemmer()
PATTERN = r's$|able$|ly$'
STEMMER = RegexpStemmer(PATTERN, min=4)


def clean(doc):
    non_letters_free = " ".join(re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', word)).strip().lower() for word in doc.split())
    stop_free = " ".join([i for i in non_letters_free.lower().split() if i not in STOP])
    punc_free = "".join(ch for ch in stop_free if ch not in EXCLUDE)
    stemmered = " ".join([STEMMER.stem(word) for word in punc_free.split()])
    long_short_free = " ".join(word for word in stemmered.split() if len(word) <= 12 and len(word) >= 2)  # Remove words longer than 10 or shorter than 2
    return long_short_free


def preprocess_text(text):
    tokens = [clean(doc).split() for doc in text]
    
    return tokens


def get_filtered_corpus(corpus):
    total_doc = len(corpus)

    # Creating document-term matrix
    dictionary = corpora.Dictionary(corpus)

    min_freq = total_doc * 0.0001
    max_freq = total_doc * 0.999
    dictionary.filter_extremes(no_below=min_freq, no_above=max_freq)

    dictionary.filter_tokens(good_ids=[dictionary.token2id[word] for word in words])

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


def get_filtered_vocabulary(corpus):
    # Creating document-term matrix
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_tokens(good_ids=[dictionary.token2id[word] for word in words])

    # Create a vocabulary set to collect unique words
    vocabulary = set()
    # Convert the tokenized corpus to a bag-of-words representation using the filtered dictionary
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in corpus]
    
    # Accumulate unique words into the vocabulary set
    for bow in bow_corpus:
        vocabulary.update(dictionary[word_id] for word_id, freq in bow)

    return vocabulary



def prepare_cleaner_data():
    # Read the list of lists from the file
    with open('docs.json', 'r') as file:
        docs = json.load(file)
    
    clean_corpus = preprocess_text(docs)
    return clean_corpus


def prepare_cleaner_csv_data(filename):
    sentences = []
    labels  = []
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter = ',')
        next(file, None)
        
        for row in file:
            row_n = clean(row[1])
            sentences.append(row_n.split())
            labels.append(row[2])

    return sentences, labels


def save_file_txt(path, list):
    with open(f'{path}.txt', 'w') as f:
        for item in list:
            f.write(f"{item}\n")


with open('BBCwords.txt', 'r') as file:
    words = file.read().splitlines()

FILE_PATH = "BBC/BBC_News_Train.csv"

def run_20NewsGroup():
    data = prepare_cleaner_data()
    print(data[:5])
    _, corpus = get_filtered_corpus(data)
    corpus_input = [doc.split() for doc in corpus]
    vocabFilter = get_filtered_vocabulary(corpus_input)

    print(f"Number of docs: {len(corpus)}")
    print(f"Number of words: {len(vocabFilter)}")

    save_file_txt("Results/clean_corpus", corpus)
    save_file_txt("Results/clean_vocab", vocabFilter)
 

def run_BBC(file):
    sentences, labels = prepare_cleaner_csv_data(file)
    print(sentences[:5])
    _, corpus = get_filtered_corpus(sentences)
    corpus_input = [doc.split() for doc in corpus]
    vocabFilter = get_filtered_vocabulary(corpus_input)

    print(f"Number of docs: {len(corpus)}")
    print(f"Number of words: {len(vocabFilter)}")

    save_file_txt("Results/BBCGlove/clean_corpus", corpus)
    save_file_txt("Results/BBCGlove/clean_vocab", vocabFilter)
    save_file_txt("Results/BBCGlove/labels", labels)

# run_BBC(FILE_PATH)
