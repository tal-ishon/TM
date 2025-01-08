# Create a graph class that will hold word nodes and their relationships.

import networkx as nx
from collections import Counter
from itertools import islice
from scipy.sparse import lil_matrix
import numpy as np
from gensim import corpora


class Graph(nx.Graph):

    def __init__(self, dictionary, corpus, window_size=3):
        super().__init__()

        self.dictionary = dictionary
        self.corpus = corpus
        self.window_size = window_size

        self._build_graph()

    def _build_graph(self):
        """
        Construct a graph of word-word co-occurrence for a given corpus using Pointwise Mutual Information (PMI).
        
        Parameters:
            dictionary: gensim.corpora.Dictionary
                The dictionary of the corpus.
            corpus: list of list of str
                The tokenized corpus.
            window_size: int
                The size of the sliding window (default is 3).

        """
        self.dictionary = corpora.Dictionary(self.corpus)
        # Calculate unigram probabilities
        unigram_counts = self.dictionary.cfs
        total_word_count = sum(unigram_counts.values())
        unigram_probs = {word_id: count / total_word_count for word_id, count in unigram_counts.items()}

        # Initialize counters for bigrams
        bigram_counts = Counter()

        # Calculate bigram counts using a sliding window
        for sentence in self.corpus:
            for i, word_id in enumerate(sentence):
                 # Look only ahead to avoid double counting
                for j in range(i + 1, min(i + self.window_size + 1, len(sentence))):
                    context_word_id = sentence[j]
                    if word_id != context_word_id:
                        bigram_counts[(word_id, context_word_id)] += 1

        # Total co-occurrences
        total_co_occurrences = sum(bigram_counts.values())

        # Add edges with PMI as weights
        for (word_id1, word_id2), count in bigram_counts.items():
            joint_prob = count / total_co_occurrences
            pmi_weight = np.log(joint_prob / (unigram_probs[self.dictionary.token2id[word_id1]] * unigram_probs[self.dictionary.token2id[word_id2]]))
            if pmi_weight < 0:
                pmi_weight = -pmi_weight

            self.add_edge(self.dictionary.token2id[word_id1], self.dictionary.token2id[word_id2], weight=pmi_weight)

        # Extract all unique word_ids in bigram_counts
        bigram_words_id = {self.dictionary.token2id[word] for pair in bigram_counts.keys() for word in pair}

        # Find word_ids that are in unigram_counts but not in bigram_words
        lonly_words = [word_id for word_id in unigram_counts.keys() if word_id not in bigram_words_id]

        self._add_lonly_nodes(lonly_words)


    def _add_lonly_nodes(self, lonly_nodes):
        for node_id in lonly_nodes:
            self.add_node(node_id, label=self.dictionary[node_id])

    
    def update_window_size(self, size):
        self.window_size = size


    def get_graph_as_affinity_matrix(self):
        """
        Take the graph and return an affinity matrix.
        """
        # Compose the affinity matrix obtained from the graph
        nodes = list(self.nodes)
        n = len(nodes)
        
        affinity_matrix = lil_matrix((n, n), dtype=float)

        for node1, node2, data in self.edges(data=True):
            i, j = node1, node2
            affinity_matrix[i, j] = data['weight']
            affinity_matrix[j, i] = data['weight']  # Ensure the matrix is symmetric

        return affinity_matrix



