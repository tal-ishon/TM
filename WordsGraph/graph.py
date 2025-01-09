# Create a graph class that will hold word nodes and their relationships.

import networkx as nx
from collections import Counter
from itertools import islice
from scipy.sparse import lil_matrix
import numpy as np
from gensim import corpora


class Graph(nx.Graph):

    def __init__(self, dictionary, corpus, window_size=3, metric="pmi"):
        super().__init__()

        self.dictionary = dictionary
        self.corpus = corpus
        self.window_size = window_size

        self._build_graph(metric=metric)


    def _build_graph(self, metric):
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
        unigram_probs, unigram_counts = self._calculate_unigram_probs()
        bigram_counts, total_co_occurrences = self._calculate_co_occurrences()

        self._construct_graph_edges(unigram_probs, bigram_counts, total_co_occurrences, metric=metric)

        self._deal_with_non_connectivity(unigram_counts, bigram_counts)


    def _calculate_unigram_probs(self):
        """
        Calculate the probabilities of each word in the corpus.
        """
        unigram_counts = self.dictionary.cfs
        total_word_count = sum(unigram_counts.values())
        unigram_probs = {word_id: count / total_word_count for word_id, count in unigram_counts.items()}

        return unigram_probs, unigram_counts
    

    def _construct_graph_edges(self, unigram_probs, bigram_counts, total_co_occurrences, metric="pmi"):

        # Add edges with PMI as weights
        for (word_id1, word_id2), count in bigram_counts.items():
            joint_prob = count / total_co_occurrences
            word_id1 = self.dictionary.token2id[word_id1]
            word_id2 = self.dictionary.token2id[word_id2]
            weight = np.log(joint_prob / (unigram_probs[word_id1] * unigram_probs[word_id2]))

            if metric == 'npmi':
                weight = weight / -np.log(joint_prob)

            if weight < 0: # Make sure positive and negative correlation are meaningfull in graph
                weight = -weight

            self.add_edge(word_id1, word_id2, weight=weight)
    

    def _calculate_co_occurrences(self):
        """
        Calculate the co-occurrences of words in the corpus.
        """
        # Initialize counters for bigrams
        bigram_counts = Counter()

        # Calculate bigram counts using a sliding window
        for sentence in self.corpus:
            for i, word_id in enumerate(sentence):
                window = islice(sentence, max(i - self.window_size, 0), min(i + self.window_size + 1, len(sentence)))
                for context_word in window:
                    if context_word != word_id:
                        bigram_counts[(word_id, context_word)] += 1

        # Total co-occurrences
        total_co_occurrences = sum(bigram_counts.values())

        return bigram_counts, total_co_occurrences
    

    def _deal_with_non_connectivity(self, unigram_counts, bigram_counts):
        """
        If There are any nodes without edges - make sure to add them to graph.
        """
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



