# Create a graph class that will hold word nodes and their relationships.

import networkx as nx
from collections import Counter
from itertools import islice
from scipy.sparse import lil_matrix
import numpy as np
from gensim import corpora


class Graph(nx.Graph):

    def __init__(self, dictionary, corpus, similarity=False, is_ppmi=False, no_word2vec_model=False, window_size=3, metric="pmi", alpha=0.7):
        """
        is_ppmi: if pmi value is negative change value to 0. max(pmi, 0). - tried but didn't influence performance.
        """
        super().__init__()

        self.dictionary = dictionary
        self.corpus = corpus
        self.window_size = window_size
        self.alpha = alpha
        self.similarity = similarity
        self.is_ppmi = is_ppmi

        if similarity:
            from gensim.models import Word2Vec
            if no_word2vec_model:
                self.model = Word2Vec(self.corpus, vector_size=100, window=self.window_size, min_count=1, workers=4, sg=0)
                self.model.save("/home/dsi/ishonta/TM/WordsGraph/Word2VecModel/word2vec.model")
            else:
                self.model = Word2Vec.load("/home/dsi/ishonta/TM/WordsGraph/Word2VecModel/word2vec.model")
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
        """
        Construct the edges of the graph according to chosen metric.
        """
        weights = []
        joint_probs = []
        # Add edges with PMI as weights
        for (word1, word2), count in bigram_counts.items():
            joint_prob = count / total_co_occurrences
            word_id1 = self.dictionary.token2id[word1]
            word_id2 = self.dictionary.token2id[word2]
            weight = np.log(joint_prob / (unigram_probs[word_id1] * unigram_probs[word_id2]))
            weights.append(weight)
            joint_probs.append(joint_prob)
        
        max_weight = max(weights)

        for i, (word1, word2) in enumerate(bigram_counts.keys()):
            joint_prob = joint_probs[i]
            word_id1 = self.dictionary.token2id[word1]
            word_id2 = self.dictionary.token2id[word2]
            if 'npmi' in metric:
                weight = weights[i] / -np.log(joint_prob)
            else:
                weight = weights[i]

            if weight < 0: # Make sure positive and negative correlation are meaningfull in graph
                if self.is_ppmi:
                    weight = 0
                else:
                    weight = -weight
            
            if self.similarity:
                if "npmi" in metric:
                    weight = self.alpha * weight + (1 - self.alpha) * self._norm_cosine_similarity(word1, word2)
                else:
                    # Add scaling to cosine similarity to make it more impactfull over pmi range values
                    weight = self.alpha * weight + (1 - self.alpha) * (self._norm_cosine_similarity(word1, word2) * max_weight)


            self.add_edge(word_id1, word_id2, weight=weight)
    

    def _norm_cosine_similarity(self, word1, word2):
        """
        Calculate the normalized cosine similarity between two words.
        Returns a value in the range [0, 1].
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec1 = self.model.wv[word1]
        vec2 = self.model.wv[word2]

        cos_similaity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        norm_cos_similarity = (cos_similaity + 1) / 2  # Normalize to [0, 1] 

        return norm_cos_similarity


    def _calculate_co_occurrences(self):
        """
        Calculate the co-occurrences of words in the corpus.
        """
        # Initialize counters for bigrams
        bigram_counts = Counter()

        # Calculate bigram counts using a sliding window
        for sentence in self.corpus:
            for i, word_id in enumerate(sentence):
                window = islice(sentence, i + 1, min(i + 1 + self.window_size, len(sentence)))
                for context_word in window:
                    if context_word != word_id:
                        tuple_id = sorted((word_id, context_word))
                        bigram_counts[(tuple_id[0], tuple_id[1])] += 1
                        
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



