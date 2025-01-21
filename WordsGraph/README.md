# Results So Far

### eta 10
Results were worst!
Continue with eta=1.

### main_Tal_idea:

1. Construct pmi/npmi graph where negative values are treated as positive - if value < 0, value = -value.
2. Use gaussian kernel - give higher values to similar words and lower values to more different words.
3. Apply RW laplacian - multipy eigvec by correspond eigval -> embedding.
4. Apply PCA with n_components=50.
5. Apply GMM over PCA output.

This is the words by topic - prior for LDA.

### main_uri:

# Use 20 k_neighbours. 
- k=50 - didnt improve results.
- k=100 - improved results for 1. npmi with PCA before gmm(top 10 - 49% WI). 2. pmi regular algoritm (top 10 - 47% WI).
1. 