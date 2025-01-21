import graph
from  matrix_utils import get_random_walk_laplacian, load_data, create_prior, apply_diffusion_operator


def main(whose_idea, metrics=["pmi"]):
    path = "ProcessedData/Trump'sTweets"
    save_prior_path = f"WordsGraph/{CURRENT_DIR}/priors"
    dictionary, _, _, corpus = load_data(path)
    k_neighbors = 100
    use_exp = False

    for m in metrics:
        words_graph = graph.Graph(dictionary=dictionary, corpus=corpus, metric=m, whose_idea=whose_idea, similarity=False)
        affinity_matrix = words_graph.get_graph_as_affinity_matrix(k=k_neighbors)
        if whose_idea == "tal":
            _, matrix = get_random_walk_laplacian(affinity_matrix, k=500)
        else:
            _, matrix = apply_diffusion_operator(affinity_matrix, use_exp=use_exp, k=k_neighbors)
        create_prior(save_prior_path, matrix, n_components=200, to_save=True, metric=m, whose_idea=whose_idea)


if __name__ == "__main__":
    import sys
    argv = sys.argv

    argc = len(argv)    
    if argc > 1:
        CURRENT_DIR = argv[1]
        whose_id = argv[2]
        metrics = argv[3:]
        print("Directory: {}\nWhose idea: {}\nmetrics: {}".format(CURRENT_DIR, whose_id, metrics))

        main(metrics=metrics, whose_idea=whose_id)

    else:
        CURRENT_DIR = "Uri/Neighbours"
        whose_id = "uri"
        main(whose_idea=whose_id)
