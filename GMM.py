from sklearn.mixture import GaussianMixture
import torch
from spectralnet._cluster import SpectralNet
from spectralnet._utils import get_affinity_matrix
from sklearn.manifold import SpectralEmbedding
import numpy as np

def fit_gmm_model():
    Embedding = torch.load(f"{DIR}/embedding")
    print("Loaded Embed")
    gmm = GaussianMixture(n_components=TOP, n_init=1)
    gmm.fit(Embedding)
    print("Fit gmm to Embed")

    calculate_predictions(gmm, Embedding)
    print("Finish predicting gmm to Embed")

    # torch.save(gmm, "no_filter_gmm_model")


def calculate_predictions(gmm, Embedding):
    predictions = gmm.predict_proba(Embedding)
    torch.save(predictions, f"{DIR}/{TOP}_WE_predictions")



def fit_and_predict_SN_gmm_model():
    Embedding = torch.load(f"{DIR}/embedding")
    sn = SpectralNet(n_clusters=TOP)
    sn.fit(Embedding)
    predictions = sn.predict(Embedding)
    torch.save(predictions, f"{DIR}/{TOP}_SN_WE_predictions")



def fit_SE_and_gmm():
    Embedding = torch.load(f"{DIR}/embedding")
    # W = get_affinity_matrix(Embedding, 10, "cpu")
    se = SpectralEmbedding(n_components=50)
    embed = se.fit_transform(np.array(Embedding))
    torch.save(embed, "spectral_embed")
    gmm = GaussianMixture(n_components=TOP, n_init=1)
    gmm.fit(embed)
    calculate_predictions(gmm, embed)



DIR = "NewResults/20NewsGroup"
TOP = 100
fit_gmm_model()
# fit_and_predict_SN_gmm_model()
# fit_SE_and_gmm()
