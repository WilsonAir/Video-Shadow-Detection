from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
import torch

digits = datasets.load_digits(n_class=2)
X = digits.data
Y = digits.target

def tsne(X, Y):
    xx=[]

    for i in range(0,len(X)):
        xx.append((X[i] - np.min(X)) / (np.max(X) - np.min(X)))
    X=xx

    # X = X[0].reshape(1024,-1)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))

    # target = np.random
    pca = decomposition.TruncatedSVD(n_components=2)
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    pca2=pca.fit(X)
    pca2.components_
    pca3=pca2.transform(X)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=1,n_iter=1500)
    X_tsne = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10,5))
    plt.subplot2grid((1,2), (0,0))
    plt.title('PRINCIPAL COMPONENTS ANALYSIS')
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target)
    plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=2)
    plt.title('t-SNE')
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target)
    plt.show()

    ## ORIGINAL DATA DIMENSIONS
    print('ORIGINAL DATA DIMENSION:',np.array(X).shape)

    ## DIMENSIONS AFTER t-SNE
    # print('DIMENSIONS AFTER t-SNE',np.array(X_tsne).shape)

# tsne(X,Y)


def pca(feature_map):

    # Reshape feature map to 2D (1024, 256) for PCA
    feature_map_2d = feature_map.view(feature_map.shape[1], -1).T

    # Initialize PCA to reduce to 1 component
    pca = decomposition.PCA(n_components=1)

    # Apply PCA
    single_channel = pca.fit_transform(feature_map_2d)

    # Reshape the result back to (1, 1, 16, 16)
    single_channel_feature_map = torch.tensor(single_channel.T).view(1, 1, 128, 128)



    return single_channel_feature_map