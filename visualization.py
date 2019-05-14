import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import pickle



def load_embedding():
    # with open("../data/Word2Vec/weighted.pickle", 'rb') as file:
    with open("../data/Word2Vec/embeddin.pickle", 'rb') as file:
        result = pickle.load(file)
        print(result.shape)
    return result


def load_lyrics():
    with open('../data/mxm/lyrics.pickle', 'rb') as file:
        bow = pickle.load(file)
    return bow


def load_genre():
    genre_dict = {}
    genre_set = set()

    with open('../data/MSD/msd_tagtraum_cd2.cls', 'r') as fp:
        lines = fp.readlines()
        for i in range(7, len(lines)):
            split = lines[i].strip().split()
            genre_set.add(split[1])
            genre_dict[split[0]] = split[1]

    print("Genres: ")
    print(genre_set)
    return genre_set, genre_dict


def load_match():
    with open('../data/mxm/match.pickle', 'rb') as file:
        match = pickle.load(file)
    print(list(match.items())[:10])
    return match


def tsne(X, y):
    # print("---> Start TSNE")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("---> Start Plotting")
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(15, 15))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig("tsne.png")


if __name__=='__main__':
    bow = load_lyrics()
    genre_set, genre_dict = load_genre()
    match = load_match()

    genre_repr = dict()
    for i, genre in enumerate(genre_set):
        genre_repr[genre] = i

    embedding = load_embedding()

    X = list()
    y = list()

    print("---> INIT")
    print(embedding.shape)
    print(len(match))

    embedding = embedding[:5000, :]

    for i in range(embedding.shape[0]):
        track_id = match[i]
        if track_id in genre_dict:
            genre = genre_dict[track_id]
            if genre in genre_repr:
                X.append(list(embedding[i]))
                y.append(genre_repr[genre])

    print("---> AFTER")
    X = np.array(X)
    X = np.squeeze(X, axis=(1, 2))
    y = np.array(y)

    print(X.shape)
    print(y.shape)

    tsne(X, y)
