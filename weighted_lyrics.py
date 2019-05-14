import gensim
from gensim.models.keyedvectors import KeyedVectors
import pickle
import sys
import numpy as np
from sklearn.preprocessing import normalize


def calc():
    train_file = "../data/mxm/mxm_dataset_train.txt"

    print("Load Lyrics...")
    with open('../data/mxm/lyrics.pickle', 'rb') as file:
        bow = pickle.load(file)
        song_ids = list(bow.keys())
        bows = list(bow.values())
        bows = np.matrix(bows)

    with np.errstate(divide='ignore', invalid='ignore'):
        bows = np.true_divide(1, bows)
        bows[bows == np.inf] = 0
        bows = np.nan_to_num(bows)
    bows = normalize(bows, axis=1, norm='l1')
    print(bows.shape)

    embedding = list()
    print("Loading Embedding...")
    with open("../data/Word2Vec/embedding.pickle", 'rb') as file:
        embedding = pickle.load(file)

    print(embedding.shape)
    print("Calculating Results...")
    result = np.matmul(bows, embedding)

    print("Saving Results...")
    with open("../data/Word2Vec/weighted.pickle", 'wb') as file:
        pickle.dump(result, file)

def check():
    with open("../data/Word2Vec/weighted.pickle", 'rb') as file:
        result = pickle.load(file)
        print(result.shape)
        print(result[:10, :])

if __name__=='__main__':
    # calc()
    check()
