import gensim
from gensim.models.keyedvectors import KeyedVectors
import pickle
import sys
import numpy as np


def calc():
    train_file = "../data/mxm/mxm_dataset_train.txt"

    print("Load Lyrics...")
    with open('../data/mxm/lyrics.pickle', 'rb') as file:
        bow = pickle.load(file)
        song_ids = list(bow.keys())
        bows = list(bow.values())
        bows = np.matrix(bows)

    print("Loading Model...")
    model = KeyedVectors.load_word2vec_format('../data/Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)

    embedding = list()
    print("Loading Embedding...")
    with open("../data/Word2Vec/embedding.pickle", 'rb') as file:
        embedding = pickle.load(file)

    print("Calculating Results...")
    result = np.matmul(bows, embedding)

    print("Saving Results...")
    # dict_result = {song_ids[i]: result[i] for i in range(len(song_ids))}
    with open("../data/Word2Vec/embedding.pickle", 'wb') as file:
        pickle.dump(result, file)
    with open("../data/Word2Vec/embedding_ids.pickle", 'wb') as file:
        pickle.dump(song_ids, file)
    # print(dict_result)

def check():
    with open("../data/Word2Vec/embedding.pickle", 'rb') as file:
        result = pickle.load(file)
        print(result.shape)
        print(result[:10, :])

if __name__=='__main__':
    calc()
    check()
