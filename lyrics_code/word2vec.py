import gensim
from gensim.models.keyedvectors import KeyedVectors
import pickle
import sys
import numpy as np


def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    """
    Create progress bar for large task
    """
    percent = round(progress / float(total) * 100, 2)
    buf = "{0}|{1}| {2}{3}/{4} {5}%".format(lbar_prefix, ('#' * round(percent)).ljust(100, '-'),
        rbar_prefix, progress, total, percent)
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()


def matching():

    print("Loading Words...")
    words = None
    with open(train_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] == '%':
                words = line.strip()[1:].split(',')
                break

    mapping = dict()
    with open("../data/Word2Vec/reverse_mapping.txt", 'r') as file:
        for pair in file.readlines():
            pair = pair.replace('\xad', '').replace('\n', '')
            key, value = pair.split("<SEP>")
            if key != value:
                mapping[str(key)] = str(value)

    print("Loading Model...")
    model = KeyedVectors.load_word2vec_format('../data/Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)

    w2v = dict()
    non_match = list()

    print("Embedding Matching...")
    embedding = list()
    total = len(words)
    for i, word in enumerate(words):
        if word in model.vocab:
            vector = model[word]
            w2v[word] = vector
            embedding.append(vector)
        elif word in mapping and mapping[word] in model.vocab:
            word = mapping[word]
            vector = model[word]
            w2v[word] = vector
            embedding.append(vector)
        else:
            non_match.append(word)
            embedding.append(np.zeros([300,]))
        report_progress(i, total)
    embedding = np.matrix(embedding)

    with open("../data/Word2Vec/matching.pickle", 'wb') as file:
        pickle.dump(w2v, file)

    with open("../data/Word2Vec/non_match.pickle", 'wb') as file:
        pickle.dump(non_match, file)

    with open("../data/Word2Vec/embedding.pickle", 'wb') as file:
        pickle.dump(embedding, file)


    print("Total: [{}], Match: [{}], Non-Match: [{}]".format(len(words), len(w2v), len(non_match)))


def check(non_match_only=False):
    w2v, non_match = None, None
    if not non_match_only:
        print("Matchings: ")
        with open("../data/Word2Vec/matching.pickle", 'rb') as file:
            w2v = pickle.load(file)
            w2v = list(w2v.items())
            print(w2v[0])
            print(w2v[0][1].shape)

    print("\nNon-Matching: ")
    with open("../data/Word2Vec/non_match.pickle", 'rb') as file:
        non_match = pickle.load(file)
        print(non_match)

    print("\nEmbedding: ")
    with open("../data/Word2Vec/embedding.pickle", 'rb') as file:
        embedding = pickle.load(file)
        print(embedding[:10, :])
        print(embedding.shape)


    return w2v, non_match, embedding




if __name__=='__main__':
    train_file = "../data/mxm/mxm_dataset_train.txt"

    matching()
    check()
