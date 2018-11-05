import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

"""
load training and test data from file. Return test/train datasets.
Make the entire sentences lower cased as the GloVe models vector is uncased.
"""
def loadData(fname, train_size = 5000, test_size = 500):
    print('loading training and test data sets from: ',fname)
    df = pd.read_csv(fname, delimiter=',', nrows=train_size+test_size)
    dataset = [(str(x[5]).lower(), int(x[0])) for x in df.values]



"""
Load pretrained GloVe model into a word -> vector dictionary.
"""
def loadGlove(fname):
    print("Loading GloVe model...")
    file = open(fname)
    mapping = {}
    for line in file:
        # Seperate line by spaces to construct individual elements of the vector.
        # Format is : "word 1.22 3.44 5.66", where first string is the word and the rest are the vector representation
        line_arr = line.split()
        word = line_arr[0]
        vector = np.array([float(val) for val in line_arr[1:]])
        mapping[word] = vector

    print('Completed loading GloVe model.')
    return mapping

def main():
    glove_fname = '/Users/issa/Downloads/glove.twitter.27B/glove.twitter.27B.50d.txt'
    sentiment_fname = '/Users/issa/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv'
    #mapping = loadGlove(fname)
    dataset = loadData(sentiment_fname)


if __name__ == '__main__':
    main()
