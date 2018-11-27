import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
#np.set_printoptions(threshold=nan)

"""
load training and test data from file. Return test/train datasets.
Make the entire sentences lower cased as the GloVe models vector is uncased.
"""
print(os.getcwd())

def loadData(fname, train_size = 5000, test_size = 500):
    print('loading training and test data sets from: ',fname)
    num_lines = len(list(open(fname, encoding="latin1")))
    print('number of lines: ',num_lines)

    df = pd.read_csv(fname, delimiter=',', encoding="latin1")
    #df = df.dropna(subset=['x'])
    #np.random.shuffle(df) # to get a random distribution of positive and negative sentiment values
    #dataset = [(str(x[5]).lower().split(), int(x[0])) for x in df.values]
    docs = []
    sentiments = []

    values = df.get_values()
    np.random.shuffle(values)

    for x in values:
        docs.append(str(x[5]))
        sentiments.append(int(x[0]))

    return docs[:train_size+test_size], sentiments[:train_size+test_size]

def construct_vector_data(docs, sentiments, embedding):
    train_x = [[embedding.get(word) if word in embedding else np.zeros(25) for word in doc] for doc in docs]
    print('Outputting first tweet ...')
    train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=55, padding='pre', truncating='pre')
    print('we got past train_x, onto train y...')
    train_y = np.asarray([1 if x==4 else 0 for x in sentiments], dtype=np.int32)
    print("train_x length: ",len(train_x), ", train_y length: ",len(train_y))
    #print(train_x[1])
    """
    print(', '.join([str(x) for x in train_x[1]]))
    print("target value: ", train_y[1])
    print("train_x shape: ",train_x.shape)
    print("train_y shape: ",train_y.shape)
    """
    return train_x, train_y


"""
Load pretrained GloVe model into a word -> vector dictionary.
"""
def loadGlove(fname):
    print("Loading GloVe model...")
    print(os.getcwd())
    print(fname)
    file = open(fname, encoding="utf8")
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

    glove_fname = '/home/issahassan/project/datasets/glove_embeddings/glove.twitter.27B.25d.txt'
    sentiment_fname = '/home/issahassan/project/datasets/sentiment140/training.1600000.processed.noemoticon.csv'
    train_size = 1000000
    test_size = 10000

    mapping = loadGlove(glove_fname)
    docs, sentiments = loadData(sentiment_fname, train_size, test_size)
    data_x, data_y = construct_vector_data(docs, sentiments, mapping)
    train_x, test_x = data_x[:train_size], data_x[train_size:train_size+test_size]
    train_y, test_y = data_y[:train_size], data_y[train_size:train_size+test_size]

    print("Train x shape: ",train_x.shape)
    print("Train y shape: ",train_y.shape)
    print("Test x shape: ", test_x.shape)
    print("Test y shape: ", test_y.shape)
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 25)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    """

    model = tf.keras.models.Sequential()
    keras.layers.Flatten(input_shape=(100, 25))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(250, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    #model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=6)
    test_results = model.evaluate(test_x, test_y)
    print("Test Accuracy: ", test_results[1]*100)



if __name__ == '__main__':
    main()