from tensorflow import keras
import numpy as np
import pandas as pd
from word_embedding import WordEmbedding
class CsvSequence(keras.utils.Sequence):

    def __init__(self, fpath, embedding, batch_size):
        self.fpath = fpath
        self.batch_size = batch_size
        self.embedding = embedding

    def __len__(self):
        return int(np.ceil(sum(1 for row in open(self.fpath, encoding='latin1')) / float(self.batch_size)))

    def __getitem__(self, idx):
        df_values = pd.read_csv(self.fpath, skiprows=(idx*self.batch_size), nrows=(idx+1)*self.batch_size).get_values()
        batch_x = [str(val[5]) for val in df_values]
        batch_y = [int(val[0]) for val in df_values]
        #batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        #return np.zeros(25)
        return transform_data(batch_x, batch_y)

        #return np.array([resize(imread(file_name), (200, 200)) for file_name in batch_x]), np.array(batch_y)

    def transform_data(self, x_data, y_data):
        train_x = [[embedding.get(word) if word in embedding else np.zeros(25) for word in doc.lower().split()] for doc in docs]
        train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=50, padding='pre', truncating='pre', dtype='float')
        train_y = np.asarray([1 if x==4 else 0 for x in sentiments], dtype=np.int32)
        print("train_x length: ",len(train_x), ", train_y length: ",len(train_y))
        return train_x, train_y

"""
def main():
    f_glove = '/Users/issa/Downloads/glove.twitter.27B/glove.twitter.27B.25d.txt'
    f_data = '/Users/issa/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.unsorted.csv'
    n = 25
    e = WordEmbedding(f_glove, n)
    seq = CsvSequence(f_data, e, 100)
    print(seq.__len__())

if __name__ == '__main__':
    main()
"""
