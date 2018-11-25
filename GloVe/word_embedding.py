import numpy as np
class WordEmbedding:

    def __init__(self, fpath, num_dimensions):
        self.fpath = fpath
        self.num_dimensions = num_dimensions
        self.embedding = self.load_glove_model()

    def get_num_dimensions(self):
        return self.num_dimensions

    def load_glove_model(self):
        print('Loading GloVe model consisting of {this.num_dimensions} dimensions...')
        file = open(self.fpath)
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

    def get_word_embedding(self):
        return self.embedding
