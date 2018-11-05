import numpy as np

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
    fname = '/Users/issa/Downloads/glove.twitter.27B/glove.twitter.27B.50d.txt'
    mapping = loadGlove(fname)
    print(mapping['hi'])
    print(mapping['hello'])


if __name__ == '__main__':
    main()
