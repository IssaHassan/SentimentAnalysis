#Open emoji dataset, one-hot encode, and feed through basic NN
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#random seed for reproducibility
seed = 7
np.random.seed(seed)

# 3 layer neural network
def nn():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=6, activation='relu'))
	model.add(Dense(7, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#load file
df = pandas.read_csv(r'C:\Users\katie\Documents\4thYEARENGINEERING\MLSAProject\2019\EmotionDetection\EmojisAnnotated.txt', sep = ' ', error_bad_lines=False, header=None)
print('opened file with pandas')
# print(df.head())

# separate input and target columns, print to verify
dataset = df.values
Emoji = dataset[:,0:1].astype(str)
# print(Emoji)
Sentiment = dataset[:,1:2]
# print(Sentiment)

#encode sentiments
Sentiment_array = np.ravel(Sentiment)
encoder = LabelEncoder()
encoder.fit(Sentiment_array)
encoded_sentiment = encoder.transform(Sentiment_array)
# convert integers to a one hot encoding
one_hot_sentiment = np_utils.to_categorical(encoded_sentiment)
print(one_hot_sentiment)

#encode emojis 
encoded_emoji = pandas.get_dummies(df.iloc[:,0:1])
print(encoded_emoji)

# build classifier 
estimator = KerasClassifier(build_fn=nn, epochs=100, batch_size=5, verbose=0)

# k-fold cross-validation 
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# evaluate 
results = cross_val_score(estimator, encoded_emoji, one_hot_sentiment, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))










