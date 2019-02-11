#Read in dataset, extract emojis, create new dataset of emojis only, and annotate sentiments 
import os 

#open dataset containing words and emojis 
raw_file = open(r'C:\Users\katie\Documents\4thYEARENGINEERING\MLSAProject\2019\EmotionDetection\100EmoticonDataset.csv',  encoding='utf-8')
print('opened file')
#opep blank text file for saving emojis
clean_file = open('EmojisAnnotated.txt','w', encoding = 'utf-8')

#define utf encoded emoticons
neutral = u'\U0001f610'         #annotation = 0
sad = u'\U0001f641'         #annotation = 4
angry = u'\U0001F620'        #annotation = 5
happy = u'\U0001f642'          #annotation = 1
love = u'\U0001f60d'        #annotation = 2
surprised = u'\U0001f62e'       #annotation = 3

#extract and write all emoticons to blank file
store_emoji = ' '
sentiment = 7
for line in raw_file:
    tokens = line.split()
    for word in tokens: 
        if word == neutral:
            store_emoji = word
            sentiment = int(0)
        elif word == sad:
            store_emoji = word
            sentiment = int(4)
        elif word == angry:
            store_emoji = word
            sentiment = int(5)
        elif word == happy:
            store_emoji = word
            sentiment = int(1)
        elif word == love:
            store_emoji = word
            sentiment = int(2)
        elif word == surprised:
            store_emoji = word
            sentiment = int(3)

    #create and save new file with ONLY emoticons and annotations 
    clean_file.write(store_emoji)
    clean_file.write(" ")
    clean_file.write('%d' % sentiment)
    clean_file.write('\n')
clean_file.close()








         

