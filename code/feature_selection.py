import numpy as np
import math
from config_variables import *
from os import listdir
from random import shuffle
from sklearn.feature_selection import SelectKBest, chi2
from collections import OrderedDict

input_path = PROCESSED_DATA_PATH
files = listdir(input_path)


print("Shuffling data...")
#shuffle(files)


print("Splitting data...")
total_size = len(files)
train_bound = int(TRAIN_RATIO*total_size)
test_bound = train_bound + int(TEST_RATIO*total_size)
train_files = files[:train_bound]
test_files = files[train_bound:test_bound]
validation_files = files[test_bound:]


print("Creating global vocubalary...")
vocabulary = OrderedDict()
vocabulary['articles'] = OrderedDict()
vocabulary['words'] = OrderedDict()
for file_name in train_files:
    with open(input_path + '/' + file_name) as infile:
        words = infile.read().split(" ")
        vocabulary['articles'][file_name] = OrderedDict()
        vocabulary['articles'][file_name]['words'] = words
        vocabulary['articles'][file_name]['label'] = LABELS[file_name.split("-")[0]] 
        for word in words:
            if word not in vocabulary['words']: vocabulary['words'][word] = 1
            else:                           vocabulary['words'][word] += 1


print("Creating feature vectors...")
articles = vocabulary['articles'].keys()
words = vocabulary['words'].keys()

data = []
labels = []
for article in articles:
    vector = [0]*len(vocabulary['words'])
    for i, word in enumerate(words):
        if BAG_OF_WORDS_TECHNIQUE == "COUNT":
            vector[i] = vocabulary['articles'][article]['words'].count(word)
        elif BAG_OF_WORDS_TECHNIQUE == "FREQUENCY":
            count = vocabulary['articles'][article]['words'].count(word)
            article_size = len(vocabulary['articles'][article]['words'])
            vector[i] = count / article_size
        elif BAG_OF_WORDS_TECHNIQUE == "TL_IDF":
            count = vocabulary['articles'][article]['words'].count(word)
            article_size = len(vocabulary['articles'][article]['words'])
            tf = count / article_size
            N = len(vocabulary['articles'].keys())
            inverse_count = vocabulary['words'][word]
            idf = math.log(N/inverse_count, 10)
            tfidf = tf * idf 
            if tfidf > 0:
                vector[i] = tfidf
            else:
                vector[i] = 0
    labels.append(vocabulary['articles'][article]['label'])                
    data.append(np.array(vector))
data = np.array(data)
labels = np.array(labels)


print("Selecting %d best features..." % NUMBER_OF_FEATURES)
test = SelectKBest(score_func=chi2, k=NUMBER_OF_FEATURES)
fit = test.fit(data, labels)
features = fit.transform(data)


print("Initial data shape:", data.shape)
print("Lables shape:", labels.shape)
print("Data shape after selection:", features.shape)


print("Saving features names...")
with open(SELECTED_FEATURES_NAMES_PATH + BAG_OF_WORDS_TECHNIQUE + '-names', 'w') as outfile:
    for i, key in enumerate(vocabulary['words'].keys()):
        if fit.get_support()[i]:
            outfile.write(key + '\n')

print("Saving features...")
with open(FEATURES_FILE_PATH + BAG_OF_WORDS_TECHNIQUE + '-features', 'wb+') as outfile:
    np.save(outfile, features)