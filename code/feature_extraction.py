import numpy as np
import math
from config_variables import *
from os import listdir
from random import shuffle
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def split_files():
    print("Splitting data...")
    files = listdir(PROCESSED_DATA_PATH)
    shuffle(files)
    total_size = len(files)
    train_bound = int(TRAIN_RATIO*total_size)
    test_bound = train_bound + int(TEST_RATIO*total_size)
    train_files = files[:train_bound]
    test_files = files[train_bound:test_bound]
    validation_files = files[test_bound:]
    return train_files, test_files, validation_files


def get_features_and_labels(files):
    data = []
    labels = []
    for f in files:
        with open(PROCESSED_DATA_PATH + f, 'r') as document:
            text = document.read()
            data.append(text)
            labels.append(LABELS[f.split('-')[0]])
    labels = np.array(labels)
    return data, labels


def get_vectorizer():
    if BAG_OF_WORDS_TECHNIQUE == 'COUNT':
        return CountVectorizer()
    elif BAG_OF_WORDS_TECHNIQUE == 'TF_IDF':
        return TfidfVectorizer()
    else:
        return HashingVectorizer(n_features=20000)


def get_selector():
    if SELECTOR == 'chi2':
        return SelectKBest(score_func=chi2, k=NUMBER_OF_FEATURES)
    else:
        return SelectKBest(score_func=f_classif, k=NUMBER_OF_FEATURES)


def transform(fit, data, labels):
    data = fit.transform(data)
    print(labels.shape)
    labels = labels.reshape(len(labels),1)
    return np.concatenate((data, labels), axis=1)


def save(data, name):
    with open(FEATURES_FILE_PATH + BAG_OF_WORDS_TECHNIQUE + '-features-' + name, 'wb') as outfile:
        np.save(outfile, data)


def main():
    # PODELI FAJLOVE PREMA RAZMERI
    train_files, test_files, validation_files = split_files()

    # ZA SVAKI SKUP DOBAVI PODATKE I OZNAKE
    train_data, train_labels = get_features_and_labels(train_files)
    test_data, test_labels = get_features_and_labels(test_files)
    validation_data, validation_labels = get_features_and_labels(validation_files)

    # NAPRAVI VECTORIZER I VOKABULAR PREMA TRAIN PODACIMA
    vectorizer = get_vectorizer()
    vectorizer.fit(train_data)

    # TRANFORMISI PODATKE PREMA VOKABULARU
    train_data = np.array(vectorizer.transform(train_data).toarray())
    test_data = np.array(vectorizer.transform(test_data).toarray())
    validation_data = np.array(vectorizer.transform(validation_data).toarray())

    # NAPRAVI SELEKTOR I IZABERI FEATURE PREMA TRAIN PODACIMA
    selector = get_selector()   
    fit = selector.fit(train_data, train_labels)
    
    # TRANSFORMISI PODATKE I SPREMI IH ZA SNIMANJE
    train = transform(fit, train_data, train_labels)
    test = transform(fit, test_data, test_labels)
    validation = transform(fit, validation_data, validation_labels)

    # SNIMANJE
    save(train, 'train')
    save(test, 'test')
    save(validation, 'validation')

if __name__ == '__main__':
    main()