import numpy as np 
from config_variables import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def extract_features_and_labels(data):
    features = data[:, :data.shape[1]-1]
    if BAG_OF_WORDS_TECHNIQUE != 'COUNT': features = features * 50
    labels = data[:, data.shape[1]-1:data.shape[1]].reshape(data.shape[0],)
    return features, labels

print("TECHNIQUE:", BAG_OF_WORDS_TECHNIQUE)

train = np.load(FEATURES_FILE_PATH + BAG_OF_WORDS_TECHNIQUE + '-features-train')
test = np.load(FEATURES_FILE_PATH + BAG_OF_WORDS_TECHNIQUE + '-features-test')
validation = np.load(FEATURES_FILE_PATH + BAG_OF_WORDS_TECHNIQUE + '-features-validation')

train_features, train_labels = extract_features_and_labels(train)
test_features, test_labels = extract_features_and_labels(test)
validation_features, validation_labels = extract_features_and_labels(validation)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(train_features, train_labels)
predicted_labels = classifier.predict(test_features)

print(confusion_matrix(test_labels, predicted_labels))
print(classification_report(test_labels, predicted_labels))