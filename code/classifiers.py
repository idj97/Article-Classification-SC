import numpy as np 
from config_variables import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier


# def create_nn_model():
#     model = Sequential()
#     model.add(Dense(8, input_dim=NUMBER_OF_FEATURES, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


def create_classifier(type):
    if type == 'svc':
        return SVC(kernel=SVC_KERNEL)
    elif type == 'knn':
        return KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBOURS)
    elif type == 'random_forest':
        return RandomForestClassifier(max_depth=RANDOM_FOREST_MAX_DEPTH, random_state=RANDOM_FOREST_MAX_DEPTH)
    # elif type == 'neural_network':
    #     return KerasClassifier(build_fn = create_classifier, epochs=200, batch_size=5, verbose=0) 


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

classifier = create_classifier(CLASSIFIER_TYPE)
classifier.fit(train_features, train_labels)
accuracy = classifier.score(test_features, test_labels)
print(accuracy)

predicted_labels = classifier.predict(test_features)
print(confusion_matrix(test_labels, predicted_labels))
print(classification_report(test_labels, predicted_labels))