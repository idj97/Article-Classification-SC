ORIGINAL_DATA_PATH = 'original-data/'
PROCESSED_DATA_PATH = 'processed-data/cleaned-articles/'
STOPWORDS_PATH = 'processed-data/stop-words'
SELECTED_FEATURES_NAMES_PATH = 'processed-data/feature-names/'
FEATURES_FILE_PATH = 'processed-data/features/'

USE_LEMMATIZER = True

TRAIN_RATIO = 0.6
TEST_RATIO = 0.4
VALIDATION_RATIO = 0

LABELS = {
    "business":       0,
    "entertainment" : 1,
    "politics":       2,
    "sport":          3,
    "tech":           4
}

INVERTED_LABELS = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}

NUMBER_OF_FEATURES = 1500
BAG_OF_WORDS_TECHNIQUE = 'COUNT'
#BAG_OF_WORDS_TECHNIQUE = 'HASHING'
#BAG_OF_WORDS_TECHNIQUE = 'TF_IDF'

#SELECTOR = 'chi2'
SELECTOR = 'other'

CLASSIFIER_TYPE = 'random_forest'
SVC_KERNEL = 'linear'
KNN_N_NEIGHBOURS = 1
RANDOM_FOREST_MAX_DEPTH = 5
RANDOM_FOREST_RANDOM_STATE = 0