ORIGINAL_DATA_PATH = 'original-data/'
PROCESSED_DATA_PATH = 'processed-data/cleaned-articles/'
STOPWORDS_PATH = 'processed-data/stop-words'
SELECTED_FEATURES_NAMES_PATH = 'processed-data/feature-names/'
FEATURES_FILE_PATH = 'processed-data/features/'

USE_LEMMATIZER = False

TRAIN_RATIO = 0.05
TEST_RATIO = 0.3
VALIDATION_RATIO = 0.1

LABELS = {
    "business":       0,
    "entertainment" : 1,
    "politics":       2,
    "sport":          3,
    "tech":           4
}

NUMBER_OF_FEATURES = 5
BAG_OF_WORDS_TECHNIQUE = 'TL_IDF'
