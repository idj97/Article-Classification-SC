import re
import config_variables
import nltk
from os import walk
from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')
#lemmatizer = WordNetLemmatizer()

def process_file(input_path, output_path, stopwords):
    with open(input_path, 'r') as infile, open(output_path, 'w+') as outfile:
        output = ''
        for line in infile:
            words = line.split(' ')
            words = [process_word(word) for word in words if valid_word(word, stopwords)]
            if words: 
                line = ' '.join(words) + ' '
                output += line
        outfile.write(output)


#TODO: ubaci podrsku za lemmatizer za opciju USE_LEMMATIZER
def process_word(word):
    word = re.sub(r'[^a-zA-Z0-9]+', '', word)
    word = word.lower()
    return stem(word)


def valid_word(word, stopwords):
    word = process_word(word)
    return word not in stopwords and word != '' 


def main():
    stopwords_path = config_variables.STOPWORDS_PATH
    data_input_path = config_variables.ORIGINAL_DATA_PATH
    data_output_path = config_variables.PROCESSED_DATA_PATH

    print("Loading stopwords...")
    stopwords=[]
    with open(stopwords_path, 'r') as stopwords_file:
        for line in stopwords_file:
            line = line.replace('\n','')
            stopwords.append(line)
    
    print("Processing articles...")
    for (dirpath, _, filenames) in walk(data_input_path):
        for file in filenames:
            input_path = dirpath + '/' + file
            output_path = data_output_path + dirpath.split('/')[1] + '-' + file
            process_file(input_path, output_path, stopwords)
            print('Finished processing %s' % input_path)
    
    #process_file(data_input_path+'business/001.txt', data_output_path+'business-001.txt', stopwords)

if __name__ == '__main__':
    main()