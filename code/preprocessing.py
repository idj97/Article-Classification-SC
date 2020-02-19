import re
import nltk
from config_variables import *
from os import walk
from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()


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
    if USE_LEMMATIZER and word != '':
        tag = nltk.pos_tag([word])
        tag = tag[0][1]
        tag = get_wordnet_pos(tag)
        if tag == '':
            return word
        return lemmatizer.lemmatize(word, tag)
    else:
        return stem(word)


def valid_word(word, stopwords):
    word = process_word(word)
    return word not in stopwords and word != '' 


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def main():
    print("Loading stopwords...")
    stopwords=[]
    with open(STOPWORDS_PATH, 'r') as stopwords_file:
        for line in stopwords_file:
            line = line.replace('\n','')
            stopwords.append(line)
    
    print("Processing articles...")
    for (dirpath, _, filenames) in walk(ORIGINAL_DATA_PATH):
        for file in filenames:
            input_path = dirpath + '/' + file
            output_path = PROCESSED_DATA_PATH + dirpath.split('/')[1] + '-' + file
            process_file(input_path, output_path, stopwords)
            print('Finished processing %s' % input_path)
    
    #process_file(data_input_path+'business/001.txt', data_output_path+'business-001.txt', stopwords)

if __name__ == '__main__':
    main()