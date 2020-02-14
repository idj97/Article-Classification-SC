import re
from os import walk
from stemming.porter2 import stem


def process_file(input_path, output_path, stopwords):
    with open(input_path, 'r') as infile, open(output_path, 'w+') as outfile:
        for line in infile:
            line = re.sub(r'[^a-zA-Z0-9 \n]+', '', line).lower()
            words = line.split(' ')
            words = [process_word(word) for word in words if valid_word(word, stopwords)]
            if words: 
                line = ' '.join(words) + ' '
                outfile.write(line)


def process_word(word):
    word = re.sub(r'[^a-zA-Z0-9]+', '', word)
    word = word.lower()
    return stem(word)


def valid_word(word, stopwords):
    return word not in stopwords and re.sub(r'[^a-zA-Z0-9]+', '', word) != '' 


def main():
    stopwords=[]
    with open('stop-words', 'r') as stopwords_file:
        for line in stopwords_file:
            line = line.replace('\n','')
            stopwords.append(line)

    data_input_path = 'original-data/'
    data_output_path = 'processed-data/'
    for (dirpath, _, filenames) in walk(data_input_path):
        for file in filenames:
            input_path = dirpath + '/' + file
            output_path = data_output_path + dirpath.split('/')[1] + '-' + file
            print('Finished processing %s' % input_path)
            process_file(input_path, output_path, stopwords)


if __name__ == '__main__':
    main()