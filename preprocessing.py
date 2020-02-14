import re
from os import walk
from stemming.porter2 import stem

def process_file(input_path, output_path, stopwords):
    with open(input_path, 'r') as infile, open(output_path, 'w+') as outfile:
        for line in infile:
            line = line.replace('\n', '').lower()
            line = re.sub(r'[^a-zA-Z0-9 ]+', '', line)
            words = line.split(' ')
            words = [stem(word) for word in words if word not in stopwords]
            line = ' '.join(words)
            outfile.write(line)

def main():
    stopwords=[]
    with open('stop-words', 'r') as stopwords_file:
        for line in stopwords_file:
            line = line.replace('\n','')
            stopwords.append(line)

    data_input_path = 'original-data/'
    data_output_path = 'processed-data/'
    for (dirpath, dirname, filenames) in walk(data_input_path):
        for file in filenames:
            input_path = dirpath + '/' + file
            output_path = data_output_path + dirpath.split('/')[1] + '-' + file
            print('Finished processing %s' % input_path)
            process_file(input_path, output_path, stopwords)

if __name__ == '__main__':
    main()