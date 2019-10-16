import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('stopwords')
nltk.download('punkt')

tokenized_words = ['i', 'am', 'going', 'to', 'go', 'to', 'the', 'store', 'and', 'park']

stopwords = stopwords.words('english')

sentences = []
labels = []
print("reading the data and pre processing it................")
with open("bbc-text.csv", 'r') as csvfile:
    i=0
    data = csv.reader(csvfile)
    for row in data:
        if (i!=0):
            tokens = word_tokenize(row[1])
            filtered_sentence = [w for w in tokens if not w in stopwords]
            str1 = ""

            # traverse in the string
            for ele in filtered_sentence:
                str1 += ele +" "
            #print(filtered_sentence)
            sentences.append(str1)
            labels.append(row[0])
        i=1

print(len(sentences))
print(sentences[0])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index =tokenizer.word_index
print(len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

print(padded[0])
print(padded.shape)

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(labels)
label_sequence = tokenizer2.texts_to_sequences(labels)
label_word_index = tokenizer2.word_index

print(label_sequence)
print(label_word_index)