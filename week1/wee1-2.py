import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)


print("hello world")
sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

print("sentences are: \n")
print(len(sentences))
print("\n labels are: \n")
print(len(labels))
print("\n urls are:\n")
print(len(urls))


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
#print(sequences)

padded = pad_sequences(sequences, padding='post')
print(sentences[0])
print(padded[0])
print(padded.shape)