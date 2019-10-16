import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences= [
        'I love my cats.',
        'I love my dogs.',
        'People say that my dog is the best.']

test_sentence = [
        'My dog is one of the best',
        'my dog can be called my best friend.'
]

#defining tokenizer oov is out of vocab
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")

#fitting the train data on tokenizer
tokenizer.fit_on_texts(sentences)

# getting the word index for the the tokenizer
word_index = tokenizer.word_index

#converting the train set to sequences using tokenizer
sequences = tokenizer.texts_to_sequences(sentences)

#padding the sequence to make the vectors of same size. we can use maxlen property to specify the maximum length
pad_sequences = pad_sequences(sequences, padding='post')

#converting the test set to sequences using tokenizer
test_sequences = tokenizer.texts_to_sequences(test_sentence)


print("outputting worrd indexes:")
print(word_index)
print("\nprinting the sequences for train data:")
print(sequences)
print("\nprinting the padded sequences for train data:")
print(pad_sequences)
print("\nprinting the test sequence:")
print(test_sequences)