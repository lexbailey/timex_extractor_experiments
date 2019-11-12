#!/usr/bin/env python
import keras
from keras import backend as K
import numpy as np
import json

main_data = 'glove.6B.100d.txt'
#main_data = 'glove.840B.300d.txt_2'
small_data = 'test_dataset.100d.txt'
false_data = 'false_embedding.9d.txt'

def load_embeddings(data_file):
    embeddings_index = {}
    word_index = {}
    f = open(data_file)
    dim = int(data_file.split('.')[-2][0:-1])
    for i, line in enumerate(f):
        values = line.split()
        parts = len(values)
        word = ' '.join(values[0:parts-dim])
        coefs = np.asarray(values[parts-dim:], dtype='float32')
        embeddings_index[word] = coefs
        word_index[word] = i
    f.close()
    print('Loaded %s word vectors with dimension %d.' % (len(embeddings_index), dim))
    #word_index = {
    #    word: index+1 for index, word in enumerate(embeddings_index)
    #}
    open('word_index.json', 'w+').write(json.dumps(word_index))
    return embeddings_index, word_index, dim

def embedding_layer(window_size, data_file):
    embeddings_index, word_index, embedding_dim = load_embeddings(data_file)
    embedding_matrix = np.zeros((len(embeddings_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return keras.layers.Embedding(len(embeddings_index) + 1,
                     embedding_dim,
                     weights=[embedding_matrix],
                     input_length=window_size,
                     trainable=False), word_index

def pretrained_embedding_layer(window_size):
    return embedding_layer(window_size, main_data)

def encode_word(word, word_index):
    if word in word_index:
        return word_index[word]
    if word.lower() in word_index:
        return word_index[word.lower()]
    return 0

if __name__ == '__main__':
    inp = keras.layers.Input((5,))
    embedding, word_index = embedding_layer(5, small_data)
    embedding = embedding(inp)
    m = keras.models.Model(inputs=[inp], outputs=[embedding])
    words = []
    text = []
    n = 0
    for word, i in word_index.items():
        n+=1
        if n > 5:
            break
        words.append(i)
        text.append(word)
    print(words)
    print(text)
    print(m.predict([[words]]))
    embeddings_index, word_index, embedding_dim = load_embeddings(small_data)
    print("actual values...")
    for t in text:
        print(embeddings_index[t])
