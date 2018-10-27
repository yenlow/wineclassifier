import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import utils
import imp
imp.reload(utils)

import tensorflow as tf
from tensorflow import keras
import keras_metrics
layers = keras.layers
print("TensorFlow version:", tf.__version__)

#Parameters
vocab_size = 10000
embedding_dim = 64
epochs = 20
batch_size = 128


#Read in data (clean)
from data import data


#Split data test set
X_train, X_test, Y_train, Y_test = train_test_split(data[['description','variety','price','country']],
                                                    data.accept, test_size=0.1)
X_train.dtypes

###  Features
# Wide feature 1: one-hot vector of variety categories
variety_train, variety_test, variety_classes = utils.onehot_traintest(X_train.variety, X_test.variety, sparse=False)

# Wide feature 2: one-hot vector of country categories
country_train, country_test, country_classes = utils.onehot_traintest(X_train.country, X_test.country, sparse=False)
# keras' to_categorical requires int input; use sklearn utility to convert label to num first
# encoder = LabelEncoder()
# int_train = encoder.fit_transform(X_train.country)
# int_test  = encoder.transform(X_test.country)
# num_classes = len(encoder.classes_)
# country_train = keras.utils.to_categorical(int_train, num_classes)
# country_test  = keras.utils.to_categorical(int_test, num_classes)

# Wide feature 3: price
#X_train.price, X_test.price

# Wide feature 4: sparse BOW
# Tokenize text description into BOW
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(X_train.description) # only fit on train
bow_train = tokenize.texts_to_matrix(X_train.description)
bow_test = tokenize.texts_to_matrix(X_test.description)
bow_train.shape

# Deep feature 1: word embeddings
train_embed = tokenize.texts_to_sequences(X_train.description)
test_embed  = tokenize.texts_to_sequences(X_test.description)
max_seq_length = max([len(i) for i in train_embed])
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed,
                                                         maxlen=max_seq_length, padding="post")
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed,
                                                        maxlen=max_seq_length, padding="post")

### Models
# Specify wide model
variety_inputs = layers.Input(shape=(variety_classes,))
country_inputs = layers.Input(shape=(country_classes,))
price_inputs = layers.Input(shape=(1,), dtype='float32')
bow_inputs = layers.Input(shape=(vocab_size,))
inputs = [variety_inputs, country_inputs, price_inputs, bow_inputs]
#inputs = [variety_inputs, bow_inputs]
merged_layer = layers.concatenate(inputs)
merged_layer = layers.Dense(256, activation='relu', kernel_initializer=keras.initializers.glorot_normal(seed=1))(merged_layer)
predictions = layers.Dense(1, activation='sigmoid')(merged_layer)
wide_model = keras.Model(inputs, outputs=predictions)

wide_model.compile(loss='binary_crossentropy', optimizer='adam',
                   metrics=[keras_metrics.precision(), keras_metrics.recall()])
wide_model.summary()

# Specify deep model
deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, embedding_dim, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1, activation='sigmoid')(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)

deep_model.compile(loss='binary_crossentropy', optimizer='adam',
                   metrics=[keras_metrics.precision(), keras_metrics.recall()])
deep_model.summary()


# Specify wide and deep joint models
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1, activation='sigmoid')(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)

combined_model.compile(loss='binary_crossentropy', optimizer='adam',
                       metrics=[keras_metrics.precision(), keras_metrics.recall()])
combined_model.summary()


# Fit model
combined_model.fit([variety_train, country_train, X_train.price, bow_train] + [train_embed],
                   Y_train, epochs=epochs, batch_size=batch_size)

# Evaluate model
# 0.9131966113958655, 0.8428056304408202 (variety + BOW),
# 0.5584932894755105, 0.06715930690218691 (variety + country)
# 0, 0 (variety + price)
# 0.9075242707753298, 0.874058631539188 (variety + price + BOW)
# 0.9032618561977273, 0.8766712077635164 (variety + country + price + BOW)
combined_model.evaluate([variety_test, country_test, X_test.price, bow_test] + [test_embed],
                        Y_test, batch_size=batch_size)

#Top predictions
predictions = pd.DataFrame(mod_all.predict([variety_test, country_test, X_test.price, bow_test] + [test_embed]),
                           columns=['pred'])
predictions.index = Y_test.index
predictions['points'] = data.points[Y_test.index]
predictions.plot.scatter('points','pred')
plt.show()

predictions.pred.describe()
best =predictions.sort_values('pred', axis=0, ascending=False, inplace=False)[:20]
worst = predictions.sort_values('pred', axis=0, ascending=True, inplace=False)[:20]


X_test.loc[best.index.values]
X_test.loc[worst.index.values]

#save model
mod_variety_bow = combined_model
mod_all = combined_model

mod_all.save('mod_all.h5')
