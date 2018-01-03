""" Word-based ConvNet toxic comment classifier.

For Kaggle competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Heavily inspired by/cribbed from the Keras
imdb_cnn and pretrained_embeddings examples.
"""
from __future__ import print_function
import codecs
from datetime import datetime
from time import time
import re
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import *
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Merge
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import *
# import ipdb


# DEFINE SOME USEFUL STUFF
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 20000
BATCH_SIZE = 64
FILTERS = 150
KERNEL_SIZES = (2, 3, 4, 5)
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
P_DROPOUT = 0.5
EPOCHS = 10
STOPWORDS = frozenset(open("stopwords.txt").read().rstrip().split("\n"))
PUNCT_SPLITTER = re.compile(ur"[^\w'-]", re.UNICODE)


# START TRACKING TIMING
tick = time()

# LOAD/PREPROCESS DATA
print("LOADING DATA...")
train_df = pd.read_csv('train.csv')
texts = train_df[u'comment_text'].values
# remove most punctuation and stopwords
texts = np.array([" ".join(filter(lambda x: x not in STOPWORDS, PUNCT_SPLITTER.sub(" ", t.lower()).split())) for t in texts])
labels = train_df[[u'toxic', u'severe_toxic', u'obscene', u'threat', u'insult', u'identity_hate']].values

print('##### %d seconds elapsed #####' % (time() - tick,))

print('Tokenizing')
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %d unique tokens' % (len(word_index),))
print('##### %d seconds elapsed #####' % (time() - tick,))

print('Padding, encoding, train/dev split')
data = pad_sequences(sequences,
                     maxlen=MAX_SEQUENCE_LENGTH,
                     padding='pre',
                     truncating='pre')
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print("Shuffle & create train/val/test split")
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-2 * nb_validation_samples]
y_train = labels[:-2 * nb_validation_samples]
x_val = data[-2 * nb_validation_samples:-nb_validation_samples]
y_val = labels[-2 * nb_validation_samples:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
print('Shape of training set:', x_train.shape)
print('Shape of validation set:', x_val.shape)
print('Shape of test set:', x_test.shape)
print('##### %d seconds elapsed #####' % (time() - tick,))

# PREPARE THE EMBEDDINGS INDEX AND MATRIX
print('Preparing embedding index')
embeddings_index = {}
for line in open("GloVe/glove.6B.%dd.txt" % EMBEDDING_DIM):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

print('Preparing embedding matrix')
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word, None)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print('##### %d seconds elapsed #####' % (time() - tick,))


print('Build multi-kernel-width word model')
# Functional API
x = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# traininable char embeddings
emb = Embedding(len(word_index) + 1,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=True)(x)
conv_maxpool_layers = []
for kw in KERNEL_SIZES:
    conv_maxpool_layers.append(GlobalMaxPooling1D()(Conv1D(FILTERS, kw, padding='valid',
                               activation='relu', strides=1)(emb)))
merged = concatenate(conv_maxpool_layers, axis=-1)
hidden = Dense(HIDDEN_DIMS, activation='relu')(merged)
dropout = Dropout(P_DROPOUT)(hidden)
out = Dense(6, activation='sigmoid')(dropout)

# # Sequential API
# submodels = []
# for kw in KERNEL_SIZES:
#     submodel = Sequential()
#     # fixed (pre-trained) word embeddings
#     submodel.add(Embedding(len(word_index) + 1,
#                            EMBEDDING_DIM,
#                            weights=[embedding_matrix],
#                            input_length=MAX_SEQUENCE_LENGTH,
#                            trainable=True))
#     submodel.add(Conv1D(FILTERS,
#                         kw,
#                         padding='valid',
#                         activation='relu',
#                         strides=1))
#     submodel.add(GlobalMaxPooling1D())
#     submodels.append(submodel)
# word_model = Sequential()
# word_model.add(Merge(submodels, mode="concat"))
# word_model.add(Dense(HIDDEN_DIMS))
# word_model.add(Dropout(P_DROPOUT))
# word_model.add(Activation('relu'))
# word_model.add(Dense(6))
# word_model.add(Activation('sigmoid'))

word_model = Model(inputs=x, outputs=out)
print('Compiling model')
word_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
print(word_model.summary())
print('##### %d seconds elapsed #####' % (time() - tick,))


print('Training word model model with KERNEL_SIZES=%s' % str(KERNEL_SIZES))
callbacks = [EarlyStopping(patience=2, min_delta=0.001, verbose=1)]
# replicate inputs across all "filter" channels (for train and val sets)
hist = word_model.fit(x_train, y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(x_val, y_val),
                      callbacks=callbacks)
print('##### %d seconds elapsed #####' % (time() - tick,))

# print('History')
# print(json.dumps(hist.history, indent=2))
# now = datetime.now().strftime("%Y%d_%H%M")
# word_model.save('big_model_%s.dat' % (now,))
# print('##### %d seconds elapsed #####' % (time() - tick,))


def columnwise_log_loss(y_true, y_pred, eps=1e-15):
    """ Return columnwise log-losses on N>1 multilabel predictions.

        y_true: (n_samples, n_classes) array of multilabel ground truths
        y_pred: (n_samples, n_classes) array of multilabel class probabilities
    """
    y_hat = np.clip(y_pred, eps, 1 - eps)
    log_losses = ((y_true * (-np.log(y_hat))) + ((1 - y_true) * (1 - np.log(y_hat))))
    col_log_losses = log_losses.sum(axis=0)
    return col_log_losses/log_losses.shape[0]


def calc_loss(y_true, y_pred):
    return np.mean([log_loss(y_true[:, i], y_pred[:, i])
                    for i in range(y_true.shape[1])])


print('\n##### ConvNet eval (train) #####\n')
yhat_train = np.round(word_model.predict(x_train)).astype('int32')
# print(classification_report(y_train, yhat_train, digits=3,
#                             target_names=["toxic", "severe", "obscene", "threat", "insult", "hate"]))
print("Mean columnwise log-loss: %0.3f" % np.mean(columnwise_log_loss(y_train, yhat_train)))
print("Other mean log-loss: %0.3f" % calc_loss(y_train, yhat_train))

print('\n##### ConvNet eval (val) #####\n')
yhat_val = np.round(word_model.predict(x_val)).astype('int32')
# print(classification_report(y_val, yhat_val, digits=3,
#                             target_names=["toxic", "severe", "obscene", "threat", "insult", "hate"]))
print("Mean columnwise log-loss: %0.3f" % np.mean(columnwise_log_loss(y_val, yhat_val)))
print("Other mean log-loss: %0.3f" % calc_loss(y_val, yhat_val))

print('\n##### ConvNet eval (test) #####\n')
yhat_test = np.round(word_model.predict(x_test)).astype('int32')
# print(classification_report(y_test, yhat_test, digits=3,
#                             target_names=["toxic", "severe", "obscene", "threat", "insult", "hate"]))
print("Mean columnwise log-loss: %0.3f" % np.mean(columnwise_log_loss(y_test, yhat_test)))
print("Other mean log-loss: %0.3f" % calc_loss(y_test, yhat_test))


print("\nPredicting on test data...")
df = pd.read_csv("test.csv").fillna('')
test_ids = df['id'].values
test_texts = df['comment_text'].str.lower().values
cleaned_test_texts = [PUNCT_SPLITTER.sub(" ", t).split() for t in test_texts]
cleaned_test_texts = [" ".join(filter(lambda x: x not in STOPWORDS, t)) for t in cleaned_test_texts]
# cleaned_test_texts = np.array(cleaned_test_texts)

assert len(test_ids) == len(cleaned_test_texts)
print("%d test texts" % len(cleaned_test_texts))

test_sequences = tokenizer.texts_to_sequences(cleaned_test_texts)
test_data = pad_sequences(test_sequences,
                          maxlen=MAX_SEQUENCE_LENGTH,
                          padding='pre',
                          truncating='pre')
preds = word_model.predict(test_data)
predictions = zip(test_ids, preds)
now_str = datetime.now().strftime("%Y%m%d_%H%M")
f_name = "test_submission.word_cnn.%s.csv" % now_str
print("Writing submission file: %s" % f_name)
with codecs.open(f_name, "w", "utf8") as f_out:
    f_out.write(",".join(['id'] + list(train_df.columns[2:].str.lower().values)))
    for ident, pred in predictions:
        f_out.write(",".join([str(ident)] + map(str, list(pred))) + "\n")

print('##### %d seconds elapsed TOTAL #####\n' % (time() - tick,))
