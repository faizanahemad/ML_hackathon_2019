import numpy as np
from collections import Counter
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm_plain

from keras import backend as K
import time
import numpy as np_utils

np.random.seed(2017)
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, DepthwiseConv2D, Conv2D, SeparableConv2D, \
    MaxPooling1D
from keras.layers import Input, concatenate
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l2
from keras_contrib.callbacks import CyclicLR
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from data_science_utils.vision.keras import *
from time import time
import pandas as pd
import numpy as np

import missingno as msno
import re
from joblib import Parallel, delayed
from data_science_utils import dataframe as df_utils
from data_science_utils import models as model_utils
from data_science_utils import plots as plot_utils
from data_science_utils.dataframe import column as column_utils
from data_science_utils import misc as misc
from data_science_utils import preprocessing as pp_utils
from data_science_utils import nlp as nlp_utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from data_science_utils.dataframe import get_specific_cols

import more_itertools
from more_itertools import flatten
import ast
from sklearn.preprocessing import LabelEncoder


def build_dict(data, vocab_size=100000, min_count=5):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    word_count = Counter()  # A dict storing the words that appear in the reviews along with how often they occur
    for sentence in tqdm(data):
        word_count.update(sentence)

    print("Total Words before Min frequency filtering", len(word_count))
    sorted_words = [word for word, freq in word_count.most_common() if freq >= min_count]
    print("Total Words after Min frequency filtering", len(sorted_words))
    word_dict = {}  # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]):  # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2  # 'infrequent' labels

    return word_dict


def get_text_le(colname, vocab_size=100000):
    le = {}
    INFREQ = 1
    NOWORD = 0
    UNKNOWN_TOKEN = '<unknown>'

    def le_train(df):
        le['wd'] = build_dict(df[colname].values, vocab_size=vocab_size)
        return le['wd']

    def word2label(word):
        word_dict = le['wd']
        if word in word_dict:
            return word_dict[word]
        else:
            return INFREQ

    def wordarray2labels(wordarray):
        return list(map(word2label, wordarray))

    def le_transform(df):
        word_list = [wordarray2labels(x) for x in tqdm(df[colname])]
        return word_list

    return le_train, le_transform, le

def get_le(colname="GL"):
    le = LabelEncoder()
    UNKNOWN_TOKEN = '<unknown>'

    def le_train(df):
        return le.fit(list(df[colname]) + [UNKNOWN_TOKEN])

    def le_transform(df):
        uniq_labels = set(le.classes_)
        return le.transform(df[colname].apply(lambda label: label if label in uniq_labels else UNKNOWN_TOKEN))

    return le_train, le_transform, le


def preprocess_for_word_cnn(df, text_column='text_raw', output_column="text", word_length_filter=2, jobs=20):
    """
    Preprocess and convert all text columns to one column named text
    """
    pp = lambda text: nlp_utils.combined_text_processing(text, word_length_filter=word_length_filter)
    text = Parallel(n_jobs=jobs, backend="loky")(delayed(pp)(x) for x in tqdm(df[text_column].fillna(' ').values))
    df[output_column] = text
    return df



def conv_layer(inputs, n_kernels=32, kernel_size=3, dropout=0, dilation_rate=1, padding='valid', strides=1):
    out = Conv1D(n_kernels,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding,
                 kernel_regularizer=l2(1e-4),
                 activation='relu',
                 dilation_rate=dilation_rate)(inputs)
    out = BatchNormalization()(out)
    if dropout > 1e-8:
        out = Dropout(dropout)(out)
    return out

def fc_layer(inputs, neurons=32, dropout=0, bn=True):
    out = Dense(neurons)(inputs)
    out = BatchNormalization()(out) if bn else out
    out = Activation('relu')(out)
    if dropout > 1e-8:
        out = Dropout(dropout)(out)
    return out


def transition_layer(inputs, n_kernels=32, dropout=0):
    out = conv_layer(inputs, n_kernels, kernel_size=1, dropout=dropout, padding='same')
    return out


def pre_dense_layer(inputs):
    out1 = GlobalAveragePooling1D()(inputs)
    out2 = GlobalMaxPooling1D()(inputs)
    out = concatenate([out1, out2])
    return out

def pad_text_sequences(sequences,maxlen,empty='',jobs=2):
    def pad(seq):
        ls = len(seq)
        len_empty = maxlen - ls
        if len_empty<=0:
            return seq[:maxlen] if ls>maxlen else seq
        return [empty]*len_empty + list(seq)
    sequences = Parallel(n_jobs=jobs, backend="loky")(delayed(pad)(x) for x in tqdm(sequences))
    return sequences

class PreTrainedEmbeddingsTransformer:
    def __init__(self, model="fasttext-wiki-news-subwords-300", size=300,
                 normalize_word_vectors=True):
        self.normalize_word_vectors = normalize_word_vectors
        self.model = model
        self.size = size
        self.token2vec_dict = {}

    def fit(self, X=None, y='ignored'):
        if type(self.model) == str:
            self.model = api.load(self.model)

    def partial_fit(self, X=None, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        uniq_tokens = set(more_itertools.flatten(X))
        uniq_tokens = uniq_tokens - self.token2vec_dict.keys()
        empty = np.full(self.size, 0)
        token2vec = {k: self.model.wv[k] if k in self.model.wv else empty for k in uniq_tokens}
        # token2vec = {k: np.nan_to_num(v / np.linalg.norm(v)) for k, v in token2vec.items()}
        token2vec = {k: np.nan_to_num(v) for k, v in token2vec.items()}
        self.token2vec_dict.update(token2vec)
        token2vec = self.token2vec_dict
        uniq_tokens = set(token2vec.keys())
        def tokens2vec(token_array):
            empty = np.full(self.size, 0)
            if len(token_array) == 0:
                return [empty]
            return [token2vec[token] if token in uniq_tokens else empty for token in token_array]

        # ft_vecs = list(map(tokens2vec, X))
        ft_vecs = [tokens2vec(t) for t in X]
        ft_vecs = np.array(ft_vecs)
        return ft_vecs

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X)

