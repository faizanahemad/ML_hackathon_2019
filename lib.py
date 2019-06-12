import numpy as np
from collections import Counter
from tqdm import tqdm_notebook as tqdm

from keras import backend as K
import time
import numpy as np_utils
np.random.seed(2017)
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, DepthwiseConv2D, Conv2D, SeparableConv2D, MaxPooling1D
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

def build_dict(data, vocab_size = 50000,min_count=5):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    word_count = Counter() # A dict storing the words that appear in the reviews along with how often they occur
    for sentence in tqdm(data):
        word_count.update(sentence)
    
    print("Total Words before Min frequency filtering",len(word_count))
    sorted_words = [word for word,freq in word_count.most_common() if freq>=min_count]
    print("Total Words after Min frequency filtering",len(sorted_words))
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict


def get_text_le(colname):
    le = {}
    INFREQ = 1
    NOWORD = 0
    UNKNOWN_TOKEN = '<unknown>'
    def le_train(df):
        le['wd'] = build_dict(df[colname].values)
        return le['wd']
    
    def word2label(word):
        word_dict = le['wd']
        if word in word_dict:
            return word_dict[word]
        else:
            return INFREQ
    def wordarray2labels(wordarray):
        return list(map(word2label,wordarray))
    def le_transform(df):
        word_list = [wordarray2labels(x) for x in tqdm(df[colname])]
        return word_list
        
    return le_train,le_transform, le


def build_char_dict(data, vocab_size = 128,min_count=100):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    word_count = Counter() # A dict storing the words that appear in the reviews along with how often they occur
    for sentence in tqdm(data):
        word_count.update(list(sentence))
    
    print("Total Words before Min frequency filtering",len(word_count))
    sorted_chars = [word for word,freq in word_count.most_common() if freq>=min_count]
    print("Total Words after Min frequency filtering",len(sorted_chars))
    char_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_chars[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        char_dict[word] = idx + 2                              # 'infrequent' labels
        
    return char_dict


def get_char_le(colname):
    le = {}
    INFREQ = 1
    NOWORD = 0
    def le_train(df):
        le['wd'] = build_char_dict(df[colname].values)
        return le['wd']
    
    def char2label(char):
        char_dict = le['wd']
        if char in char_dict:
            return char_dict[char]
        else:
            return INFREQ
    def sentence2labels(wordarray):
        return list(map(char2label,wordarray))
    def le_transform(df):
        char_list = [sentence2labels(x) for x in tqdm(df[colname])]
        return char_list
        
    return le_train,le_transform, le

def get_le(colname="GL"):
    le = LabelEncoder()
    UNKNOWN_TOKEN = '<unknown>'
    def le_train(df):
        return le.fit(list(df[colname])+[UNKNOWN_TOKEN])
    def le_transform(df):
        uniq_labels = set(le.classes_)
        return le.transform(df[colname].apply(lambda label:label if label in uniq_labels else UNKNOWN_TOKEN))
    return le_train,le_transform, le

preprocess_string = lambda x:re.sub('[^ a-zA-Z0-9%@_]',' ',nlp_utils.clean_text(x)) if x is not None and type(x)==str else x



def preprocess_for_word_cnn(df,text_columns=['TITLE', 'BULLET_POINTS', 'GL'], output_column="text",jobs=20):
    """
    Preprocess and convert all text columns to one column named text
    """
    pp = lambda text: nlp_utils.combined_text_processing(preprocess_string(text))
    df[output_column] = df[text_columns[0]].fillna(' ')
    for col in text_columns[1:]:
        df[output_column] = df[output_column] + df[col].fillna(' ')
    
    text = Parallel(n_jobs=jobs, backend="loky")(delayed(pp)(x) for x in tqdm(df[output_column].values))
    df[output_column] = text
    return df

def preprocess_for_char_cnn(df,text_columns=['TITLE', 'BULLET_POINTS', 'GL'], output_column="char",jobs=20):
    """
    Preprocess and convert all text columns to one column named text
    """
    df[output_column] = df[text_columns[0]].fillna(' ')
    for col in text_columns[1:]:
        df[output_column] = df[output_column] + df[col].fillna(' ')
    
    return df

preprocess_v2 = lambda x:re.sub('[^ a-zA-Z0-9%@_()\[\]]',' ',nlp_utils.clean_text(x)) if x is not None and type(x)==str else x
def preprocess_for_fasttext_cmd(df,text_columns=['TITLE', 'BULLET_POINTS', 'GL'], output_column="char",jobs=32):
    """
    Preprocess and convert all text columns to one column named text
    """
    df[output_column] = df[text_columns[0]].fillna(' ')
    for col in text_columns[1:]:
        df[output_column] = df[output_column] + df[col].fillna(' ')
    
    text = Parallel(n_jobs=jobs, backend="loky")(delayed(preprocess_v2)(x) for x in tqdm(df[output_column].values))
    df[output_column] = text
    return df


def conv_layer(inputs, n_kernels=32, kernel_size=3, dropout=0.1,dilation_rate=1, padding='valid'):
    out = Conv1D(n_kernels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                kernel_regularizer=l2(1e-4),
                dilation_rate=dilation_rate)(inputs)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(dropout)(out)
    return out

def transition_layer(inputs, n_kernels=32,dropout=0):
    out = conv_layer(inputs, n_kernels, kernel_size=1,dropout=dropout, padding='same')
    return out

def pre_dense_layer(inputs):
    out1 = GlobalAveragePooling1D()(inputs)
    out2 = GlobalMaxPooling1D()(inputs)
    out = concatenate([out1,out2])
    return out

def grouped_layer(inputs, group_configs, out_channels):
    groups = []
    for group_config in group_configs:
        out1 = inputs
    for layer_config in group_config:
        out1 = conv_layer(out1, **layer_config)
    groups.append(out1)
    y = concatenate(groups)
    y = transition_layer(y, out_channels)
    return y


class PreTrainedEmbeddingsTransformer:
    def __init__(self,model="fasttext-wiki-news-subwords-300",size=300,
                 normalize_word_vectors=True):
        self.normalize_word_vectors = normalize_word_vectors
        self.model = model
        self.size = size
        
    def fit(self, X=None, y='ignored'):
        if type(self.model) == str:
            self.model = api.load(self.model) 

    def partial_fit(self, X=None, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        print("Fasttext Transforms start at: %s" % (str(pd.datetime.now())))
        uniq_tokens = set(more_itertools.flatten(X))
        print("Number of Unique Test Tokens for Fasttext transform %s"%len(uniq_tokens))
        empty = np.full(self.size, 0)
        token2vec = {k: self.model.wv[k] if k in self.model.wv else empty for k in uniq_tokens}
        token2vec = {k: np.nan_to_num(v / np.linalg.norm(v)) for k, v in token2vec.items()}


        def tokens2vec(token_array):
            empty = np.full(self.size, 0)
            if len(token_array) == 0:
                return empty
            return [token2vec[token] if token in uniq_tokens else empty for token in token_array]

        # ft_vecs = list(map(tokens2vec, X))
        ft_vecs = [tokens2vec(t) for t in tqdm(X)]
        ft_vecs = np.array(ft_vecs)
        print("Fasttext Transforms done at: %s" % (str(pd.datetime.now())))
        return ft_vecs

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X)

    