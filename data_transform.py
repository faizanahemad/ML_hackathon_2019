import pandas as pd
import numpy as np
from lib_research import *


maxlen = 100
df = pd.read_csv("price_prediction/india-asins.csv",sep='\t',header=None,names=["asin","price","gl","text_raw"])
df = preprocess_for_word_cnn(df,jobs=4)
df.drop('text_raw', axis=1, inplace=True)
df['text'] = list(pad_text_sequences(df['text'].values, maxlen=maxlen,jobs=4))
df.to_csv("price_prediction/india-asins-processed.csv",index=False)



df_sample = df.sample(20*1000)
df_sample.to_csv("price_prediction/india-asins-train.csv",index=False)
df_sample = df.sample(5*1000)
df_sample.to_csv("price_prediction/india-asins-test.csv",index=False)


