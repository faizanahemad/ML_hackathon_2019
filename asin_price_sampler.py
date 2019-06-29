import pandas as pd
import numpy as np

df = pd.read_csv("price_prediction/india-asins.csv",sep='\t',header=None,names=["asin","price","gl","text_raw"])

df_sample = df.sample(20*1000)
df_sample.to_csv("price_prediction/india-asins-train.csv",index=False)

df_sample = df.sample(5*1000)
df_sample.to_csv("price_prediction/india-asins-test.csv",index=False)

df.to_csv("price_prediction/india-asins.csv",index=False)
