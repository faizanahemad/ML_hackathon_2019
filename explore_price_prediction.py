import pandas as pd
import numpy as np

df_train = pd.read_csv("price_prediction/training.csv")

df_test = pd.read_csv("price_prediction/public_test_features.csv")

print(df_train.shape)
print(df_test.shape)

df_train.head()