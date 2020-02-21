import pandas as pd

train = pd.read_csv('C:/Anaconda3/train.csv')
test = pd.read_csv('C:/Anaconda3/test.csv')

train.head(5)

train.shape
test.shape

train.info()

test.info()

train.isnull().sum()

test.isnull().sum()
