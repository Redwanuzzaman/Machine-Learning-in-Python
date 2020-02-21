# We need to import the "pandas" package to read csv files and much more functionalities
import pandas as pd

train = pd.read_csv('C:/Anaconda3/train.csv')
test = pd.read_csv('C:/Anaconda3/test.csv')

# Informations of train DF row 1-5
# train.head(5)

# Row-Columns of train and test Data Frame
train.shape
test.shape

# Details of train and test table
train.info()
test.info()

# To know how many fields are empty in train and test DF
train.isnull().sum()
test.isnull().sum()

# matplotlib.pyplot লাইব্রেরি দেয় আমাদের ম্যাটল্যাবের মতো চমৎকার প্লটিং ফ্রেমওয়ার্ক। 
# ছবিগুলো জুপিটার নোটবুকে একসাথে দেখানোর জন্য inline মোড নিয়ে আসা হয়েছে। 
# seaborn হচ্ছে পাইথনের matplotlib ভিত্তিক স্ট্যাটিসটিকাল গ্রাফিক্যাল লাইব্রেরি।
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

# making a function to display the survival and dead ratio for all variables 
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))

# Checking bar charts of different variables
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart("Parch")

# making another function which compares survival and dead ration with total passengers
def bar_chart(feature):
    total = train[train['Survived'] >= 0][feature].value_counts()
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([total, survived, dead])
    df.index = ['Total', 'Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))

bar_chart('Pclass')
bar_chart('Parch')
