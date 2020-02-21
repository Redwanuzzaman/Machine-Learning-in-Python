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

bar_chart('Sex')

bar_chart('Pclass')

bar_chart('SibSp')

bar_chart("Parch")

# Another function for total passenger with survival records

def bar_chart(feature):
    total = train[train['Survived'] >= 0][feature].value_counts()
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([total, survived, dead])
    df.index = ['Total', 'Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))

bar_chart('Sex')

bar_chart('Pclass')

bar_chart('SibSp')

bar_chart("Parch")
