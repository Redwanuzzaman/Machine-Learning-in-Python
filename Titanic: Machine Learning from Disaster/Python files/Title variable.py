# Making a Title variable for both datasets
train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    
train['Title'].value_counts()    
test['Title'].value_counts()
