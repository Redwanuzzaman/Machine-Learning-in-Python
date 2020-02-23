# delete unnecessary feature from dataset

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

train.head()
