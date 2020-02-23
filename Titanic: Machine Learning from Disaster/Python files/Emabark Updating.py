#filling up most of the places with Southampton symboling "S"
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
