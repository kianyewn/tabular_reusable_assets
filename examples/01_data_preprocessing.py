import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

categorical_columns = ['Embarked', 
                      'Parch',
                      'SibSp',
                      'Sex',
                      'Pclass',
                      'Ticket',
                      'Cabin']

numerical_columns = ['Age', 'Fare']
feature_columns = categorical_columns + numerical_columns
identifier = 'PassengerId'
label = 'Survived'

# missing columns
def process_missing_cols(data, missing_cols):
    for col in missing_cols:
        if col in numerical_columns:
            data.loc[:, col] = data[col].fillna('9999990')
        if col in categorical_columns:
            data.loc[:, col] = data[col].fillna('NA')
    return data
def get_missing_cols(data):
    null_df = data.isnull().sum()
    null_df_ms = null_df[null_df!=0]
    missing_cols = null_df_ms.index.tolist()
    return missing_cols

train_missing_cols = get_missing_cols(train)
train = process_missing_cols(train, missing_cols=train_missing_cols)
test_missing_cols = get_missing_cols(test)
test = process_missing_cols(test, missing_cols=test_missing_cols)

def clean_invalid_dtype(data, invalid_dtype_cols):
    for col in invalid_dtype_cols:
        if col in categorical_columns:
            data[col] = data[col].apply(str)
            
        if col in numerical_columns:
            data[col] = data[col].apply(float)
    return data

invalid_dtype_cols = ['SibSp', 'Parch', 'Age', 'Fare']       
train = clean_invalid_dtype(train, invalid_dtype_cols=invalid_dtype_cols)
test = clean_invalid_dtype(test, invalid_dtype_cols=invalid_dtype_cols)

from sklearn.preprocessing import LabelEncoder

def train_categorical_encoder(data):
    encoder_objects = {}
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(data[col])
        data[f'{col}'] = le.transform(data[col])
        encoder_objects[col] = le

    return data, encoder_objects

train_encoded, train_encoder_object = train_categorical_encoder(train)

def test_categorical_encoder(test, train_encoder_objects):
    for col in categorical_columns:
        le = train_encoder_objects[col]
        le.fit(test[col])
        test[f'{col}'] = le.transform(test[col])

    return test
    
test_encoded = test_categorical_encoder(test, 
                                        train_encoder_objects=train_encoder_object)

from sklearn.model_selection import train_test_split

xy_train,  xy_val = train_test_split(train_encoded, test_size= 0.2, random_state=99, stratify=train_encoded[label])
features = categorical_columns + numerical_columns

X_train, y_train = xy_train[features], xy_train[label]
X_val, y_val = xy_val[features], xy_val[label]

X_train.to_csv('data/X_train.csv', index=False)
y_train.to_frame().to_csv('data/y_train.csv', index=False)
X_val.to_csv('data/X_val.csv', index=False)
y_val.to_frame().to_csv('data/y_val.csv', index=False)
test_encoded.to_csv('data/test_encoded.csv', index=False)