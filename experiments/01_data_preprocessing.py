import pandas as pd
import os 
import sys
# sys.path.append(os.path.join(os.getcwd(), 'tabular_reusable_assets'))
# sys.path
# os.listdir('tabular_reusable_assets/data')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#  to do Explore df function

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

# test_encoded[numerical_columns + categorical_columns]

# 
# !pip install optuna
# !pip install optuna-integration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import time

from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import optuna_integration

xy_train,  xy_val = train_test_split(train_encoded, test_size= 0.2, random_state=99, stratify=train_encoded[label])
features = categorical_columns + numerical_columns

X_train, y_train = xy_train[features], xy_train[label]
X_val, y_val = xy_val[features], xy_val[label]


X_train.to_csv('data/X_train.csv', index=False)
y_train.to_frame().to_csv('data/y_train.csv', index=False)
X_val.to_csv('data/X_val.csv', index=False)
y_val.to_frame().to_csv('data/y_val.csv', index=False)
test_encoded.to_csv('data/test_encoded.csv', index=False)


# X_test, y_test = test_encoded[features], test_encoded[label] # no labels for test

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
def score_model(model, score_df):
    y_score = model.predict_proba(score_df[features])
    y_true = score_df[label]
    return roc_auc_score(y_true, y_score[:,1])

score_model(model, xy_val)   


# Stage 1: Tune Tree Parameters with Optuna
## Naive Model
learning_rate = 0.3

metric = 'auc'
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': metric,
}

params = {
    'tree_method': 'approx',
    'learning_rate': learning_rate
}
params.update(base_params)
tic = time.time()

# help(XGBClassifier)
eval_set = [(X_train, y_train,), (X_val, y_val)]
model = XGBClassifier(**params,
                      n_estimators=500,
                    #   eval_set = eval_set,
                      early_stopping_rounds=50,
                      random_state=99)

model.fit(X_train, 
          y_train,
          eval_set = eval_set, 
        #   eval_names = ['train','valid'],
          verbose=False)

model.best_iteration
model.evals_result().keys()
model.evals_result()['validation_1']['auc']
print(f'{time.time() - tic:.1f} seconds')

# Set up objective for tuning
def objective(trial):
    params = {
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 250),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
        'learning_rate': learning_rate,
    }
    num_boost_round = 10000
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'validation_1-auc') 
    model = XGBClassifier(**params,
                        #   eval_names=['train','valid'],
                          n_estimators=num_boost_round,
                          early_stopping_rounds=50,
                          random_state=99)
    model = model.fit(X_train, y_train, eval_set = eval_set, callbacks=[pruning_callback])
    
    # model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round,
    #                   evals=[(dtrain, 'train'), (dvalid, 'valid')],
    #                   early_stopping_rounds=50,
    #                   verbose_eval=0,
    #                   callbacks=[pruning_callback])
    trial.set_user_attr('best_iteration', model.best_iteration)
    return model.best_score

# optuna study
study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=50)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
tic = time.time()
while time.time() - tic < 300:
    study.optimize(objective, n_trials=1)
    
print('Stage 1 ==============================')
print(f'best score = {study.best_trial.value}')
print('boosting params ---------------------------')
print(f'fixed learning rate: {learning_rate}')
print(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
print('best tree params --------------------------')
for k, v in study.best_trial.params.items():
    print(k, ':', v)
 

dtrain = xgb.DMatrix(data=xy_train[features], label=xy_train[label], 
                     enable_categorical=True)

dvalid = xgb.DMatrix(data=xy_val[features], label=xy_val[label], 
                     enable_categorical=True)

dtest = xgb.DMatrix(data=test_encoded[features], label=test_encoded[label], 
                    enable_categorical=True)

dtrainvalid = xgb.DMatrix(data=pd.concat([xy_train, xy_val])[features], 
                          label=pd.concat([xy_train, xy_val])[label], 
                          enable_categorical=True)

metric = 'rmse'
base_params = {
    'objective': 'reg:squarederror',
    'eval_metric': metric,
}

def score_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    y_true = dmat.get_label() 
    y_pred = model.predict(dmat) 
    return mean_squared_error(y_true, y_pred, squared=False)

# Stage 1: Tune Tree Parameters with Optuna
## Naive Model
learning_rate = 0.3

params = {
    'tree_method': 'approx',
    'learning_rate': learning_rate
}
params.update(base_params)
tic = time.time()
model = xgb.train(params=params, dtrain=dtrain,
                  evals=[(dtrain, 'train'), (dvalid, 'valid')],
                  num_boost_round=10000,
                  early_stopping_rounds=50,
                  verbose_eval=0)
print(f'{time.time() - tic:.1f} seconds')

# Set up objective for tuning
def objective(trial):
    params = {
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 250),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
        'learning_rate': learning_rate,
    }
    num_boost_round = 10000
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{metric}')
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      early_stopping_rounds=50,
                      verbose_eval=0,
                      callbacks=[pruning_callback])
    trial.set_user_attr('best_iteration', model.best_iteration)
    return model.best_score

# optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
tic = time.time()
while time.time() - tic < 300:
    study.optimize(objective, n_trials=1)
    
print('Stage 1 ==============================')
print(f'best score = {study.best_trial.value}')
print('boosting params ---------------------------')
print(f'fixed learning rate: {learning_rate}')
print(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
print('best tree params --------------------------')
for k, v in study.best_trial.params.items():
    print(k, ':', v)
    
    
# Stage 2: Intensify the Boosting Parameters
low_learning_rate = 0.01

params = {}
params.update(base_params)
params.update(study.best_trial.params)
params['learning_rate'] = low_learning_rate
model_stage2 = xgb.train(params=params, dtrain=dtrain, 
                         num_boost_round=10000,
                         evals=[(dtrain, 'train'), (dvalid, 'valid')],
                         early_stopping_rounds=50,
                         verbose_eval=0)

print('Stage 2 ==============================')
print(f'best score = {score_model(model_stage2, dvalid)}')
print('boosting params ---------------------------')
print(f'fixed learning rate: {params["learning_rate"]}')
print(f'best boosting round: {model_stage2.best_iteration}')


# Train and evaluate final model on both train and validation set
model_final = xgb.train(params=params, dtrain=dtrainvalid, 
                        num_boost_round=model_stage2.best_iteration,
                        verbose_eval=0)

print('Final Model ==========================')
print(f'test score = {score_model(model_final, dtest)}')
print('parameters ---------------------------')
for k, v in params.items():
    print(k, ':', v)
print(f'num_boost_round: {model_stage2.best_iteration}')


    
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import sklearn.metrics

n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state = 42)

    
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_folds, shuffle=True, random_state = 42)

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

# X_train = data[[col for col in data.columns if col in feature_columns]]
# y_train = data[label]

# X_train.loc[:, 'Age'] = X_train.loc[:,'Age'].apply(float) 


# model.fit(X_train, y_train)


# cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=kf.get_n_splits(X_train))


