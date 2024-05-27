# https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/

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

n_valid = 12000
n_test = 12000

sorted_df = df.sort_values(by='saledate')
train_df = sorted_df[:-(n_valid + n_test)] 
valid_df = sorted_df[-(n_valid + n_test):-n_test] 
test_df = sorted_df[-n_test:]

dtrain = xgb.DMatrix(data=train_df[features], label=train_df[target], 
                     enable_categorical=True)
dvalid = xgb.DMatrix(data=valid_df[features], label=valid_df[target], 
                     enable_categorical=True)
dtest = xgb.DMatrix(data=test_df[features], label=test_df[target], 
                    enable_categorical=True)
dtrainvalid = xgb.DMatrix(data=pd.concat([train_df, valid_df])[features], 
                          label=pd.concat([train_df, valid_df])[target], 
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