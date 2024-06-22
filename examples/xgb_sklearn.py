# https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/#stage-2-intensify-the-boosting-parameters
# !pip install optuna
# !pip install optuna-integration
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import numpy as np
import time


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

X_train=pd.read_csv('data/X_train.csv',)
y_train=pd.read_csv('data/y_train.csv',)
X_val=pd.read_csv('data/X_val.csv',)
y_val = pd.read_csv('data/y_val.csv',)
test_encoded = pd.read_csv('data/test_encoded.csv',)


# # Naive Model
# Default configs
metric = 'auc'
objective = 'binary:logistic'
learning_rate = 0.3
num_boost_rounds = 10000
early_stopping_rounds = 50

base_params = {'eval_metric': metric,
               'objective': objective}
tree_params = {'tree_method': 'approx',
               'learning_rate': learning_rate}
tree_params.update(base_params)

boosting_params = {'n_estimators': num_boost_rounds,
                   'early_stopping_rounds': early_stopping_rounds}

def score_model(model, features, labels):
    if features is None or labels is None:
        raise ValueError('Need to specify `features` and `labels` arguments.')
    y_probs = model.predict_proba(features)
    y_true = labels
    score = roc_auc_score(y_true=y_true, y_score=y_probs[:,1])
    return score

model = XGBClassifier()
model.fit(X_train, y_train)

naive_model_score = score_model(model, features=X_val[feature_columns], labels=y_val)
naive_model_score # 0.89

base_model = XGBClassifier(**tree_params)
base_model.fit(X_train, y_train)
base_model_score = score_model(base_model, features=X_val[feature_columns], labels=y_val)
base_model_score, naive_model_score

model2 = XGBClassifier(**tree_params, 
                       n_estimators=num_boost_rounds)
model2.fit(X_train,
           y_train, 
           eval_set=[(X_train, y_train), (X_val, y_val)],
           early_stopping_rounds=early_stopping_rounds,
           verbose=False)
score_model(model2, features=X_val[feature_columns], labels=y_val)   


#################################################
#### STAGE 1 Get the best tree parameters #######
#################################################
def objective(trial):
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    tree_params = {
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 250),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 50, log=True),
        'learning_rate': learning_rate,
    }
    num_boost_rounds = 10000 # high value because of early stopping
    early_stopping_rounds = 50 # possible to increase this if learning rate is small, to increase chance of breaking local min.
    tree_params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, observation_key=f'validation_1-{metric}')

    model = XGBClassifier(**tree_params, 
                          n_estimators=num_boost_rounds)

    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=False, 
              callbacks=[pruning_callback],
              early_stopping_rounds=early_stopping_rounds,)
    
    trial.set_user_attr('best_iteration', model.best_iteration) # best number of iteration
    return model.best_score

# optuna study
study  = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)
   
print('Stage 1 ==============================')
print(f'best score = {study.best_trial.value}') # 0.93
print('boosting params ---------------------------')
print(f'fixed learning rate: {learning_rate}')
print(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
print('best tree params --------------------------')
for k, v in study.best_trial.params.items():
    print(k, ':', v)
 
 # Optuna tune with time based as opposed to n_trials
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
tic = time.time()
while time.time() - tic < 300:
    study.optimize(objective, n_trials=1)

######################################################
#######  ## Continue from previous study. ############
######################################################
# https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
# 1. Save study
# study_name = "example-study"  # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)
# study = optuna.create_study(study_name=study_name, storage=storage_name)

# # 2. If using sampler, save the sampler
# import pickle
# # Save the sampler with pickle to be loaded later.
# with open("sampler.pkl", "wb") as fout:
#     pickle.dump(study.sampler, fout)
# restored_sampler = pickle.load(open("sampler.pkl", "rb"))

# study = optuna.create_study(
#     study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
# )
# study.optimize(objective, n_trials=3)
# # 3. if without sampler, just load_if_exist
# study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
# study.optimize(objective, n_trials=1)
######################################################
further_tuning_best_score = study.best_trial.value  # 0.94

stage_1_best_params = study.best_trial.params
stage_1_best_params.update(base_params)
stage_1_best_params.update({'learning_rate': learning_rate})
stage_1_model = XGBClassifier(**stage_1_best_params,
                              n_estimators = num_boost_rounds,)
stage_1_model.fit(X_train, 
                  y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=False)
                            
stage_1_model_score = score_model(stage_1_model, X_val[feature_columns], y_val)
assert stage_1_model_score == further_tuning_best_score

#################################################
#### STAGE 2 Get the best num boost round #######
#################################################
lower_learning_rate = 0.01
stage_2_best_params = study.best_trial.params
stage_2_best_params.update(base_params)
stage_2_best_params.update({'learning_rate': lower_learning_rate})

stage_2_model = XGBClassifier(**stage_2_best_params,
                              n_estimators= num_boost_rounds)
stage_2_model.fit(X_train, 
                  y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  early_stopping_rounds=5000,
                  verbose=False)

stage_2_best_n_estimators = stage_2_model.best_iteration
stage_2_best_n_estimators


stage_2_model_score = score_model(stage_2_model, X_val[feature_columns], y_val) 
## Stage 1 model score is actually higher than stage 2 model score here
stage_1_model_score, stage_2_model_score
                         
#########################
##### Final model #######
#########################

X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

final_model_parameters = study.best_trial.params
final_model_parameters.update(base_params)
final_model_parameters.update({'learning_rate': lower_learning_rate})

final_model_best_round = stage_2_best_n_estimators
final_model = XGBClassifier(**final_model_parameters,
                            n_estimators = final_model_best_round)

############################################################################
###### K FOLD cross validation to have the proxy for test performance ######
############################################################################
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

fold_info = {'fold_num':[],
             'model_score':[],
             'trained_model':[]}

for i, (train_idx, oof_idx) in tqdm(enumerate(kfold.split(X_trainval, y_trainval))):
    final_model_fold = clone(final_model)
    
    X_train_fold = X_trainval.loc[train_idx, :]
    y_train_fold = y_trainval.loc[train_idx, :]
    X_oof = X_trainval.loc[oof_idx,:]
    y_oof = y_trainval.loc[oof_idx,:]
    
    final_model_fold.fit(X_train_fold, y_train_fold)
    final_model_fold_score = score_model(final_model_fold, X_oof[feature_columns], y_oof)
    fold_info['fold_num'].append(i)
    fold_info['model_score'].append(final_model_fold_score)
    fold_info['trained_model'].append(final_model_fold)    

cross_val_model_score = np.mean(fold_info['model_score'])
cross_val_model_score, stage_1_model_score # (0.8705717591456933, 0.9293807641633729)
# It is posible that we overfitted to the validation dataset used for hyper-parameter tuning
# Can include averaged kfold validation inside the optimization opjective in Optuna

####################################
##### TRAIN ON ENTIRE DATA SET #####
####################################
final_model = XGBClassifier(**final_model_parameters,
                            n_estimators = final_model_best_round)

final_model.fit(X_trainval, y_trainval)

# Final score
assert round(naive_model_score,3) == 0.898
assert round(base_model_score,3) == 0.901
assert round(stage_1_model_score, 3) == 0.944
assert round(stage_2_model_score, 3) == 0.930