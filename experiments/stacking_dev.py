from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
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

X = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
X_test = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)

fold_info = {'fold_num':[],
             'model_score':[],
             'trained_model':[]}

# X_train_val, y_train_val = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
# X_trainval = X_train_val.copy().reset_index(drop=True)
# y_trainval = y_train_val.copy().reset_index(drop=True)
# X_train = pd.concat([X_trainval, y_trainval], axis=1)

def create_stratified_kfolds(X, y, n_splits=5, shuffle=True, random_state=99):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    X = X.copy()
    X.loc[:, 'fold'] = -1
    for i, (train_idx, oof_idx) in tqdm(enumerate(kfold.split(X=X, y=y))):
        X.loc[oof_idx,'fold'] = i
    return X

X = create_stratified_kfolds(X=X, y=X[label], n_splits=5, shuffle=True, random_state=99)

def score_model(model, features, labels):
    if features is None or labels is None:
        raise ValueError('Need to specify `features` and `labels` arguments.')
    y_probs = model.predict_proba(features)
    y_true = labels
    score = roc_auc_score(y_true=y_true, y_score=y_probs[:,1])
    return score

fold_info = {'fold_num': [],
             'model_score': [],
             'trained_model': [], 
             'oof_preds': [],
             'X_test_preds': []}

fold_num = []
valid_scores = []
valid_predictions = []
test_predictions = []
model_from_fold_lst = [] 
X['xbg_oof_pred'] = -1
X['cust_id'] = np.arange(X.shape[0])
val_dict = {}
for fold in range(5):
    X_train_fold = X[X['fold']!=fold].reset_index(drop=True)
    X_val_fold = X[X['fold']==fold].reset_index(drop=True)
    X_train = X_train_fold[feature_columns]
    y_train = X_train_fold[label]

    # final_model_fold = clone(final_model)
    model_fold = XGBClassifier(random_state=fold)
    model_fold.fit(X_train, y_train, verbose=False)
    
    val_pred = model_fold.predict_proba(X_val_fold[feature_columns])[:,1]
    test_pred = model_fold.predict_proba(X_test[feature_columns])[:,1]
    valid_score = score_model(model_fold, X_val_fold[feature_columns], X_val_fold[label])    

    cust_ids = X_train_fold['cust_id'].values
    #update dict
    val_dict.update(dict(zip(cust_ids, val_pred)))
    X.loc[X['fold']==fold,'xbg_oof_pred'] = val_pred
    # X.loc[X[X['fold']==fold].index, 'oof_pred'] = val_pred
    fold_num.append(fold)
    valid_scores.append(valid_score)
    valid_predictions.append(val_pred)
    test_predictions.append(test_pred)
    model_from_fold_lst.append(model_fold)

# Averaged oof model score #
avg_score = np.mean(valid_scores) # 0.8607
avg_score

### Out of fold predictions ####
val_dict_df = pd.DataFrame.from_dict(val_dict, orient='index').reset_index()
valid_predictions[0].shape # fails because each fold is not necessarily the same
# Test pandas df from dictionary
some_dict = {}
a,b = ['a','b','c'], [1,2,3]
some_dict.update(dict(zip(a,b))) # inplace operation
df = pd.DataFrame.from_dict(some_dict, orient='index').reset_index()
df.columns = ['name', 'value']

### See score from one model fold ### 
test_score = score_model(model_fold, features=X_test[feature_columns], labels=X_test[label])
test_score # 0.892

#### Score from model trained on entire train dataset wihtout any folds ####
retrained_model = XGBClassifier()
retrained_model.fit(X[feature_columns],X[label])
retrained_score = score_model(retrained_model, features=X_test[feature_columns], labels=X_test[label])
retrained_score # 0.8981

# Model in fold test predictions
fold_test_predictions = np.column_stack([test_predictions])
assert fold_test_predictions.shape == (5, test_predictions[0].shape[0])
simple_blend_base1 = np.mean(fold_test_predictions, axis=0)
simple_blend_score = roc_auc_score(X_test[label], y_score=simple_blend_base1) # 0.914

## To run separate model blend scores
fold_num = []
valid_scores = []
valid_predictions = []
test_predictions = []
model_from_fold_lst = [] 
X['rf_oof_pred'] = -1
X['cust_id'] = np.arange(X.shape[0])
val_dict = {}
for fold in range(5):
    X_train_fold = X[X['fold']!=fold].reset_index(drop=True)
    X_val_fold = X[X['fold']==fold].reset_index(drop=True)
    X_train = X_train_fold[feature_columns]
    y_train = X_train_fold[label]

    # final_model_fold = clone(final_model)
    model_fold = RandomForestClassifier(random_state=fold)
    model_fold.fit(X_train, y_train)
    
    val_pred = model_fold.predict_proba(X_val_fold[feature_columns])[:,1]
    test_pred = model_fold.predict_proba(X_test[feature_columns])[:,1]
    valid_score = score_model(model_fold, X_val_fold[feature_columns], X_val_fold[label])    

    cust_ids = X_train_fold['cust_id'].values
    #update dict
    val_dict.update(dict(zip(cust_ids, val_pred)))
    X.loc[X['fold']==fold,'rf_oof_pred'] = val_pred
    # X.loc[X[X['fold']==fold].index, 'oof_pred'] = val_pred
    fold_num.append(fold)
    valid_scores.append(valid_score)
    valid_predictions.append(val_pred)
    test_predictions.append(test_pred)
    model_from_fold_lst.append(model_fold)

# Average score
rf_avg_score = np.mean(valid_scores)
rf_avg_score # 0.861

# Simple fold model score
sample_fold_score = score_model(model_fold, features=X_test[feature_columns], labels=X_test[label])
sample_fold_score # 0.898

# Retrained model score
retrained_rf_model = RandomForestClassifier()
retrained_rf_model.fit(X[feature_columns], X[label])
retrained_rf_score = score_model(retrained_rf_model, features=X_test[feature_columns], labels=X_test[label])
retrained_rf_score # 0.905

# Model in fold test scores
rf_fold_test_predictions = np.column_stack(test_predictions)
assert rf_fold_test_predictions.shape == (X_test.shape[0], 5)
rf_blend_test_predictions = rf_fold_test_predictions.mean(axis=1)
rf_blend_scores = roc_auc_score(y_true=X_test[label], y_score=rf_blend_test_predictions)
rf_blend_scores # 0.91324

# Blend of blend
final_blend = 0.5 * rf_blend_test_predictions  + 0.5 * simple_blend_base1
final_blend_score = roc_auc_score(y_true=X_test[label],y_score=final_blend)
final_blend_score # 0.9175, increased by few percentage points

# Meta Model
oof_model_pred_cols = ['xgb_oof_pred', 'rf_oof_pred']

# X = X.rename(columns={'xbg_oof_pred': 'xgb_oof_pred'})
meta_fold_models = []

## To run separate model blend scores
fold_num = []
valid_scores = []
valid_predictions = []
test_predictions = []
model_from_fold_lst = [] 
val_dict = {}

simple_blend_base1.shape
rf_blend_test_predictions.shape
meta_X_test = pd.DataFrame({'xgb_oof_pred': simple_blend_base1, 'rf_oof_pred': rf_blend_test_predictions })
X_test['xgb_oof_pred'] = simple_blend_base1
X_test['rf_oof_pred'] = rf_blend_test_predictions
meta_X_test = X_test
coefs= []
for fold in range(5):
    X_train_fold = X[X['fold']!=fold].reset_index(drop=True)
    X_val_fold = X[X['fold']==fold].reset_index(drop=True)
    X_train = X_train_fold[oof_model_pred_cols]
    y_train = X_train_fold[label]

    meta_fold_model = LogisticRegression(penalty='l2', C=1.5, random_state=fold, solver='saga')
    # meta_fold_model = Lasso(alpha=0.1) 
   
    meta_fold_model.fit(X_train, y_train)
    # valid_score = score_model(meta_fold_model, X_val_fold[oof_model_pred_cols], X_val_fold[label])
    test_pred = meta_fold_model.predict_proba(meta_X_test[oof_model_pred_cols])[:,1]
    valid_pred = meta_fold_model.predict_proba(X_val_fold[oof_model_pred_cols])[:,1]
    valid_score = roc_auc_score(X_val_fold[label], valid_pred)
    valid_scores.append(valid_score)
    test_predictions.append(test_pred)
    valid_predictions.append(valid_pred)

valid_predictions[0].shape

## Get the averaged stacked validation score
stacked_score = np.mean(valid_scores) # 0.8606 for logistic regression
stacked_score  # 0.864

meta_fold_model.coef_

roc_auc_score(X_test[label], np.array(test_predictions).mean(axis=0))

meta_predictions = np.column_stack(test_predictions)
meta_blend = meta_predictions.mean(axis=-1)
meta_score = roc_auc_score(X_test[label], meta_blend)
meta_score # 0.914

meta_score = roc_auc_score(X_test[label], meta_X_test.mean(axis=1))
meta_score

from functools import partial
import scipy

def neg_auc_score(coef, y_true, y1, y2):
    pred = coef[0] * y1 + coef[1] * y2
    return -roc_auc_score(y_true, y_score=pred)
 
initial_coef = [0.2, 0.3]
roc_auc_score(X_test[label], 0.47 * X_test['xgb_oof_pred'] + 0.52 * X_test['rf_oof_pred'])
# negated_auc_score  = neg_auc_score(coef, y_true=X[label], y1=X['xgb_oof_pred'], y2 = X['rf_oof_pred'])
partial_loss = partial(neg_auc_score, y_true=X[label],y1=X['xgb_oof_pred'], y2 = X['rf_oof_pred']) 
best_coef = scipy.optimize.minimize(partial_loss, initial_coef, method='nelder-mead')

best_coef

neg_auc_score(best_coef.x, X_test[label], X_test['xgb_oof_pred'], X_test['rf_oof_pred']) # 0.91699

# Averaged cofficients

import np


class OptimizedRounder_v2(object):
    """https://www.kaggle.com/code/naveenasaithambi/optimizedrounder-improved"""
    def __init__(self):
        self.coef_ = 0

    def get_loss(self, coef, X, y):
        pred = (X * coef).sum(axis=1)
        return -roc_auc_score(y_true=y, y_score=pred)
    
    def fit(self, X, y, random_init=True):
        loss_partial = partial(self.get_loss, X=X, y=y)
        if not random_init:
            initial_coef = [0.5, 0.5]
        else:
            initial_coef = scipy.special.softmax(np.random.uniform(low=0, high=1, size=X.shape[1]))
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        
    def predict(self, X):
        return (self.coefficients() * X).sum(axis=1)

    def coefficients(self):
        return self.coef_.x

opt = OptimizedRounder_v2()       
opt.fit(X=X[['xgb_oof_pred', 'rf_oof_pred']], y=X[label])      
opt.coefficients()
opt_x = opt.predict(X_test[['xgb_oof_pred', 'rf_oof_pred']])

roc_auc_score(y_true=X_test[label], y_score=opt_x) # 0.917523

X_test[['xgb_oof_pred', 'rf_oof_pred']] * opt.coef_

(X_test[['xgb_oof_pred', 'rf_oof_pred']] * [0.1, 0.5]).sum(axis=-1)
        
#### Experiment on averaged nelder-mead optimzed coefficients #####

meta_fold_models = []

## To run separate model blend scores
fold_num = []
valid_scores = []
valid_predictions = []
test_predictions = []
model_from_fold_lst = [] 
val_dict = {}

meta_X_test = pd.DataFrame({'xgb_oof_pred': simple_blend_base1, 'rf_oof_pred': rf_blend_test_predictions })
X_test['xgb_oof_pred'] = simple_blend_base1
X_test['rf_oof_pred'] = rf_blend_test_predictions
meta_X_test = X_test
coefs= []
for fold in range(5):
    X_train_fold = X[X['fold']!=fold].reset_index(drop=True)
    X_val_fold = X[X['fold']==fold].reset_index(drop=True)
    X_train = X_train_fold[oof_model_pred_cols]
    y_train = X_train_fold[label]

    meta_fold_model = LogisticRegression(penalty='l2', C=1.5, random_state=fold, solver='saga')
    # meta_fold_model = Lasso(alpha=0.1) 
   
    meta_fold_model.fit(X_train, y_train)
    
    opt = OptimizedRounder_v2()       
    opt.fit(X=X_train_fold[oof_model_pred_cols], y=X_train_fold[label], random_init=False) 
    coefs.append(opt.coefficients())
    # valid_score = score_model(meta_fold_model, X_val_fold[oof_model_pred_cols], X_val_fold[label])
    test_pred = meta_fold_model.predict_proba(meta_X_test[oof_model_pred_cols])[:,1]
    valid_pred = meta_fold_model.predict_proba(X_val_fold[oof_model_pred_cols])[:,1]
    valid_score = roc_auc_score(X_val_fold[label], valid_pred)
    valid_scores.append(valid_score)
    test_predictions.append(test_pred)
    valid_predictions.append(valid_pred)

# Really marginally better, and it is random
averaged_coefs = np.stack(coefs).mean(axis=0)
a_coef_pred = (averaged_coefs * X[['rf_oof_pred','xgb_oof_pred']]).mean(axis=1)
roc_auc_score(y_true=X[label], y_score=a_coef_pred)  # 0.86278338, 0.8626 without random init

a_coef_pred = ([0.5,0.5] * X[['rf_oof_pred','xgb_oof_pred']]).mean(axis=1)
roc_auc_score(y_true=X[label], y_score=a_coef_pred)  # 0.86273

averaged_coefs = np.stack(coefs).mean(axis=0)
a_coef_pred = (averaged_coefs * X_test[['rf_oof_pred','xgb_oof_pred']]).mean(axis=1)
roc_auc_score(y_true=X_test[label], y_score=a_coef_pred)  #0.91739







    
        
    # def _kappa_loss(self, coef, X, y):
    #     preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
    #     return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    # def fit(self, X, y):
    #     loss_partial = partial(self._kappa_loss, X = X, y = y)
    #     initial_coef = [0.5, 1.5, 2.5, 3.5]
    #     self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    # def predict(self, X, coef):
    #     preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
    #     return preds
    
    # def coefficients(self):
    #     return self.coef_['x']
    
    
# # Test blended model
# class StackingAveragedModels:
#     def __init__(self, base_models, meta_model, n_folds=5, add_original_features=False):
#         self.base_models = base_models
#         self.meta_model = meta_model
#         self.n_folds = n_folds
#         self.add_original_features = add_original_features

#     def get_oof_num(self, X, y):
#         kfold = KFold(n_splits = self.n_folds, shuffle=True, random_state=156)
#         all_data = pd.concat([X, y], axis=1)
#         all_data.loc[:, 'oof_num'] = 0
#         for idx, (train_idx, oof_idx) in enumerate(kfold.split(X,y)):
#             all_data.loc[oof_idx, 'oof_num'] = idx
#         return all_data
    
#     def fit(self, X, y):
#         self.fitted_base_models = [list() for _ in range(len(self.base_models))]
#         out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
#         out_of_fold_predictions = pd.DataFrame(out_of_fold_predictions, columns=[f'base_model_{idx+1}_oof_pred' for idx in range(len(self.base_models))])
#         oof_metadata = self.get_oof_num(X, y)
#         for oof_idx in range(self.n_folds):
#             oof_X_train = X[all_data['oof_num'] != oof_idx]
#             oof_y_train = y[all_data['oof_num'] != oof_idx]

#             oof_X_test = X[all_data['oof_num'] == oof_idx]
#             for model_idx, base_model in enumerate(self.base_models):
#                 fitted_base_model = base_model.fit(oof_X_train, oof_y_train)
#                 self.fitted_base_models[model_idx].append(fitted_base_model)
#                 oof_predictions = fitted_base_model.predict_proba(oof_X_test)[:,1]
#                 out_of_fold_predictions.iloc[oof_X_test.index, model_idx] = oof_predictions
#         if self.add_original_features:
#             out_of_fold_predictions = pd.concat([X, out_of_fold_predictions], axis=1)
#         self.fitted_meta_model = self.meta_model.fit(out_of_fold_predictions, y)
#         return self
    
#     def predict_proba(self, X):
#         meta_features = np.column_stack(
#             [np.column_stack([model.predict_proba(X)[:,1] for model in within_fold_models]).mean(axis=1)
#              for within_fold_models in self.fitted_base_models])
        
#         meta_features = pd.DataFrame(meta_features, columns=[f'base_model_{idx+1}_oof_pred' for idx in range(len(self.base_models))])
#         if self.add_original_features:
#             meta_features = pd.concat([X, meta_features], axis=1)
#         self.meta_features = meta_features
#         predictions = self.fitted_meta_model.predict_proba(meta_features)[:,1]
#         return predictions
    
    
# model1 = 
# base_models = [model, model, model]
# meta_model = metamodel = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# sam = StackingAveragedModels(base_models, meta_model, add_original_features=True)
# sam_f = sam.fit(X_train, y_train)
# pred = sam.predict_proba(X_train)



# import numpy as np
# import pandas as pd
# import time


# categorical_columns = ['Embarked', 
#                       'Parch',
#                       'SibSp',
#                       'Sex',
#                       'Pclass',
#                       'Ticket',
#                       'Cabin']

# numerical_columns = ['Age', 'Fare']
# feature_columns = categorical_columns + numerical_columns
# identifier = 'PassengerId'
# label = 'Survived'

# X_train=pd.read_csv('data/X_train.csv',)
# y_train=pd.read_csv('data/y_train.csv',)
# X_val=pd.read_csv('data/X_val.csv',)
# y_val = pd.read_csv('data/y_val.csv',)
# test_encoded = pd.read_csv('data/test_encoded.csv',)

# X_train.groupby(['Parch']).groups.values()