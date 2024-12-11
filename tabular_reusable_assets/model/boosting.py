from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import scipy.stats as stats
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from xgboost import XGBClassifier, plot_importance, XGBRegressor



#### to put inside models folder
def baseline_xgb(X_train, y_train, X_test, y_test, cv_n_iter=10, sample_weight=None):

    xgb_param_fixed = {
        # use all possible cores for training
        'n_jobs': -1,
        # set number of estimator to a large number
        # and the learning rate to be a small number,
        # we'll let early stopping decide when to stop
        'n_estimators': 300,
        'random_state':2020,
    }

    xgb_hyperparam_options = {
        'max_delta_step': stats.randint(low = 0, high = 5), # setting it to a positive value, might help when class is extremely imbalanced, as it makes the update more conservative
        'max_depth': stats.randint(low = 3, high = 15),
        'colsample_bytree': stats.uniform(loc = 0.7, scale = 0.3),
        'subsample': stats.uniform(loc = 0.7, scale = 0.3),
        'alpha': stats.uniform(loc = 0, scale = 10),
        'lambda': stats.uniform(loc = 0, scale = 10),
        'scale_pos_weight': [(y_train.value_counts()[0] / y_train.value_counts()[1]), 1.0],
        'learning_rate': [0.1, 0.01],
    }

    xgb_fit_params = {
        'eval_metric': ['logloss','auc'],
        'eval_set': [(X_train, y_train), (X_test, y_test)],
        'early_stopping_rounds': 10,
        'verbose': False
    }

    xgb_base = XGBClassifier(**xgb_param_fixed)

    xgb_estimator  = RandomizedSearchCV(
        estimator = xgb_base,
        param_distributions = xgb_hyperparam_options,
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=2020),
        n_iter = cv_n_iter , #n_iter,
        verbose = 1,
        random_state = 2020).fit(X_train, y_train, **xgb_fit_params, sample_weight=sample_weight)

    print('Best score obtained: {0}'.format(xgb_estimator.best_score_))
    print('Best Parameters:')
    for param, value in xgb_estimator.best_params_.items():
        print('\t{}: {}'.format(param, value))
    return xgb_estimator.best_estimator_



#### to put inside models folder
def baseline_classification_xgb(X_train, y_train, X_test, y_test, cv_n_iter=10, sample_weight=None):

    xgb_param_fixed = {
        # use all possible cores for training
        'n_jobs': -1,
        # set number of estimator to a large number
        # and the learning rate to be a small number,
        # we'll let early stopping decide when to stop
        'n_estimators': 300,
        'random_state':2020,
    }

    xgb_hyperparam_options = {
        'max_delta_step': stats.randint(low = 0, high = 5), # setting it to a positive value, might help when class is extremely imbalanced, as it makes the update more conservative
        'max_depth': stats.randint(low = 3, high = 15),
        'colsample_bytree': stats.uniform(loc = 0.7, scale = 0.3),
        'subsample': stats.uniform(loc = 0.7, scale = 0.3),
        'alpha': stats.uniform(loc = 0, scale = 10),
        'lambda': stats.uniform(loc = 0, scale = 10),
        'scale_pos_weight': [(y_train.value_counts()[0] / y_train.value_counts()[1]), 1.0],
        'learning_rate': [0.1, 0.01],
    }

    xgb_fit_params = {
        'eval_metric': ['logloss','auc'],
        'eval_set': [(X_train, y_train), (X_test, y_test)],
        'early_stopping_rounds': 10,
        'verbose': False
    }

    xgb_base = XGBClassifier(**xgb_param_fixed)

    xgb_estimator  = RandomizedSearchCV(
        estimator = xgb_base,
        param_distributions = xgb_hyperparam_options,
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=2020),
        n_iter = cv_n_iter , #n_iter,
        verbose = 1,
        random_state = 2020).fit(X_train, y_train, **xgb_fit_params, sample_weight=sample_weight)

    print('Best score obtained: {0}'.format(xgb_estimator.best_score_))
    print('Best Parameters:')
    for param, value in xgb_estimator.best_params_.items():
        print('\t{}: {}'.format(param, value))
    return xgb_estimator.best_estimator_


def default_classification_xgb(X_train,y_train, X_test, y_test, sample_weight=None):
    # create an xgbost classifier
    xgb = XGBClassifier(n_estimators=100
                        ,colsample_bytree = 0.3
                        ,learning_rate= 0.1
                        ,max_depth= 5
                        ,reg_alpha = 15
                        ,reg_lambda =10,
                       scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1]))
    xgb.fit(X_train, y_train, 
            eval_metric=['logloss','auc'],
            verbose=False,
            eval_set=[((X_train, y_train)),(X_test, y_test)],
            sample_weight= sample_weight)
    return xgb



#### to put inside models folder
def baseline_regression_xgb(X_train, y_train, X_test, y_test, cv_n_iter=10, sample_weight=None):

    xgb_param_fixed = {
        # use all possible cores for training
        'n_jobs': -1,
        # set number of estimator to a large number
        # and the learning rate to be a small number,
        # we'll let early stopping decide when to stop
        'n_estimators': 300,
        'random_state':2020,
    }

    xgb_hyperparam_options = {
        'max_depth': stats.randint(low = 3, high = 15),
        'colsample_bytree': stats.uniform(loc = 0.7, scale = 0.3),
        'subsample': stats.uniform(loc = 0.7, scale = 0.3),
        'alpha': stats.uniform(loc = 0, scale = 10),
        'lambda': stats.uniform(loc = 0, scale = 10),
        'learning_rate': [0.1, 0.01],
    }

    xgb_fit_params = {
        'eval_metric': ['rmse'],
        'eval_set': [(X_train, y_train), (X_test, y_test)],
        'early_stopping_rounds': 10,
        'verbose': False
    }

    xgb_base = XGBRegressor(**xgb_param_fixed)

    xgb_estimator  = RandomizedSearchCV(
        estimator = xgb_base,
        param_distributions = xgb_hyperparam_options,
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=2020),
        n_iter = cv_n_iter , #n_iter,
        verbose = 1,
        random_state = 2020).fit(X_train, y_train, **xgb_fit_params, sample_weight=sample_weight)

    print('Best score obtained: {0}'.format(xgb_estimator.best_score_))
    print('Best Parameters:')
    for param, value in xgb_estimator.best_params_.items():
        print('\t{}: {}'.format(param, value))
    return xgb_estimator.best_estimator_


def default_regression_xgb(X_train,y_train, X_test, y_test, sample_weight=None):
    # create an xgbost classifier
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train
            ,eval_metric=['rmse']
            ,verbose=True
            ,early_stopping_rounds=10
            ,eval_set=[((X_train, y_train)),(X_test, y_test)]
           ,sample_weight=sample_weight)
    return xgb