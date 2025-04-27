import time

import mlflow
import optuna
import torch
from torchmetrics.retrieval import RetrievalNormalizedDCG
from xgboost import XGBRanker


class ModellingConfig:
    # num_users: int
    # num_movies: int
    # num_ratings: int
    num_latent_factors: int = None
    learning_rate: float = None
    num_epochs: int = None
    batch_size: int = None
    num_negative_samples: int = 5
    user_numerical_features = ["age"]
    user_categorical_features = ["gender", "occupation"]
    movie_categorical_features = ["year"]
    movie_numerical_features = []
    label_col = "label"
    qid_col = "qid"

    mlflow_experiment_name = "XGBRanker_Optimization"
    mlflow_run_name = "XGBRanker_Tuning"
    n_trials = 5
    optuna_direction = "maximize"
    optuna_mode = "sampler"  #  "n_trials"
    optuna_duration = 2

    eval_metrics = ["ndcg@1", "ndcg@10"]
    n_estimators = 200
    early_stopping_rounds = 20
    random_state = 42
    ndcg_top_k = 10


# model_config = ModellingConfig()


class XGBRankerTrainer:
    def __init__(self, train, val, model_config: ModellingConfig):
        self.model_config = model_config
        self.features = (
            model_config.user_numerical_features
            + model_config.movie_numerical_features
            + model_config.user_categorical_features
            + model_config.movie_categorical_features
        )
        self.eval_metrics = model_config.eval_metrics
        self.label_col = model_config.label_col
        self.qid_col = model_config.qid_col
        self.n_estimators = model_config.n_estimators
        self.early_stopping_rounds = model_config.early_stopping_rounds
        self.random_state = model_config.random_state

        self.ndcg_top_k = model_config.ndcg_top_k
        self.rndcg = RetrievalNormalizedDCG(top_k=model_config.ndcg_top_k)

        self.experiment_name = model_config.mlflow_experiment_name
        self.run_name = model_config.mlflow_run_name
        self.n_trials = model_config.n_trials
        self.best_n_estimator = None
        self.best_eval_metric = None

        self.train = train
        self.val = val

        self.best_n_trial = None

    # Define Optuna objective function
    def objective(self, trial):
        # Start MLflow nested run
        with mlflow.start_run(nested=True):
            # Define hyperparameters to optimize
            params = {
                "objective": "rank:ndcg",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state": self.random_state,
            }

            # Log parameters to MLflow
            mlflow.log_params(params)

            # Create and train model
            model = XGBRanker(
                n_estimators=self.n_estimators,
                eval_metric=self.eval_metrics,
                early_stopping_rounds=self.early_stopping_rounds,
                **params,
            )

            # Track training time
            start_time = time.time()

            # Fit model with early stopping
            model.fit(
                self.train[self.features],
                self.train[self.label_col],
                qid=self.train[self.qid_col],
                eval_set=[
                    (self.train[self.features], self.train[self.label_col]),
                    (self.val[self.features], self.val[self.label_col]),
                ],
                eval_qid=[self.train[self.qid_col], self.val[self.qid_col]],
                verbose=False,
            )

            training_time = time.time() - start_time

            # Get predictions
            val_preds = model.predict(self.val[self.features])

            # Calculate NDCG@10
            ndcg_10 = self.rndcg(
                torch.tensor(val_preds), torch.tensor(self.val[self.label_col]), torch.tensor(self.val[self.qid_col])
            )

            # ndcg_10 = calculate_ndcg(y_val.values, val_preds, groups_val, k=10)
            # Log metrics to MLflow
            mlflow.log_metric(f"ndcg_at_{self.ndcg_top_k}", ndcg_10)
            mlflow.log_metric("training_time", training_time)

            # Store best iteration
            trial.set_user_attr("best_iteration", model.best_iteration)
            return ndcg_10  # Optimize for NDCG@10

    def create_optuna_study(self):
        if self.model_config.optuna_mode == "n_trials":
            study = optuna.create_study(direction=self.model_config.optuna_direction)

        elif self.model_config.optuna_mode == "sampler":
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction=self.model_config.optuna_direction, sampler=sampler)
        return study

    def optimize_optuna(self, objective_func):
        if self.model_config.optuna_duration is not None and self.model_config.optuna_mode == "sampler":
            tic = time.time()
            while time.time() - tic < self.model_config.optuna_duration:
                self.study.optimize(objective_func, n_trials=1)
        else:
            self.study.optimize(objective_func, n_trials=self.n_trials)

    def hyperparameter_tuning(self):
        # Set up MLflow experiment
        experiment_name = self.experiment_name if self.experiment_name else "XGBRanker_Optimization"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception as e:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            print(e)

        self.experiment_id = experiment_id

        X_train = self.train[self.features]
        y_train = self.train[self.label_col]
        qid_train = self.train[self.qid_col]

        X_val = self.val[self.features]
        y_val = self.val[self.label_col]
        qid_val = self.val[self.qid_col]

        # Run the optimization
        with mlflow.start_run(experiment_id=experiment_id, run_name="XGBRanker_Tuning"):
            # Configure the study
            self.study = self.create_optuna_study()

            # Run optimization
            self.optimize_optuna(self.objective)

            # Log best parameters and score
            best_params = self.study.best_params
            best_value = self.study.best_value

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_ndcg_at_10", best_value)

            # Train final model with best parameters
            final_params = {
                "objective": "rank:ndcg",
                "n_estimators": self.n_estimators,  # 400 ,  # More estimators for final model
                # "n_estimators": self.study.best_trial.user_attrs.get(
                #     "best_iteration"
                # ),  # More estimators for final model
                "eval_metric": self.eval_metrics,
                "early_stopping_rounds": self.early_stopping_rounds,
                "random_state": self.random_state,
                **best_params,
            }

            final_model = XGBRanker(**final_params)
            final_model.fit(
                X_train,
                y_train,
                qid=qid_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_qid=[qid_train, qid_val],
                verbose=False,
            )

            # Log the final model
            # mlflow.xgboost.log_model(final_model, "xgb_ranker_model")

            # Make predictions on validation set
            val_preds = final_model.predict(X_val)
            # Calculate NDCG@10
            final_ndcg = self.rndcg(torch.tensor(val_preds), torch.tensor(y_val), torch.tensor(qid_val))
            mlflow.log_metric("final_ndcg_at_10", final_ndcg)

            print(f"Best NDCG@10: {best_value:.4f}")
            print(f"Final NDCG@10: {final_ndcg:.4f}")
            print(f"Best parameters: {best_params}")

        mlflow.end_run()


if __name__ == "__main__":
    model_config = ModellingConfig()
    train_transformed = None
    val_transformed = None
    trainer = XGBRankerTrainer(train_transformed, val_transformed, model_config)
    trainer.hyperparameter_tuning()
