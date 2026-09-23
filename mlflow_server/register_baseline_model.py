"""Регистрирует baseline-модель первого спринта в MLflow Tracking Server и Model Registry."""
import json
import os
from urllib.parse import quote_plus

import mlflow
import mlflow.sklearn
import pandas as pd
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

# доступы к БД и S3 лежат в .env в корне проекта
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net'

TRACKING_URI = 'http://127.0.0.1:5000'
EXPERIMENT_NAME = 'flats_price_improvement'
RUN_NAME = 'baseline_model'
REGISTRY_MODEL_NAME = 'flats_price_model'

TABLE = 'clean_flats_dataset'
TARGET_COL = 'price'
DROP_COLS = ['id', 'flat_id', 'building_id']
CAT_COLS = ['building_type_int', 'is_apartment', 'studio', 'has_elevator']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# гиперпараметры baseline-модели из первого спринта
MODEL_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 0,
    'thread_count': -1,
    'random_seed': RANDOM_STATE,
}


def load_data():
    """Выгружает очищенный датасет из личной БД."""
    password = quote_plus(os.environ['DB_DESTINATION_PASSWORD'])
    engine = create_engine(
        f"postgresql://{os.environ['DB_DESTINATION_USER']}:{password}"
        f"@{os.environ['DB_DESTINATION_HOST']}:{os.environ['DB_DESTINATION_PORT']}"
        f"/{os.environ['DB_DESTINATION_NAME']}"
    )
    data = pd.read_sql(f'select * from {TABLE}', engine)
    engine.dispose()
    return data


def make_features(df):
    """Убирает id-шники и цену, категории приводит к строкам."""
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    for col in CAT_COLS:
        X[col] = X[col].astype(str)
    return X


def eval_metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(mean_squared_error(y_true, y_pred)) ** 0.5,
        'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def main():
    data = load_data()
    train, test = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    X_train, y_train = make_features(train), train[TARGET_COL]
    X_test, y_test = make_features(test), test[TARGET_COL]
    num_cols = [c for c in X_train.columns if c not in CAT_COLS]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ('num', StandardScaler(), num_cols),
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', CatBoostRegressor(**MODEL_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)

    metrics = eval_metrics(y_test, pipeline.predict(X_test))
    print('Метрики на тесте:', metrics)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    experiment_id = (
        mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        or mlflow.get_experiment(mlflow.create_experiment(EXPERIMENT_NAME))
    ).experiment_id

    with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_params({
            'model_type': 'CatBoostRegressor',
            'table': TABLE,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'n_features': X_train.shape[1],
            'n_train_rows': len(X_train),
            'n_test_rows': len(X_test),
        })
        mlflow.log_metrics(metrics)

        with open('baseline_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact('baseline_metrics.json')
        mlflow.log_artifact(__file__)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path='model',
            signature=infer_signature(X_test, pipeline.predict(X_test)),
            input_example=X_test.head(5),
            registered_model_name=REGISTRY_MODEL_NAME,
        )
        print('run_id:', run.info.run_id, 'experiment_id:', experiment_id)


if __name__ == '__main__':
    main()
