# Улучшение baseline-модели оценки стоимости квартир

Проект второго спринта. Берём baseline-модель из первого спринта (CatBoost на очищенном датасете
Яндекс Недвижимости) и улучшаем её: генерация признаков, отбор признаков, подбор гиперпараметров.
Все эксперименты пишутся в MLflow, каждая версия модели регистрируется в MLflow Model Registry.

Целевая метрика — **MAE** на тестовой выборке: средняя ошибка предсказания цены в рублях.
Дополнительно считаем RMSE, MAPE и R².

**Бакет S3:** `s3-student-mle-20260908-7056ef1b44`

**Эксперимент в MLflow:** `flats_price_improvement`, id `1`
**Модель в Model Registry:** `flats_price_model`

### Стек

Python 3.10, PostgreSQL, S3 (Yandex Object Storage), MLflow, scikit-learn, CatBoost,
autofeat, mlxtend, Optuna, pandas, seaborn.

### Структура репозитория

```
mlflow_server/
    run_mlflow_server.sh        запуск MLflow Tracking Server и Model Registry
    register_baseline_model.py  регистрация baseline-модели первого спринта
    .env.example                шаблон переменных окружения
model_improvement/
    project_template_sprint_2.ipynb   ноутбук с этапами 2-5
requirements.txt
```

## Установка

```bash
git clone https://github.com/m459n9/mle-project-sprint-2.git
cd mle-project-sprint-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

В корне проекта нужен файл `.env` с доступами — скопируйте `mlflow_server/.env.example`
и заполните своими значениями:

```bash
cp mlflow_server/.env.example .env
```

* `DB_DESTINATION_*` — личная база PostgreSQL: в ней лежит таблица `clean_flats_dataset`
  с данными для обучения, она же используется как backend store для MLflow;
* `S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — бакет для артефактов MLflow.

Значения с пробелами и спецсимволами берите в кавычки — файл читается и через `source`.

## Этап 1. MLflow и регистрация baseline-модели

Shell-скрипт поднимает Tracking Server и Model Registry: метаданные пишутся в PostgreSQL,
артефакты — в S3, переменные `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` выставляются внутри скрипта.

```bash
bash mlflow_server/run_mlflow_server.sh
```

UI: `http://127.0.0.1:5000`.

Регистрация baseline-модели — обучение пайплайна первого спринта (OneHotEncoder + StandardScaler
+ CatBoostRegressor), логирование параметров, метрик на тесте, сигнатуры, окружения и артефактов,
запись в Model Registry:

```bash
python mlflow_server/register_baseline_model.py
```

* скрипты: [`run_mlflow_server.sh`](mlflow_server/run_mlflow_server.sh),
  [`register_baseline_model.py`](mlflow_server/register_baseline_model.py)
* запуск в MLflow: `baseline_model`
* версия модели в реестре: **1**

## Этап 2. EDA

Раздел 2 ноутбука [`project_template_sprint_2.ipynb`](model_improvement/project_template_sprint_2.ipynb).

Что сделано: загрузка данных из личной БД через pandas, приведение типов (булевы флаги в int),
проверка пропусков и дублей, распределения числовых и категориальных признаков, анализ целевой
переменной (в том числе в логарифмическом масштабе), корреляции Спирмена, связь цены с площадью,
числом комнат, типом дома и географией, цена за квадратный метр.

Ключевые выводы — в markdown-ячейке раздела 2.5; тот же текст пишется в `eda_conclusions.md`.

* запуск в MLflow: `eda`
* артефакты: ноутбук, `eda_conclusions.md`, графики EDA

## Этап 3. Генерация признаков

Раздел 3 ноутбука.

1. Ручные признаки по гипотезам из EDA: `area_per_room`, `living_ratio`, `kitchen_ratio`,
   `other_area`, `volume`, `floor_ratio`, `is_first_floor`, `is_last_floor`, `building_age`,
   `flats_per_floor`, `dist_to_center`.
2. `ColumnTransformer`: `OneHotEncoder` для категорий, `PolynomialFeatures(degree=2)` для площадей
   и числа комнат, `KBinsDiscretizer` для года постройки и координат, `StandardScaler` для остального.
   Корзины добавляются к исходным значениям, а не заменяют их.
3. Всё вместе собрано в `sklearn.Pipeline`: ручные признаки → `ColumnTransformer` → CatBoost.
4. Автоматическая генерация признаков `autofeat` на подвыборке, найденные формулы сохраняются
   артефактом `autofeat_features.json`.
5. Обучение новой версии модели, замер скорости обучения и предсказания.

* запуск в MLflow: `feature_generation`
* версия модели в реестре: **2**

## Этап 4. Отбор признаков

Раздел 4 ноутбука.

Два метода из `mlxtend`: forward selection и backward selection (`SequentialFeatureSelector`).
Чтобы перебор укладывался в разумное время, кандидаты сначала сужаются до топ-20 признаков
по важности CatBoost, а сам отбор идёт на подвыборке с облегчённой моделью. Финальный набор —
пересечение результатов обоих методов, он встраивается в пайплайн через `ColumnSelector`.

* запуски в MLflow: `feature_selection`
* артефакты: `selected_features.json`, график важности признаков, график качества по ходу отбора
* версия модели в реестре: **3**

## Этап 5. Подбор гиперпараметров

Раздел 5 ноутбука.

1. **Optuna** (TPE): 15 trials, каждый trial логируется в MLflow через `MLflowCallback`
   вложенными запусками.
2. **RandomizedSearchCV**: 10 случайных комбинаций по той же сетке плюс выбор функции потерь
   (RMSE или MAE), полная таблица переборов сохраняется артефактом.

Лучший по кросс-валидации набор параметров идёт в финальную модель, она обучается на полном train
и сравнивается с предыдущими версиями на тесте.

* запуски в MLflow: `hyperopt_optuna`, `hyperopt_random_search`, `final_model`
* версия модели в реестре: **4**

## Результаты

### Метрики на тестовой выборке

| Этап | Запуск в MLflow | Версия | MAE, ₽ | RMSE, ₽ | MAPE | R² |
|---|---|---|---|---|---|---|
| Baseline первого спринта | `baseline_model` | 1 | 1 664 928 | 2 062 755 | 16.23% | 0.648 |
| Генерация признаков | `feature_generation` | 2 | 1 661 530 | 2 057 278 | 16.18% | 0.649 |
| Отбор признаков | `feature_selection` | 3 | 1 663 945 | 2 061 066 | 16.23% | 0.648 |
| Подбор гиперпараметров | `final_model` | 4 | **1 633 113** | 2 023 947 | 15.93% | 0.661 |

Итог: MAE лучше baseline на **1.9%** (31 815 ₽), MAPE 16.23% → 15.93%, R² 0.648 → 0.661.

Что показали этапы:

* **Генерация признаков** дала совсем небольшой прирост (0.2%): CatBoost и без ручных подсказок
  вытягивает взаимодействия из исходных признаков. Полезнее всего оказались `dist_to_center`,
  `volume` и полиномы от площадей — именно они потом прошли отбор.
* **Отбор признаков** сократил набор с 50 признаков до 10 и метрику чуть ухудшил (на 0.06% к baseline),
  зато модель стала обучаться в 3 раза быстрее (4.9 с против 14.9 с). Дальше гиперпараметры
  подбирались уже на этом компактном наборе.
* **Подбор гиперпараметров** дал основной прирост. Победила Optuna: `depth=9`, `iterations=700`,
  `learning_rate=0.067`, `l2_leaf_reg=8.26`, `min_data_in_leaf=34` (MAE на кросс-валидации 1 703 367
  против 1 707 219 у RandomizedSearchCV).

Отобранные признаки финальной модели: `dist_to_center`, `volume`, `latitude`, `longitude`,
`total_area × ceiling_height`, `total_area × kitchen_area`, `rooms × ceiling_height`,
`build_year` (корзины), `floor`, `total_area²`.

### Запуски в MLflow

Все запуски — в эксперименте `flats_price_improvement` (id 1): `baseline_model`, `eda`,
`feature_generation`, `feature_selection`, `hyperopt_optuna` (плюс 15 вложенных запусков по trial-ам
от `MLflowCallback`), `hyperopt_random_search`, `final_model`.
