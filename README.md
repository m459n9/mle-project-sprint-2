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
по важности CatBoost, а сам отбор идёт на подвыборке с облегчённой моделью. Дальше четыре
набора-кандидата (forward, backward, их пересечение и объединение) сравниваются по кросс-валидации,
и лучший встраивается в пайплайн через `ColumnSelector`.

* запуски в MLflow: `feature_selection`
* артефакты: `selected_features.json`, график важности признаков, график качества по ходу отбора
* версия модели в реестре: **3**

## Этап 5. Подбор гиперпараметров

Раздел 5 ноутбука.

1. **Optuna** (TPE): 15 trials, каждый trial логируется в MLflow через `MLflowCallback`
   вложенными запусками.
2. **RandomizedSearchCV**: 10 случайных комбинаций по той же сетке, полная таблица переборов
   сохраняется артефактом.

Оба метода перебирают `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `min_data_in_leaf`
и функцию потерь (RMSE или MAE — MAE это наша целевая метрика, поэтому проверяем и обучение
прямо под неё).

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
| Отбор признаков | `feature_selection` | 3 | 1 665 130 | 2 062 512 | 16.23% | 0.648 |
| Подбор гиперпараметров | `final_model` | 4 | **1 639 116** | 2 029 011 | 15.99% | 0.659 |

Итог: MAE лучше baseline на **1.55%** (25 812 ₽), MAPE 16.23% → 15.99%, R² 0.648 → 0.659.

Что показали этапы:

* **Генерация признаков** дала 0.2%. CatBoost и без подсказок вытягивает взаимодействия из исходных
  признаков, поэтому ручные отношения площадей и расстояние до центра добавляют немного. Полезнее
  всего оказались `dist_to_center`, `volume` и полиномы от площадей — они же прошли отбор.
* **Отбор признаков** качество не улучшил (−0.01% к baseline): все четыре набора-кандидата —
  forward, backward, их пересечение и объединение — показали практически одинаковый MAE на
  кросс-валидации (1.737–1.739 млн), разница в пределах шума. Победил forward с 11 признаками
  вместо 50. Ценность этапа не в метрике, а в том, что модель стала компактнее и обучается быстрее.
* **Подбор гиперпараметров** дал основной прирост. Победила Optuna (MAE на кросс-валидации
  1 702 504 против 1 710 246 у RandomizedSearchCV): `depth=9`, `iterations=1200`,
  `learning_rate=0.022`, `l2_leaf_reg=1.08`, `min_data_in_leaf=38`, функция потерь RMSE.
  Обучение прямо под MAE тоже проверялось в обоих методах, но выигрыша не дало.

Признаки финальной модели (11 из 50): `dist_to_center`, `volume`, `latitude`, `longitude`,
`total_area × rooms`, `total_area × ceiling_height`, `total_area × kitchen_area`, `latitude`
(корзины), `rooms × ceiling_height`, `build_year` (корзины), `floor`.

### Куда двигаться дальше

Потолок упирается не в методы, а в данные: R² около 0.66 при 15 признаках без адреса и района.
Что дало бы больше, чем ещё один круг подбора гиперпараметров:

* признаки по локации: район/округ, расстояние до метро, цена за м² по соседним объектам
  (target encoding по географическим кластерам);
* внешние данные — инфраструктура, транспортная доступность;
* Optuna снова упёрлась в верхнюю границу `iterations=1200` при почти минимальном
  `learning_rate` — есть смысл пробовать ещё более длинное обучение с ранней остановкой на
  валидационной выборке.

### Запуски в MLflow

Все запуски — в эксперименте `flats_price_improvement` (id 1): `baseline_model`, `eda`,
`feature_generation`, `feature_selection`, `hyperopt_optuna` (плюс 15 вложенных запусков по trial-ам
от `MLflowCallback`), `hyperopt_random_search`, `final_model`.
