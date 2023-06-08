'''
    Модуль предварительной обработки данных.
    Функционал:
    1) Удаление столбцов с заданным процентом пропусков
    2) Удаление коллинеарных признаков
    3) Удаление строк с пропущенными значениями
    4) Формирование предобработанного датасета
    5) Объединение датасетов
    6) Нормализация значений датасета
    7) Разделение датасета на обучающую и тестовую выборки
'''
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
from app import app
from datetime import datetime


# region Вспомогательные функции
def normalize_dataset(dataset):
    """
    Нормализует значения датасета.
    :param dataset: Датасет для нормализации.
    :return: Нормализованный датасет и объект нормализации.
    """
    scaler = preprocessing.StandardScaler()
    transformed = scaler.fit_transform(dataset)
    # Сохранение scaler в файл
    filepath = os.path.join(app.config['DATA_FOLDER'], 'scaler.pkl')
    with open(filepath, 'wb') as file:
        pickle.dump(scaler, file)
    return pd.DataFrame(transformed, columns=dataset.columns, index=dataset.index)

from sklearn import preprocessing


def denormalize_dataset(normalized_dataset):
    """
    Денормализует значения датасета.
    :param normalized_dataset: Нормализованный датасет.
    :return: Денормализованный датасет.
    """
    try:
        # Загрузка scaler из файла
        filepath = os.path.join(app.config['DATA_FOLDER'], 'scaler.pkl')
        with open(filepath, 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Файл scaler.pkl не найден.")

    if scaler is None:
        raise ValueError("Не удалось загрузить scaler из файла scaler.pkl.")

    denormalized = scaler.inverse_transform(normalized_dataset)
    return pd.DataFrame(denormalized, columns=normalized_dataset.columns, index=normalized_dataset.index)

def denormalize_temp(normalized_dataset):
    """
    Денормализует значения датасета.
    :param normalized_dataset: Нормализованный датасет.
    :return: Денормализованный датасет.
    """
    new_dataset = pd.concat(
        [pd.DataFrame(None, index=normalized_dataset.index, columns=['new_col']), normalized_dataset,
         pd.DataFrame(columns=[f'extra_{i + 1}' for i in range(7)])], axis=1)

    denormalized = denormalize_dataset(new_dataset)
    return denormalized['temp_audience']

# Удалить столбцы с заданным процентом пропусков
def del_cols_with_skips(data=None, cols=None, percentage_skips=50):
    """
    Удаляет столбцы с заданным процентом пропусков.
    :param data: Датафрейм.
    :param cols: Список столбцов для проверки. Если None, будут проверены все столбцы датафрейма (по умолчанию: None).
    :param percentage_skips: Пороговое значение процента пропусков. Если столбец имеет процент пропусков выше порога, он будет удален (по умолчанию: 50).
    :return: None
    """
    if cols is None:
        cols = data.columns
    elif not isinstance(cols, list):
        raise ValueError("Неправильный тип для параметра 'cols'. Ожидалось None или список.")
    for col in cols:
        # Вычисляем процент пропусков в столбце
        pct_missing = round(np.mean(data[col].isnull()) * 100)
        # print('{} - {}%'.format(col, round(pct_missing*100)))
        if pct_missing > percentage_skips:
            data.drop(col, axis=1, inplace=True)


def del_collinear_cols(data=None, cols=None, threshold=90):
    """
    Удаляет коллинеарные (сильно коррелирующие) признаки.
    :param data: Датафрейм.
    :param cols: Список столбцов для расчета корреляции. Если None, будет использованы все столбцы датафрейма. (по умолчанию: None)
    :param threshold: Пороговое значение корреляции в процентах. Если два признака имеют корреляцию выше порога, один из них будет удален. (по умолчанию: 90)
    :return: None
    """
    if cols is None:
        correlation = data.corr()
    elif not isinstance(cols, list):
        raise ValueError("Неправильный тип для параметра 'cols'. Ожидалось None или список.")
    else:
        correlation = data[cols].corr()
    drop_columns = []
    for col in correlation.columns:
        for col2 in correlation.columns:
            if (col != col2) and (col not in drop_columns) and (col2 not in drop_columns):
                if round(correlation[col][col2] * 100) > threshold:
                    drop_columns.append(col2)
    data.drop(drop_columns, axis=1, inplace=True)


def del_rows_with_skips(data):
    """
    Удаляет строки с пропущенными значениями.
    :param data: Датафрейм.
    :return: None
    """
    for col in data.columns:
        data[col].replace('', np.nan, inplace=True)
        data.dropna(subset=[col], inplace=True)


# Объединить датафреймы
def merged_dataframes(sensor_info, weather_info, logging=True):
    """
    Объединяет два датафрейма.
    :param sensor_info: Первый датафрейм.
    :param weather_info: Второй датафрейм.
    :param logging: Флаг для логирования (по умолчанию: True).
    :return: Объединенный датафрейм.
    """
    if logging:
        print('Объединение датасетов')
    merged = sensor_info.merge(weather_info, left_on="date_time", right_on="date_time", how="left")
    merged.drop_duplicates(keep="first", inplace=True)
    return merged


# Преобразовать строку в дату и время
def str_to_date_time(date, format="%Y-%m-%d %H:%M:%S"):
    if type(date) == str:
        date = datetime.strptime(date, format)
    return date
# endregion


# region Основные функции
def preprocess_dataset(dataframe, cols=None, logging=True):
    """
    Предобрабатывает датасет.
    :param dataframe: Исходный датасет.
    :param cols: Список столбцов, которые нужно предобработать (по умолчанию None, т.е. все столбцы).
    :param logging: Флаг для логирования процесса (по умолчанию True).
    :return: Обработанный датасет.
    """
    if logging:
        print('Предобработка данных\n\tУдаление столбцов с заданным процентом пропусков')
    try:
        del_cols_with_skips(dataframe, cols, 50)
    except Exception as e:
        print(f"Ошибка при удалении столбцов с пропусками: {str(e)}")
    # Отобразить корреляционную матрицу признаков о погоде
    # Visualization.plot_correlation_matrix(dataframe)
    if logging:
        print('\tУдаление столбцов с заданным процентом корреляции')
    try:
        del_collinear_cols(dataframe, cols, 90)
    except Exception as e:
        print(f"Ошибка при удалении столбцов с заданным процентом корреляции: {str(e)}")
    if logging:
        print('\tУдаление пустых строк')
    try:
        del_rows_with_skips(dataframe)
    except Exception as e:
        print(f"Ошибка при удалении пустых строк: {str(e)}")
    return dataframe


# Разделение датасета на обучающую и тестовую выборки
def get_train_test_for_train_from_scratch(dataset, target_column, test_size=0.333, random_state=None):
    index_y = list(dataset.columns).index(target_column)
    indexes_x = [i for i in range(len(dataset.columns)) if i != index_y]
    y = dataset[dataset.columns[index_y]]
    x = dataset[dataset.columns[indexes_x]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False, stratify=None,
                                                        random_state=random_state)
    train_index = y_train.index
    test_index = y_test.index
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    return x_train, y_train, x_test, y_test, train_index, test_index

def get_train_test_for_online_learning(dataset, target_column):
    index_y = list(dataset.columns).index(target_column)
    indexes_x = [i for i in range(len(dataset.columns)) if i != index_y]
    y = dataset[dataset.columns[index_y]]
    x = dataset[dataset.columns[indexes_x]]
    train_index = x.index
    test_index = x.index
    x_train = x.values
    y_train = y.values
    x_test = x.values
    y_test = y.values
    return x_train, y_train, x_test, y_test, train_index, test_index

# endregion
