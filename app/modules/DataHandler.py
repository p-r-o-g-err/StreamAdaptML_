"""
    Модуль работы с данными.
    Функционал:
        1) Сохранение, считывание и обновление датасета
        2) Сохранение, считывание настроек
        3) Сохранение, считывание и отправка модели
"""
import keras
import pandas as pd
import json
import os
from app import app
from app.modules import SensorData, MeteostatData, DataPreprocessing
import numpy as np
from datetime import timedelta
from werkzeug.datastructures import FileStorage
from flask import send_file, redirect, url_for
from keras.models import Model


# region Работа с датасетом
def save_dataset(dataset, filename='dataset.csv'):
    """
    Сохраняет датасет в формате CSV.
    :param dataset: Датафрейм.
    :param filename: Название файла (по умолчанию 'dataset.csv').
    """
    # Сохранить каждый датчик отдельно, если передан список DataFrame'ов
    if type(dataset) is list:
        for d in dataset:
            d = d.reset_index()
            d.to_csv(d.columns[-1] + '_data.csv', encoding='utf-8', index=False)
    # Сохранить все датчики в одном csv файле
    else:
        data = dataset.reset_index()
        filepath = os.path.join(app.config['DATA_FOLDER'], filename)
        data.to_csv(filepath, encoding='utf-8', index=False)


def read_dataset(filename='dataset.csv'):
    """
    Считывает датасет из CSV файла.
    :param filename: Название файла (по умолчанию 'dataset.csv').
    :return: DataFrame, если файл по указанному пути существует, иначе None.
    """
    try:
        filepath = os.path.join(app.config['DATA_FOLDER'], filename)
        df = pd.read_csv(filepath)
        # Преобразование столбца date_time из текстового типа к datetime
        df['date_time'] = pd.to_datetime(df['date_time'])
        return df.set_index('date_time')
    except FileNotFoundError:
        print("Файл не найден. Укажите корректный путь к файлу.")
        return None


def update_dataset(logging=True):
    """
    Обновляет датасет.
    :param logging: Флаг для включения или отключения логирования. По умолчанию установлено значение True.
    :return: Возвращает True, если обновление датасета выполнено успешно, иначе False.
    """
    # Получить текущий датасет
    dataset = read_dataset()
    # Если датасет найден
    if dataset is not None:
        try:
            # Получить обновленный предобработанный датасет
            new_dataset, number_new_rows = get_merged_sensor_weather_data_update(dataset, logging)
            print(f'Новых строк: {number_new_rows}')
            if number_new_rows == 0:
                return False
            # Сохранить датасет
            save_dataset(new_dataset)
            return True
        except Exception as e:
            print(f"Ошибка при обновлении датасета: {str(e)}")
            return False
    # Если датасет не найден
    else:
        try:
            # Получить предобработанный датасет
            dataset = get_merged_sensor_weather_data(start='2021-01-02 00:00:00', res_period='10T',
                                                     preprocessing_weather_data=True, fill_in_the_gaps=True,
                                                     logging=logging)
            # Сохранить датасет
            save_dataset(dataset)
            return True
        except Exception as e:
            print(f"Ошибка при обновлении датасета: {str(e)}")
            return False


def get_dataset_for_model(audience_name=None, start_date=None, end_date=None, normalize=False):
    """
    Создает набор данных для модели.
    :param audience_name: Название столбца для прогнозирования. Если не указано, выбирается столбец
                          с наименьшим количеством пропусков среди столбцов с температурой кабинетов.
    :param start_date: Начальная дата для фильтрации данных. (по умолчанию: None)
    :param end_date: Конечная дата для фильтрации данных. (по умолчанию: None)
    :param normalize: Флаг для нормализации данных. (по умолчанию: False)
    :return: Предобработанный набор данных.
    """
    # Чтение данных
    dataset = read_dataset()
    if dataset is not None:
        # Формирование списка столбцов с метеоданными
        weather_cols = [col for col in dataset.columns if not col.endswith('_temp')]
        # Формирование столбца для прогнозирования
        if audience_name is None:
            # Выбор столбца с наименьшим количеством пропусков среди столбцов с температурой кабинетов
            room_number_wall = \
                (dataset.count() / len(dataset)).sort_values(ascending=False).filter(regex='wall_temp').index[0]
        else:
            if audience_name in dataset.columns:
                room_number_wall = audience_name
            else:
                raise ValueError(f'Передано неизвестное название столбца: {audience_name}')
        # Объединение показаний датчиков температуры в кабинете с метеоданными
        dataset = DataPreprocessing.merged_dataframes(dataset.filter(regex=room_number_wall.split('_')[0]),
                                                      dataset[weather_cols], logging=False)

        # Предобработка полученного датасета
        if start_date is not None:
            dataset = dataset[dataset.index > start_date]
        if end_date is not None:
            dataset = dataset[dataset.index < end_date]
        if normalize:
            dataset = DataPreprocessing.normalize_dataset(dataset)
        return dataset
    return None


# region Вспомогательные функции
def verifyExt(filename, exts):
    """
    Проверяет расширение файла на то, что оно допустимо.
    :param filename: название файла.
    :param exts: список расширений.
    :return: True, если расширение допустимо, иначе False.
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in set(exts)


def get_merged_sensor_weather_data(start, res_period="1H", preprocessing_weather_data=True,
                                   fill_in_the_gaps=True, logging=True):
    """
    Получает данные о датчиках и метеоданные, выполняет предобработку метеоданных (по желанию) и объединяет их
    в один датафрейм.
    :param start: Начальная дата и время для получения данных.
    :param res_period: Период ресэмплинга метеоданных (по умолчанию '1H').
    :param preprocessing_weather_data: Флаг для выполнения предобработки метеоданных (по умолчанию True).
    :param fill_in_the_gaps: Флаг для заполнения пропущенных значений (по умолчанию True).
    :param logging: Флаг для логирования процесса (по умолчанию True).
    :return: Датафрейм, содержащий объединенные данные о датчиках и погоде.
    """
    if logging:
        print("Получение информации о датчиках")
    sensor_data = SensorData.get_sensor_info(start, res_period=res_period, fill_in_the_gaps=fill_in_the_gaps,
                                             logging=logging)
    if logging:
        print("Получение метеоданных")
    weather_data = MeteostatData.get_weather_info(start=start, end=sensor_data.index[-1], res_period=res_period,
                                                  logging=logging)
    if preprocessing_weather_data:
        if logging:
            print("Предобработка метеоданных")
        DataPreprocessing.preprocess_dataset(weather_data, logging=logging)
    if logging:
        print("Объединение данных датчиков и метеоданных")
    dataset = DataPreprocessing.merged_dataframes(weather_data, sensor_data, logging=logging)
    return dataset


def get_merged_sensor_weather_data_update(dataset, fill_in_the_gaps=True, logging=True):
    """
    Получает обновленный предобработанный датасет.
    :param dataset: Исходный датасет.
    :param fill_in_the_gaps: Флаг для заполнения пропущенных значений (по умолчанию True).
    :param logging: Флаг для логирования процесса (по умолчанию True).
    :return: Обновленный датасет.
    """
    if logging:
        print("Обновление датасета")
    if dataset.index.dtype != np.dtype('datetime64[ns]'):
        raise 'Исходный датасет не имеет индексов даты и времени'
    if logging:
        print("Получение новой информации о датчиках")
    # Определение периода res_period на основе переданного датасета
    res_period = pd.infer_freq(dataset.index)
    # Определение start для получения информации (должно быть dataset.index[-1] + delta, равная res_period)
    # Определить значение времени, соответствующее res_period
    if res_period.endswith('T'):
        minutes = int(res_period[:-1])
        delta = timedelta(minutes=minutes)
    elif res_period.endswith('H'):
        hours = int(res_period[:-1])
        delta = timedelta(hours=hours)
    elif res_period.endswith('D'):
        days = int(res_period[:-1])
        delta = timedelta(days=days)
    else:
        raise ValueError("Неподдерживаемый формат res_period")

    new_sensor_data = SensorData.get_sensor_info(start=dataset.index[-1] + delta, res_period=res_period,
                                                 fill_in_the_gaps=fill_in_the_gaps, logging=logging)
    if logging:
        print("Получение новых метеоданных")
    new_weather_data = MeteostatData.get_weather_info(start=dataset.index[-1], end=new_sensor_data.index[-1],
                                                      res_period=res_period, logging=logging)
    if logging:
        print("Удаление столбцов, которых нет в исходном датасете")
    deleted_columns = list(set(new_weather_data.columns) - set(dataset.columns))
    new_weather_data.drop(deleted_columns, axis=1, inplace=True)
    if logging:
        print("Объединение данных датчиков и метеоданных")
    new_dataset = DataPreprocessing.merged_dataframes(new_weather_data, new_sensor_data, logging=logging)
    if logging:
        print("Объединение старого и нового датасета")
    concat_dataset = pd.concat([dataset, new_dataset], axis=0)
    number_new_rows = len(concat_dataset) - len(dataset)
    return concat_dataset, number_new_rows
# endregion

# endregion

# region Работа с настройками
def read_settings():
    """
    Считывает настройки обучения из файла settings.json.
    :return: настройки в виде словаря или None, если возникла ошибка.
    """
    # Проверяем, существует ли файл настроек
    if not os.path.exists(app.config['SETTINGS_FULLNAME']):
        # Если файл не существует, создаем его и записываем в него словарь со стандартными настройками
        with open(app.config['SETTINGS_FULLNAME'], 'w', encoding='utf-8') as f:
            default_settings = {
                'data_shift_detection_method': 'ADWIN',
                'training_method': 'mini_batch_online_learning',
                'training_method_with_data_shift': 'autofit',
                'window_size': 10
            }
            json.dump(default_settings, f)
    # Считываем содержимое файла
    with open(app.config['SETTINGS_FULLNAME'], mode='r', encoding='utf-8') as file:
        try:
            settings = json.load(file)
        except Exception as e:
            print(f'Ошибка считывания "settings.json". Причина: {e}')
            settings = None
    return settings


def update_settings(data_shift_detection_method=None,
                    training_method=None,
                    training_method_with_data_shift=None,
                    window_size=None):
    """
    Обновляет настройки в файле settings.json.
    :param data_shift_detection_method: метод обнаружения сдвига данных.
    :param training_method: метод обучения ММО при отсутствии сдвига данных.
    :param training_method_with_data_shift: метод обучения ММО при наличии сдвига данных.
    :param window_size: размер окна потока данных.
    :return:
    """
    settings = read_settings()
    if settings is not None:
        # Сохраняем настройки
        with open(app.config['SETTINGS_FULLNAME'], 'w', encoding='utf-8') as f:
            if data_shift_detection_method is not None:
                settings['data_shift_detection_method'] = data_shift_detection_method
            if training_method is not None:
                settings['training_method'] = training_method
            if training_method_with_data_shift is not None:
                settings['training_method_with_data_shift'] = training_method_with_data_shift
            if window_size is not None:
                settings['window_size'] = window_size
            json.dump(settings, f, indent=4, ensure_ascii=False)


# endregion

# region Работа с моделью
def save_model(file: FileStorage):
    """
    Загрузка модели на сервер.
    :param file: файл, выбранный пользователем для загрузки.
    """
    # Если файл выбран
    if file:
        # Если формат файла '.keras', то он сохраняется на сервере в указанной директории
        if verifyExt(file.filename, ['keras']):
            # Извлечение имени файла
            filename = file.filename
            # Удаление ранее загруженной модели
            for f in [f for f in os.listdir(app.config['MODELS_FOLDER'])]:
                os.remove(os.path.join(app.config['MODELS_FOLDER'], f))
            # Сохранение модели на сервере
            file.save(os.path.join(app.config['MODELS_FOLDER'], filename))
            # Имя файла сохраняется в конфигурации приложения
            app.config['LOADED_MODEL'] = file.filename
            print(f'Модель {filename} успешно загружена!')
        else:
            print('Разрешена только загрузка файлов формата ".keras"!')


def save_model(model: Model):
    """
    Сохранение обученной модели на сервере.
    :param model: Обученная модель.
    """
    models_files = os.listdir(app.config['MODELS_FOLDER'])
    # Если в папке с моделью есть модель
    if len(models_files) > 0:
        # Формируем полное имя модели
        filename = os.path.join(app.config['MODELS_FOLDER'], models_files[0])
    else:
        print("Модель ранее не была загружена")
        filename = os.path.join(app.config['MODELS_FOLDER'], 'temp_prediction_model1.keras')
    model.save(filename)
    print(f"Модель \'{filename}\' успешно сохранена")


def upload_model():
    """
    Отправка модели пользователю.
    :return: ответ пользователю на запрос получения модели.
    """
    models_files = os.listdir(app.config['MODELS_FOLDER'])
    # Если в папке с моделью есть модель
    if len(models_files) > 0:
        # Формируем полное имя модели
        filename = os.path.join(app.config['MODELS_FOLDER'], models_files[0])
        print(filename)
        # Отправляем модель пользователю
        return send_file(filename, as_attachment=True)
    else:
        # Если модель не найдена, перезагружаем страницу
        return redirect(url_for('index'))


def read_model():
    """
    Считывание модели.
    :return: модель в формате ".keras".
    """
    # Формируем полное имя модели
    models_files = os.listdir(app.config['MODELS_FOLDER'])
    # Если в папке с моделью есть модель
    if len(models_files) > 0:
        filename = os.path.join(app.config['MODELS_FOLDER'], models_files[0])
        loaded_model = keras.models.load_model(filename)
        print(f"Модель \'{filename}\' успешно загружена")
        return loaded_model
    else:
        print(f"Ошибка считывания модели. Модель не загружена на сервер.")
# endregion
