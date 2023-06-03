"""
    Модуль работы с данными.
    Функционал:
        1) Сохранение, считывание и обновление датасета
        2) Сохранение, считывание настроек
        3) Сохранение, считывание и отправка модели
"""

import pandas as pd
import json
import os
from app import app
from app.modules import SensorData, MeteostatData, DataPreprocessing, DataDriftDetector
import numpy as np
from datetime import timedelta
import pickle

from river import drift


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
            return True  # 210 316 412 420 СФ
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
        dataset = DataPreprocessing.merged_dataframes(dataset.filter(regex=room_number_wall.split('_')[0]), dataset[weather_cols], logging=False)

        # Предобработка полученного датасета
        if start_date is not None:
            dataset = dataset[dataset.index > start_date]
        if end_date is not None:
            dataset = dataset[dataset.index < end_date]
        if normalize:
            dataset, scaler = DataPreprocessing.normalize_dataset(dataset)
            # Сохранение scaler в файл
            filepath = os.path.join(app.config['DATA_FOLDER'], 'scaler.pkl')
            with open(filepath, 'wb') as file:
                pickle.dump(scaler, file)
        return dataset
    return None


# region Вспомогательные функции
def verifyExt(filename, exts):
    """
    Проверить расширение файла на то, что оно допустимо.
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
    Прочитать настройки обучения из файла settings.json
    :return: настройки в виде словаря или None, если возникла ошибка
    """
    # Проверяем, существует ли файл настроек
    if not os.path.exists(app.config['SETTINGS_FULLNAME']):
        # Если файл не существует, создаем его и записываем в него словарь со стандартными настройками
        with open(app.config['SETTINGS_FULLNAME'], 'w', encoding='utf-8') as f:
            default_settings = {
                'data_shift_detection_method': 'ADWIN',
                'training_method': 'mini_batch_online_learning',
                'training_method_with_data_shift': 'autofit'
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
                    training_method_with_data_shift=None):
    """
    Обновить настройки в файле settings.json
    :param data_shift_detection_method: метод обнаружения сдвига данных
    :param training_method: метод обучения МО при отсутствии сдвига данных
    :param training_method_with_data_shift: метод обучения МО при наличии сдвига данных
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
            json.dump(settings, f, indent=4, ensure_ascii=False)


# endregion

# region Работа с моделью
from werkzeug.datastructures import FileStorage
from flask import send_file, redirect, url_for, jsonify


def save_model(file: FileStorage):
    '''
    Загрузка модели на сервер
    :param file: файл, выбранный пользователем для загрузки
    :return:
    '''
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


def upload_model():
    '''
    Отправка модели пользователю
    :return: ответ пользователю на запрос получения модели
    '''
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
    '''
    Считывание модели
    :return: модель в формате ...
    '''
    pass

# endregion





# ОТЛАДКА

import datetime

learning_parameters = {
    'start_learn': None,
    'actual_dataset': pd.DataFrame(),
    'time_last_processing': None
}

def debug_update():
    #global counter

    start_date = datetime.datetime(2023, 5, 25, 16, 0, 0, 0) # datetime.datetime(2023, 5, 18, 18, 25, 42, 0) #datetime.datetime.now() - datetime.timedelta(days=10)
    new_dataset = get_dataset_for_model(start_date=start_date,
                                                    audience_name=None,
                                                    normalize=False)
    new_dataset = new_dataset.reset_index()
    temp_column_name = new_dataset.filter(like='wall_temp').columns.item()
    new_dataset = new_dataset[['date_time', temp_column_name]]
    new_dataset.rename(columns={temp_column_name: 'temp'}, inplace=True)

    if learning_parameters['actual_dataset'].empty:
        actual_dataset = new_dataset.head(1)  # learning_parameters['actual_dataset']
    else:
        # Получаем последнюю запись из глобального датасета
        last_record = learning_parameters['actual_dataset'].iloc[-1]

        # Ищем индекс записи в новом датасете, следующей после последней записи в глобальном датасете
        new_records = new_dataset[new_dataset['date_time'] > last_record['date_time']]
        if new_records.empty:
            return

        next_index = new_records.index[0]

        # Выбираем запись с найденным индексом в новом датасете
        next_record = new_dataset.loc[next_index]
        # Добавляем запись в глобальный датасет
        #actual_dataset = learning_parameters['actual_dataset'].append(next_record, ignore_index=True)
        actual_dataset = pd.concat([learning_parameters['actual_dataset'], next_record.to_frame().transpose()], ignore_index=True)
        s = 3
    # if learning_parameters['actual_dataset'].shape[0] != 0:
    #     old_dataset = learning_parameters['actual_dataset']
    #
    #     # Получаем последнюю запись из глобального датасета
    #     last_record = old_dataset.iloc[-1]
    #
    #     # Ищем индекс записи в новом датасете, следующей после последней записи в глобальном датасете
    #     next_element = new_dataset[new_dataset['date_time'] > last_record['date_time']] #.index[0]
    #
    #     merged_dataset = pd.concat([old_dataset, next_element])
    #
    #
    #     # Объединение старого и нового датасетов
    #     # combined_dataset = pd.concat([old_dataset, new_dataset])
    #     #
    #     # # Удаление дубликатов и сохранение только новых записей
    #     # new_records_dataset = combined_dataset.drop_duplicates(keep=False)
    #     #
    #     # if len(new_records_dataset) > 0:
    #     #     # Добавление одной новой записи к старому датасету
    #     #     actual_dataset = pd.concat([old_dataset, new_records_dataset.head(1)])
    #     # else:
    #     #     print()
    #     #     # Присвоение первой записи из new_dataset
    #     #     # actual_dataset = new_dataset.head(1)
    # else:
    #     # Присвоение первой записи из new_dataset
    #     actual_dataset = new_dataset.head(1)
    # print(actual_dataset)
    learning_parameters['actual_dataset'] = actual_dataset
    print("Актуальный датасет: ",actual_dataset)

results = []

drift_results = []

def debug_drift_detection():
    # Детекция дрейфа
    data_stream = learning_parameters['actual_dataset']['temp']
    adwin = drift.binary.DDM()  # drift.ADWIN()
    drift_index_adwin_a = DataDriftDetector.stream_drift_detector(data_stream, adwin)
    drift_results.append(drift_index_adwin_a)
    if len(drift_index_adwin_a) > 0:
        print()
    s = 3

    # Визуализация работы метода
    # drift_values_adwin_a = data_stream[drift_index_adwin_a]
    # test.final_chart_dots(data_stream, drift_index_adwin_a, drift_values_adwin_a, abrupt_drift=True)




if __name__ == "__main__":
    #drift.binary.ddm.test_ddm()
    # Обновление данных
    #update_dataset()
    #dataset_for_model = read_dataset()
    # Получение нормализованного датасета для модели
    current_time = datetime.datetime.now() - timedelta(days=10)
    dataset_for_model = get_dataset_for_model(start_date=current_time, audience_name=None, normalize=True)
    # Денормализация датасета
    denormalize_dataset = DataPreprocessing.denormalize_dataset(normalized_dataset=dataset_for_model)
    s = 3

    # print('Получение данных')
    # result_dataset = denormalize_dataset.copy()
    # result_dataset = result_dataset.reset_index()
    # temp_column_name = result_dataset.filter(like='wall_temp').columns.item()
    # result_dataset = result_dataset[['date_time', temp_column_name]]
    # # result_dataset.rename(columns={temp_column_name: 'temp'}, inplace=True)
    #
    # print(result_dataset)
    # # Преобразование данных в формат, пригодный для передачи через AJAX
    # json_data = result_dataset.to_dict(orient='records')
    #
    # result = jsonify(json_data)
    counter = 0
    # Обновление данных
    for i in range(0, 1000):
        debug_update()
        # Запускаем обнаружение сдвига данных, если в датасете более 1 записи
        if learning_parameters['actual_dataset'].shape[0] > 1:
            debug_drift_detection()
        actual_dataset = learning_parameters['actual_dataset']
        #print(actual_dataset)

        # Если данные уже были ранее получены, то берем только новые данные
        if learning_parameters['time_last_processing'] is not None:
            actual_dataset = actual_dataset[actual_dataset['date_time'] > learning_parameters['time_last_processing']]
        if actual_dataset.empty:
            break

        print(actual_dataset['date_time'].iloc[-1])
        # Обновляем время последнего считывания
        #print(f'actual_dataset = {actual_dataset}')
        #print(f"date_time = {actual_dataset['date_time']}")
        #print(type(actual_dataset['date_time']))
        #print(type(actual_dataset))
        learning_parameters['time_last_processing'] = actual_dataset['date_time'].iloc[-1]
        # rows = learning_parameters['actual_dataset'].iloc[0:counter + 1]

        # Преобразование данных в формат, пригодный для передачи через AJAX
        json_data = actual_dataset.reset_index().to_dict(orient='records')
        results.append(json_data)

        counter += 1
        print(counter)
    s = 2