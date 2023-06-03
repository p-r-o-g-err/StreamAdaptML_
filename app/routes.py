'''
    Описание роутов, которые обрабатывают запросы клиента
'''
import json
import math

import pandas as pd
from app import app
from flask import render_template, request, send_file, redirect, url_for, jsonify
from app.modules import DataHandler, DataDriftDetector
from river import drift
import os
import threading


def init_config():
    # Проверка наличия загруженной модели
    models_files = os.listdir(app.config['MODELS_FOLDER'])
    # Если в папке с моделью есть модель
    if len(models_files) > 0:
        # Формируем имя файла
        filename = models_files[0]
        # Имя файла сохраняется в конфигурации приложения
        app.config['LOADED_MODEL'] = filename


# region Страница настроек
@app.route('/')
def index():
    """
        Обработчик маршрута главной страницы (страницы настроек)
    """
    # Инициализация конфигураций приложения
    init_config()
    # Формирование статуса загрузки модели
    model_status = f"Загружена модель \"{app.config['LOADED_MODEL']}\"" if app.config[
                                                                               'LOADED_MODEL'] is not None else "Модель не загружена"

    # Получение настроек из settings.json
    settings = DataHandler.read_settings()
    data_shift_detection_method = settings.get('data_shift_detection_method')
    training_method = settings.get('training_method')
    training_method_with_data_shift = settings.get('training_method_with_data_shift')

    # Проверка наличия модели на сервере
    model_exists = app.config['LOADED_MODEL'] != None

    # Возврат страницы index.html с передачей в неё данных
    return render_template('index.html',
                           title='Главная',
                           model_exists=model_exists,
                           model_status=model_status,
                           data_shift_detection_method=data_shift_detection_method,
                           training_method=training_method,
                           training_method_with_data_shift=training_method_with_data_shift)


@app.route('/download_model', methods=['POST'])
def download_model():
    '''
    Загрузка модели на сервер
    '''
    file = request.files['file']
    DataHandler.save_model(file)
    return redirect(url_for('index'))


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    Скачивание модели
    """
    return DataHandler.upload_model()


@app.route('/delete_model', methods=['POST'])
def delete_model():
    """
    Удаление загруженной модели
    """
    for f in [f for f in os.listdir(app.config['MODELS_FOLDER'])]:
        os.remove(os.path.join(app.config['MODELS_FOLDER'], f))
    app.config['LOADED_MODEL'] = None
    return 'Модель удалена'


@app.route('/save_settings', methods=['POST'])
def save_settings():
    """
    Сохранение настроек обучения
    :return:
    """
    # Получить настройки
    data_shift_detection_method = request.form.get(key='data-shift-detection-select')
    training_method = request.form.get(key='training-method-select')
    training_method_with_data_shift = request.form.get(key='training-method-with-data-shift-select')
    # Обновить настройки
    DataHandler.update_settings(data_shift_detection_method, training_method, training_method_with_data_shift)
    return redirect(url_for('index'))


# endregion


# region Страница обучения

learning_parameters = {
    'start_learn': None,
    'actual_dataset': pd.DataFrame(),
    'last_reading_time_for_streaming_data_chart': None,
    'drift_indexes': []
}


@app.route('/chart_streaming_data')
def get_chart_data():
    if learning_parameters['actual_dataset'].empty:
        return jsonify(data=[], driftIndexes=[])
    json_data = learning_parameters['actual_dataset'].reset_index().to_dict(orient='records')
    print(f'Данные для отрисовки: {json_data}')
    # Преобразование данных перед отправкой на клиентскую сторону
    for item in json_data:
        if math.isnan(item['temp']):
            # Заменить NaN на None
            item['temp'] = None

    drift_indexes = test_drift_detection()
    print('Индексы точек сдвига:', drift_indexes)

    # Обновляем время последнего считывания
    learning_parameters['last_reading_time_for_streaming_data_chart'] = learning_parameters['actual_dataset']['date_time'].iloc[-1]
    print('last_reading_time_for_streaming_data_chart:', learning_parameters['last_reading_time_for_streaming_data_chart'])

    result = jsonify(data=json_data, driftIndexes=drift_indexes)
    # Возвращаем данные графика в формате JSON
    return result  # jsonify(result)


@app.route('/learning')
def learning():
    """
        Обработчик маршрута страницы обучения
    """
    # Инициализация конфигурации приложения
    init_config()
    # Получить данные для графика
    print('Данные для графика\n', learning_parameters['actual_dataset'])
    # chart_data = learning_parameters['actual_dataset']
    # chart_data = learning_parameters['actual_dataset'].to_dict(orient='records')
    # chart_data = learning_parameters['actual_dataset'].to_json()
    number_drift_points = len(learning_parameters['drift_indexes'])
    return render_template('learning.html', title='Обучение', number_drift_points=number_drift_points)


@app.route('/start_learning', methods=['POST'])
def start_learning():
    """
        Запускает обучение
        :return: результат запуска в виде строки
    """
    global background_thread, stop_event
    if background_thread and background_thread.is_alive():
        print('Фоновая работа уже запущена')
        return 'Фоновая работа уже запущена'

    app.config['IS_LEARNING_STARTED'] = True
    print('Фоновая работа запущена')
    stop_event = threading.Event()
    background_thread = Thread(target=background_work)
    background_thread.start()
    learning_parameters['start_learn'] = datetime.datetime.now()
    return 'Фоновая работа запущена'
    # return redirect(url_for('learning'))


@app.route('/stop_learning', methods=['POST'])
def stop_learning():
    """
        Останавливает обучение, если оно запущено
        :return: результат остановки в виде строки
    """
    global background_thread, stop_event
    if background_thread and background_thread.is_alive():
        app.config['IS_LEARNING_STARTED'] = False
        stop_event.set()
        background_thread.join()
        print('Фоновая работа остановлена')
        return 'Фоновая работа остановлена'
    else:
        print('Нет запущенной фоновой работы')
        return 'Нет запущенной фоновой работы'
    # return redirect(url_for('learning'))


@app.route('/check_status', methods=['GET'])
def check_status():
    """
        Вспомогательная функция для javascript
        Проверяет, запущено ли обучение
        :return: результат проверки в виде текста
    """
    if app.config['IS_LEARNING_STARTED']:
        return 'Фоновая работа запущена'
    else:
        return 'Нет запущенной фоновой работы'


@app.route('/check_model', methods=['GET'])
def check_model():
    """
        Вспомогательная функция для javascript
        Проверяет, загружена ли модель
        :return: результат проверки в виде текста
    """
    return 'Модель загружена' if app.config['LOADED_MODEL'] != None else 'Модель не загружена'


new_data = {}

from threading import Thread
import time

background_thread = None
stop_event = None

def background_work():
    global stop_event
    # Выполнять фоновые задачи, пока не будет получен запрос на остановку
    while not stop_event.is_set():
        # Для отладки
        print(get_current_time())
        # Актуализация датасета
        result_update_dataset = test_update()  # DataHandler.update_dataset(logging=False) + read_dataset
        # Если были получены новые данные
        if result_update_dataset:
            # Обновить точки сдвига
            learning_parameters['drift_indexes'] = test_drift_detection()
            # Для примера берем каждые 3 точки
            # learning_parameters['drift_indexes'] = list(learning_parameters['actual_dataset'].index[::3])

            # Получить текущий датасет
            actual_dataset = learning_parameters['actual_dataset']

            # Получение нормализованного датасета для модели
            start_learn = learning_parameters['start_learn']
            # dataset_for_model = DataHandler.get_dataset_for_model(start_date=start_learn,
            #                                                      audience_name=None,
            #                                                      normalize=True)
            # Обновляем actual_dataset
            # learning_parameters['actual_dataset'] = dataset_for_model

            s = 3
            # Обучение модели
            # learn_model(settings)
            # Обновление параметров
            # update_data()
        #time.sleep(3)


# Функция, работающая на фоне для примера
import datetime


def get_current_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def test_update():
    """
    Обновляет набор данных для отладки.
    Она извлекает новый набор данных на основе указанной даты начала,
    и добавляет к текущему набору данных 1 запись.
    :return: None
    """
    # Определяем начальное время
    start_date = datetime.datetime.now() - datetime.timedelta(days=10)
    # Считываем датасет
    new_dataset = DataHandler.get_dataset_for_model(start_date=start_date,
                                                    audience_name=None,
                                                    normalize=False)
    new_dataset = new_dataset.reset_index()
    # Определяем столбец температуры
    temp_column_name = new_dataset.filter(like='wall_temp').columns.item()
    # Переименовываем столбец
    new_dataset = new_dataset[['date_time', temp_column_name]]
    new_dataset.rename(columns={temp_column_name: 'temp'}, inplace=True)

    # print('До debug_update(): ', learning_parameters['actual_dataset'])

    if learning_parameters['actual_dataset'].empty:
        actual_dataset = new_dataset.head(1)  # learning_parameters['actual_dataset']
    else:

        # Получаем последнюю запись из глобального датасета
        last_record = learning_parameters['actual_dataset'].iloc[-1]

        new_records = new_dataset[new_dataset['date_time'] > last_record['date_time']]
        if new_records.empty:
            return False

        next_index = new_records.index[0]
        # Выбираем запись с найденным индексом в новом датасете
        next_record = new_dataset.loc[next_index]
        # Добавляем запись в глобальный датасет
        # actual_dataset = learning_parameters['actual_dataset'].append(next_record, ignore_index=True)
        actual_dataset = pd.concat([learning_parameters['actual_dataset'], next_record.to_frame().transpose()],
                                   ignore_index=True)

    learning_parameters['actual_dataset'] = actual_dataset
    return True


def test_drift_detection():
    """
    Выполняет обнаружение сдвига в наборе данных.
    :return: Список индексов, где обнаружен сдвиг.
    """
    if learning_parameters['actual_dataset'].empty:
        return []
    else:
        data_stream = learning_parameters['actual_dataset']['temp']
        #adwin = drift.ADWIN()

        # Чтение значения data_shift_detection_method из файла settings.json
        settings = DataHandler.read_settings()
        data_shift_detection_method = settings.get('data_shift_detection_method')

        # Выбор метода обнаружения сдвига на основе значения data_shift_detection_method
        if data_shift_detection_method == 'ADWIN':
            drift_detector = drift.ADWIN()
        elif data_shift_detection_method == 'DDM':
            drift_detector = drift.binary.DDM()
        elif data_shift_detection_method == 'EDDM':
            drift_detector = drift.binary.EDDM()
        else:
            # По умолчанию используется ADWIN, если значение не указано или некорректно
            drift_detector = drift.ADWIN()
        print('Метод обнаружения сдвига:', data_shift_detection_method)
        drift_indexes = DataDriftDetector.stream_drift_detector(data_stream, drift_detector)
        print(drift_indexes)
        return drift_indexes


@app.route('/data')
def get_data():
    """
    Отправка данных графику потоковых данных
    :return:
    """

    # Обновление данных
    #test_update()
    # Получить индексы строк со сдвигом
    #drift_indexes = list(learning_parameters['actual_dataset'].index[::3])
    #drift_indexes = test_drift_detection()

    drift_indexes = learning_parameters['drift_indexes']
    print('Индексы точек сдвига:', drift_indexes)
    actual_dataset = learning_parameters['actual_dataset']

    # Если данные уже были ранее получены, то берем только новые данные
    if learning_parameters['last_reading_time_for_streaming_data_chart'] is not None:
        actual_dataset = actual_dataset[actual_dataset['date_time'] > learning_parameters['last_reading_time_for_streaming_data_chart']]
    # Проверка, если actual_dataset пустой, вернуть пустой ответ
    if actual_dataset.empty:
        print('Пустой датасет:', actual_dataset)
        return jsonify(data=[], driftIndexes=[])

    # Обновляем время последнего считывания
    learning_parameters['last_reading_time_for_streaming_data_chart'] = actual_dataset['date_time'].iloc[-1]
    print('last_reading_time_for_streaming_data_chart:', learning_parameters['last_reading_time_for_streaming_data_chart'])

    # Преобразование данных в формат, пригодный для передачи через AJAX
    json_data = actual_dataset.reset_index().to_dict(orient='records')
    print(f'Данные для отрисовки: {json_data}')

    # Преобразование данных перед отправкой на клиентскую сторону
    for item in json_data:
        if math.isnan(item['temp']):
            item['temp'] = None  # Заменить NaN на None

    # Отправляем данные и индексы сдвига на клиентскую сторону
    result = jsonify(data=json_data, driftIndexes=drift_indexes)

    return result

# endregion
