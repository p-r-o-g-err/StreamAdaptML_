"""
    Описание роутов, которые обрабатывают запросы клиента
"""
import math
import pandas as pd
from app import app
from flask import render_template, request, send_file, redirect, url_for, jsonify
from app.modules import DataHandler, DataDriftDetector, DataPreprocessing
from river import drift
import os
import threading
from threading import Thread
import datetime
from app.modules.MLModelTrainerTest import ModelGeneration


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
    window_size = settings.get('window_size')
    # Проверка наличия модели на сервере
    model_exists = app.config['LOADED_MODEL'] != None

    # Возврат страницы index.html с передачей в неё данных
    return render_template('index.html',
                           title='Главная',
                           model_exists=model_exists,
                           model_status=model_status,
                           data_shift_detection_method=data_shift_detection_method,
                           training_method=training_method,
                           training_method_with_data_shift=training_method_with_data_shift,
                           window_size=window_size)


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
    window_size = int(request.form.get(key='window-size'))
    # Обновить настройки
    DataHandler.update_settings(data_shift_detection_method, training_method, training_method_with_data_shift, window_size)
    return redirect(url_for('index'))


# endregion


# region Страница обучения

learning_parameters = {
    'first_run_status': True,  # Флаг запуска приложения (для очистки локального хранилища)
    'start_learn': None,  # Флаг запуска обучения
    'actual_dataset': pd.DataFrame(),  # Окно данных (S)
    'last_reading_time_for_streaming_data_chart': None,  # Время последнего считывания данных графиком потоковых данных
    'drift_indexes': [],  # Обнаруженные точки сдвига данных
    'drift_detector': None,  # Детектор сдвига данных
    'last_element_for_drift_detector': None,  # Последний элемент, считанный при работе метода обнаружения сдвига данных
    'last_reading_time_for_learning_model': None,  # Время последнего считывания данных для обучения модели
    'time_elapsed': datetime.timedelta(seconds=0),  # Счетчик времени обучения

    'predicted_temp': None,
    'true_temp': None,
    'mse': None,
    'r2': None,
    'last_reading_time_for_predictions_data_chart': None,
    'last_reading_time_for_training_data_chart': None
}


@app.route('/learning')
def learning():
    """
        Обработчик маршрута страницы обучения
    """
    # Инициализация конфигурации приложения
    init_config()
    # Получить данные для графика
    # actual_dataset = learning_parameters['actual_dataset']
    #if not actual_dataset.empty:
    #    print('Данные для графика\n', actual_dataset[['date_time', 'temp_audience']])
    #else:
    #    print('Данные для графика\n', actual_dataset)
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
    if learning_parameters['start_learn'] is None:
        learning_parameters['start_learn'] = datetime.datetime.now()
    else:
        learning_parameters['start_learn'] = datetime.datetime.now() - learning_parameters['time_elapsed'] # learning_parameters['start_learn']
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


background_thread = None
stop_event = None


def background_work():
    global stop_event
    # Считывание настроек обучения
    settings = DataHandler.read_settings()
    # Метод обучения при отсутствии сдвига данных
    training_method = settings.get('training_method')
    # Метод обучения при наличии сдвига данных
    training_method_with_data_shift = settings.get('training_method_with_data_shift')
    # Размер окна данных для дообучения (S)
    window_size = settings.get('window_size')

    # DataHandler.update_dataset()
    # Считывание модели
    current_model = DataHandler.read_model()
    # Инициализация модели
    obj_model = ModelGeneration(model=current_model)

    # Выполнять фоновые задачи, пока не будет получен запрос на остановку
    while not stop_event.is_set():
        # Актуализация датасета
        result_update_dataset = updateData(window_size)  # DataHandler.update_dataset(logging=False) + read_dataset
        # Если были получены новые данные
        if result_update_dataset:
            # Если достигнуто необходимое количество элементов в окне
            if len(learning_parameters['actual_dataset']) == window_size:
                # Определить есть ли сдвиг и получить точки сдвига
                is_drift = run_drift_detection()

                # Подготовка актуального датасета для модели
                actual_dataset = learning_parameters['actual_dataset'].set_index('date_time')
                dataset = DataPreprocessing.normalize_dataset(actual_dataset)

                # Если сдвиг обнаружен
                if is_drift:
                    # Обучить модель, используя метод training_method_with_data_shift
                    if training_method_with_data_shift == 'online_learning':
                        obj_model.train_online(dataset)
                    elif training_method_with_data_shift == 'mini_batch_online_learning':
                        obj_model.train_mini_batch_online(dataset)
                    elif training_method_with_data_shift == 'learning_from_scratch':
                        obj_model.train_from_scratch(dataset)
                    elif training_method_with_data_shift == 'transfer_learning':
                        obj_model.train_transfer_learning(dataset)
                    elif training_method_with_data_shift == 'autofit':
                        obj_model.train_autofit(dataset)
                    else:
                        print('Передан неверный метод обучения модели при наличии сдвига данных')
                # Иначе
                else:
                    # Обучить модель, используя метод training_method
                    if training_method == 'online_learning':
                        obj_model.train_online(dataset)
                    elif training_method == 'mini_batch_online_learning':
                        obj_model.train_mini_batch_online(dataset)
                    elif training_method == 'transfer_learning':
                        obj_model.train_transfer_learning(dataset)
                    else:
                        print('Передан неверный метод обучения модели при отсутствии сдвига данных')
                obj_model.save_model()

                # Получаем фактические и предсказанные значения для окна S
                predicted_temp = DataPreprocessing.denormalize_temp(obj_model.predicted)
                true_temp = DataPreprocessing.denormalize_dataset(dataset)['temp_audience']
                # Получаем значения метрик MSE и R2 для окна S
                mse = obj_model.mse
                r2 = obj_model.r2
                learning_parameters['predicted_temp'] = predicted_temp
                learning_parameters['true_temp'] = true_temp
                learning_parameters['mse'] = mse
                learning_parameters['r2'] = r2
        # time.sleep(3)


def updateData(window_size):
    """
    Обновляет набор данных для отладки.
    Извлекает новый набор данных на основе указанной даты начала и добавляет к текущему набору данных 1 запись.
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
    new_dataset.rename(columns={temp_column_name: 'temp_audience'}, inplace=True)

    if learning_parameters['actual_dataset'].empty:
        actual_dataset = new_dataset.head(1)
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
        actual_dataset = pd.concat([learning_parameters['actual_dataset'], next_record.to_frame().transpose()],
                                   ignore_index=False)
        # Проверяем размер actual_dataset
        if len(actual_dataset) > window_size:
            # Удаляем самую старую запись
            actual_dataset = actual_dataset.iloc[1:]

    learning_parameters['actual_dataset'] = actual_dataset
    return True


def run_drift_detection():
    """
    Выполняет обнаружение сдвига в наборе данных.
    :return: Список индексов, где обнаружен сдвиг.
    """
    if learning_parameters['actual_dataset'].empty:
        return False
    else:
        actual_dataset = learning_parameters['actual_dataset']
        last_element = learning_parameters['last_element_for_drift_detector']
        if last_element is None:
            learning_parameters['last_element_for_drift_detector'] = actual_dataset.iloc[-1]
        else:
            actual_dataset = actual_dataset[actual_dataset['date_time'] > last_element['date_time']]
            learning_parameters['last_element_for_drift_detector'] = actual_dataset.iloc[-1]

        data_stream = actual_dataset['temp_audience']

        if learning_parameters['drift_detector'] is None:
            # Чтение значения data_shift_detection_method из файла settings.json
            settings = DataHandler.read_settings()
            data_shift_detection_method = settings.get('data_shift_detection_method')

            print('Метод обнаружения сдвига:', data_shift_detection_method)
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
        else:
            drift_detector = learning_parameters['drift_detector']

        drift_detector, drift_indexes = DataDriftDetector.stream_drift_detector(data_stream, drift_detector)

        learning_parameters['drift_detector'] = drift_detector
        print(drift_indexes)

        # Определить наличие сдвига на последних данных
        is_drift = len(drift_indexes) > len(learning_parameters['drift_indexes'])
        # Сохранить точки сдвига
        if len(learning_parameters['drift_indexes']) == 0:
            learning_parameters['drift_indexes'] = drift_indexes
        else:
            for i in drift_indexes:
                learning_parameters['drift_indexes'].append(i)

        # Вернуть флаг наличия сдвига
        return is_drift


@app.route('/first_run')
def get_first_run_status():
    result = learning_parameters['first_run_status']
    learning_parameters['first_run_status'] = False
    return jsonify(first_run_status=result)


@app.route('/training_data')
def get_training_data():
    """
    Отправка информации об обучении:
        + время обучения
        + значение функции потерь
        + точность
        - изменение точности за время обучения
    :return:
    """
    start_learn = learning_parameters['start_learn']
    if start_learn is not None:
        training_time = datetime.datetime.now() - start_learn
        learning_parameters['time_elapsed'] = training_time
        total_seconds = int(training_time.total_seconds())
        total_hours = total_seconds // 3600
        remaining_minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        training_time = "{:02d}:{:02d}:{:02d}".format(total_hours, remaining_minutes, remaining_seconds)

        mse = None
        r2 = None
        loss = []
        if learning_parameters['predicted_temp'] is not None:
            if not learning_parameters['predicted_temp'].to_frame().empty:
                mse = round(learning_parameters['mse'], 3)
                if mse > 1:
                    mse = 1.000
                r2 = str(round(learning_parameters['r2'] * 100, 3)) + '%'

        # Данные графика
        if mse is not None:
            # Если время последнего считывания не задано, то задаем
            if learning_parameters['last_reading_time_for_training_data_chart'] is None:
                if learning_parameters['last_reading_time_for_predictions_data_chart'] is not None:
                    learning_parameters['last_reading_time_for_training_data_chart'] = learning_parameters['last_reading_time_for_predictions_data_chart']
                    date_time = learning_parameters['last_reading_time_for_training_data_chart']
                    loss = pd.DataFrame({'date_time': [date_time], 'loss': [mse]}).reset_index().to_dict(
                        orient='records')
                else:
                    if learning_parameters['true_temp'] is not None:
                        true_temp = learning_parameters['true_temp'].to_frame().reset_index()
                        learning_parameters['last_reading_time_for_training_data_chart'] = true_temp['date_time'].iloc[-1]
                        date_time = learning_parameters['last_reading_time_for_training_data_chart']
                        loss = pd.DataFrame({'date_time': [date_time], 'loss': [mse]}).reset_index().to_dict(
                            orient='records')
            else:
                # Если время последнего считывания не совпадает с временем последнего считывания графиком прогнозов, то задаем
                if learning_parameters['last_reading_time_for_predictions_data_chart'] is not None:
                    if learning_parameters['last_reading_time_for_training_data_chart'] != learning_parameters['last_reading_time_for_predictions_data_chart']:
                        learning_parameters['last_reading_time_for_training_data_chart'] = learning_parameters['last_reading_time_for_predictions_data_chart']
                        date_time = learning_parameters['last_reading_time_for_training_data_chart']
                        loss = pd.DataFrame({'date_time': [date_time], 'loss': [mse]}).reset_index().to_dict(orient='records')
        # print('loss:', loss)
        # Отправляем данные и индексы сдвига на клиентскую сторону
        result = jsonify(training_time=training_time, mse=mse, r2=r2, loss=loss)
        return result


@app.route('/predictions_data')
def get_predictions_data():
    true_temp = learning_parameters['true_temp']
    predicted_temp = learning_parameters['predicted_temp']
    # print('true_temp:', true_temp)
    # print('predicted_temp:', predicted_temp)
    if predicted_temp is None or true_temp is None:
        return jsonify(true_temp=[], predicted_temp=[])

    true_temp = true_temp.to_frame().reset_index()
    predicted_temp = predicted_temp.to_frame().reset_index()

    # Если данные уже были ранее получены, то берем только новые данные
    if learning_parameters['last_reading_time_for_predictions_data_chart'] is not None:
        true_temp = true_temp[
            true_temp['date_time'] > learning_parameters['last_reading_time_for_predictions_data_chart']]
        predicted_temp = predicted_temp[
            predicted_temp['index'] > learning_parameters['last_reading_time_for_predictions_data_chart']]

    if true_temp.empty or predicted_temp.empty:
        return jsonify(true_temp=[], predicted_temp=[])
    # Обновляем время последнего считывания
    learning_parameters['last_reading_time_for_predictions_data_chart'] = true_temp['date_time'].iloc[-1]

    # Преобразование данных в формат, пригодный для передачи через AJAX
    true_json = true_temp.to_dict(orient='records')
    pred_json = predicted_temp.to_dict(orient='records')

    # Отправляем данные на клиентскую сторону
    result = jsonify(true_temp=true_json, pred_temp=pred_json)
    return result


@app.route('/streaming_data')
def get_streaming_data():
    """
    Отправка информации о потоковых данных:
        + данные графика
        + количество точек сдвига
    :return:
    """

    drift_indexes = learning_parameters['drift_indexes']
    print('Индексы точек сдвига:', drift_indexes)
    actual_dataset = learning_parameters['actual_dataset']
    # Проверка, если actual_dataset пустой, вернуть пустой ответ
    if actual_dataset.empty:
        print('Пустой датасет:', actual_dataset)
        return jsonify(data=[], driftIndexes=drift_indexes)

    # Оставляем только температуру кабинета и время
    actual_dataset = actual_dataset[['date_time', 'temp_audience']]

    # Если данные уже были ранее получены, то берем только новые данные
    if learning_parameters['last_reading_time_for_streaming_data_chart'] is not None:
        actual_dataset = actual_dataset[actual_dataset['date_time'] > learning_parameters['last_reading_time_for_streaming_data_chart']]

    if actual_dataset.empty:
        return jsonify(data=[], driftIndexes=drift_indexes)

    # Обновляем время последнего считывания
    learning_parameters['last_reading_time_for_streaming_data_chart'] = actual_dataset['date_time'].iloc[-1]

    # Преобразование данных в формат, пригодный для передачи через AJAX
    json_data = actual_dataset.reset_index().to_dict(orient='records')
    print(f'Данные для отрисовки: {json_data}')
    # print('last_reading_time_for_streaming_data_chart:', learning_parameters['last_reading_time_for_streaming_data_chart'])

    # Преобразование данных перед отправкой на клиентскую сторону
    for item in json_data:
        if math.isnan(item['temp_audience']):
            item['temp_audience'] = None  # Заменить NaN на None

    # Отправляем данные и индексы сдвига на клиентскую сторону
    result = jsonify(data=json_data, driftIndexes=drift_indexes)

    return result
# endregion
