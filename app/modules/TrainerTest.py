"""
    Модуль, написанный для выполнения исследовательской части работы.
"""
import os
import keras
from river import drift
from MLModelTrainer import *
import DataHandler
from app import app
from app.modules import DataDriftDetector

learning_parameters = {
    'start_learn': None,  # Флаг запуска обучения
    'actual_dataset': pd.DataFrame(),  # Датасет, полученный с начала обучения (D)
    'last_reading_time_for_streaming_data_chart': None,  # Время последнего считывания данных графиком потоковых данных
    'drift_indexes': [],  # Обнаруженные точки сдвига данных
    'drift_detector': None,  # Детектор сдвига данных
    'last_element_for_drift_detector': None,  # Последний элемент, считанный при работе метода обнаружения сдвига данных
    'last_reading_time_for_learning_model': None,  # Время последнего считывания данных для обучения модели
}


def updateData(window_size):
    """
    Обновляет набор данных для отладки.
    Извлекает новый набор данных на основе указанной даты начала и добавляет к текущему набору данных 1 запись.
    :return: None
    """
    # Определяем начальное время
    start_date = datetime.datetime.strptime("2023-04-30 23:50:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime("2023-05-02 00:00:00", "%Y-%m-%d %H:%M:%S")
    # start_date = datetime.datetime.now() - datetime.timedelta(days=10)
    # Считываем датасет
    new_dataset = DataHandler.get_dataset_for_model(start_date=start_date, end_date=end_date,
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
    :return: True - если сдвиг обнаружен, иначе False.
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


def comparison_models():
    """
    Сравнивает предварительно обученную и дообученную модели.
    """
    models_files = os.listdir(app.config['MODELS_FOLDER'])
    filename0 = os.path.join(app.config['MODELS_FOLDER'], models_files[0])
    filename1 = os.path.join(app.config['MODELS_FOLDER'], models_files[1])
    loaded_model0 = keras.models.load_model(filename0)
    loaded_model1 = keras.models.load_model(filename1)
    # Инициализация моделей
    obj_model0 = ModelGeneration(model=loaded_model0)
    obj_model1 = ModelGeneration(model=loaded_model1)

    # Получение датасета
    datasets = []

    start_date = datetime.datetime.strptime("2023-06-06 08:10:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime("2023-06-07 08:10:00", "%Y-%m-%d %H:%M:%S")
    datasets.append(DataHandler.get_dataset_for_model(audience_name=None,
                                                    normalize=False))
    datasets.append(DataHandler.get_dataset_for_model(start_date=start_date,
                                                      end_date=end_date,
                                                      audience_name=None,
                                                      normalize=False))
    for new_dataset in datasets:
        new_dataset = new_dataset.reset_index()
        # Определяем столбец температуры
        temp_column_name = new_dataset.filter(like='wall_temp').columns.item()
        # Переименовываем столбец
        new_dataset.rename(columns={temp_column_name: 'temp_audience'}, inplace=True)
        new_dataset = new_dataset.set_index('date_time')
        new_dataset = DataPreprocessing.normalize_dataset(new_dataset)

        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(new_dataset, 'temp_audience', mode='train_online')

        # Прогнозирование значений
        predicted_test = obj_model0.current_model.predict(x_test)
        # Вычисление точности (MSE, R2)
        obj_model0.compute_mse(y_test, predicted_test.flatten())
        obj_model0.compute_r_squared(y_test, predicted_test.flatten())

        predicted_test = obj_model1.current_model.predict(x_test)
        obj_model1.compute_mse(y_test, predicted_test.flatten())
        obj_model1.compute_r_squared(y_test, predicted_test.flatten())

        print('Модель 0: mse =', obj_model0.mse, ' r2 =', obj_model0.r2)
        print('Модель 1: mse =', obj_model1.mse, ' r2 =', obj_model1.r2)


def create_new_model():
    """
    Инициализирует и обучает модель на исторических данных, после чего сохраняет модель.
    """
    # Считывание модели
    end_date = datetime.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    new_dataset = DataHandler.get_dataset_for_model(end_date=end_date, audience_name=None, normalize=False)
    # Переименовываем столбец
    temp_column_name = new_dataset.filter(like='wall_temp').columns.item()
    new_dataset.rename(columns={temp_column_name: 'temp_audience'}, inplace=True)

    new_dataset = DataPreprocessing.normalize_dataset(new_dataset)

    # Инициализация модели
    obj_model = ModelGeneration()
    obj_model.create_model(new_dataset)

    obj_model.train_from_scratch(new_dataset)
    obj_model.save_model()


def check_original_model():
    """
    Проверяет качество предварительно обученной модели
    """
    start_date = datetime.datetime.strptime("2023-05-02 23:50:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime("2023-05-04 00:00:00", "%Y-%m-%d %H:%M:%S")
    new_dataset = DataHandler.get_dataset_for_model(start_date=start_date, end_date=end_date, audience_name=None, normalize=False)

    new_dataset = new_dataset.reset_index()
    # Определяем столбец температуры
    temp_column_name = new_dataset.filter(like='wall_temp').columns.item()
    # Переименовываем столбец
    new_dataset.rename(columns={temp_column_name: 'temp_audience'}, inplace=True)
    new_dataset = new_dataset.set_index('date_time')
    new_dataset = DataPreprocessing.normalize_dataset(new_dataset)

    # Инициализация модели
    current_model = DataHandler.read_model()
    obj_model = ModelGeneration(model=current_model)

    x_train, y_train, x_test, y_test, index_train, index_test = \
        get_train_test(new_dataset, obj_model.target_column, mode='train_online')

    # Прогнозирование значений
    predicted_test = obj_model.current_model.predict(x_test)

    # Вычисление точности (MSE, R2)
    obj_model.compute_mse(y_test, predicted_test.flatten())
    obj_model.compute_r_squared(y_test, predicted_test.flatten())
    print()


def run_learning():
    """
    Имитирует фоновую работу процесса дообучения модели (для отладки).
    """
    # Считывание настроек обучения
    settings = DataHandler.read_settings()
    # Метод обучения при отсутствии сдвига данных
    training_method = settings.get('training_method')
    # Метод обучения при наличии сдвига данных
    training_method_with_data_shift = settings.get('training_method_with_data_shift')
    # Размер окна данных для дообучения (S)
    window_size = settings.get('window_size')

    # Считывание модели
    current_model = DataHandler.read_model()
    # Инициализация модели
    obj_model = ModelGeneration(model=current_model)
    while True:
        # Актуализация датасета
        result_update_dataset = updateData(window_size)  # DataHandler.update_dataset(logging=False) + read_dataset
        # Если были получены новые данные
        if result_update_dataset:
            # Если достигнуто необходимое количество элементов в окне
            if len(learning_parameters['actual_dataset']) == window_size:
                # Определить есть ли сдвиг
                is_drift = run_drift_detection()

                # Подготовка актуального датасета для модели
                actual_dataset = learning_parameters['actual_dataset'].set_index('date_time')
                dataset = DataPreprocessing.normalize_dataset(actual_dataset)

                # Если сдвиг обнаружен
                if is_drift:
                    # Обучить модель, используя метод training_method
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
                    # Обучить модель, используя метод training_method_with_data_shift
                    if training_method == 'online_learning':
                        obj_model.train_online(dataset)
                    elif training_method == 'mini_batch_online_learning':
                        obj_model.train_mini_batch_online(dataset)
                    elif training_method == 'transfer_learning':
                        obj_model.train_transfer_learning(dataset)
                    else:
                        print('Передан неверный метод обучения модели при отсутствии сдвига данных')

                predicted_temp = DataPreprocessing.denormalize_temp(obj_model.predicted)
                true_temp = DataPreprocessing.denormalize_dataset(dataset)['temp_audience']
                # Получаем значения метрик MSE и R2 для окна S
                mse = obj_model.mse
                r2 = obj_model.r2


def train_one_model(training_method, training_method_with_data_shift):
    """
    Обучает модель с использованием выбранных методов обучения при наличии и отсутствии сдвига данных.
    :param training_method: метод обучения при отсутствии сдвига.
    :param training_method_with_data_shift: метод обучения при наличии сдвига.
    """
    # Считывание настроек обучения
    settings = DataHandler.read_settings()
    # Размер окна данных для дообучения (S)
    window_size = settings.get('window_size')

    # Считывание модели
    current_model = DataHandler.read_model()
    # Инициализация модели
    obj_model = ModelGeneration(model=current_model)
    start_time_learn = datetime.datetime.now()
    while True:
        # Актуализация датасета
        result_update_dataset = updateData(window_size)

        # Если были получены новые данные
        if result_update_dataset:
            # Если достигнуто необходимое количество элементов в окне
            if len(learning_parameters['actual_dataset']) == window_size:
                # Определить есть ли сдвиг
                is_drift = run_drift_detection()

                # Подготовка актуального датасета для модели
                actual_dataset = learning_parameters['actual_dataset'].set_index('date_time')
                dataset = DataPreprocessing.normalize_dataset(actual_dataset)

                # Если сдвиг обнаружен
                if is_drift:
                    # Обучить модель, используя метод training_method
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
                    # Обучить модель, используя метод training_method_with_data_shift
                    if training_method == 'online_learning':
                        obj_model.train_online(dataset)
                    elif training_method == 'mini_batch_online_learning':
                        obj_model.train_mini_batch_online(dataset)
                    elif training_method == 'transfer_learning':
                        obj_model.train_transfer_learning(dataset)
                    else:
                        print('Передан неверный метод обучения модели при отсутствии сдвига данных')
        else:
            end_time_learn = datetime.datetime.now() - start_time_learn
            start_date = datetime.datetime.strptime("2023-05-02 23:50:00", "%Y-%m-%d %H:%M:%S")
            end_date = datetime.datetime.strptime("2023-05-04 00:00:00", "%Y-%m-%d %H:%M:%S")
            new_dataset = DataHandler.get_dataset_for_model(start_date=start_date,
                                                            end_date=end_date, audience_name=None,
                                                            normalize=False)
            new_dataset = new_dataset.reset_index()
            # Определяем столбец температуры
            temp_column_name = new_dataset.filter(like='wall_temp').columns.item()
            # Переименовываем столбец
            new_dataset.rename(columns={temp_column_name: 'temp_audience'}, inplace=True)
            new_dataset = new_dataset.set_index('date_time')
            new_dataset = DataPreprocessing.normalize_dataset(new_dataset)

            x_train, y_train, x_test, y_test, index_train, index_test = \
                get_train_test(new_dataset, obj_model.target_column, mode='train_online')

            # Прогнозирование значений
            predicted_test = obj_model.current_model.predict(x_test)
            # Вычисление точности (MSE, R2)
            obj_model.compute_mse(y_test, predicted_test.flatten())
            obj_model.compute_r_squared(y_test, predicted_test.flatten())

            result_str = f'Метод при наличии сдвига: {training_method} | ' \
                         f'Метод при отсутствии сдвига: {training_method_with_data_shift} '
            result_mse = 'MSE: %.3f' % obj_model.mse
            result_r2 = 'R2: %.3f' % obj_model.r2
            result_str = result_str + ' | ' + result_mse + ' | ' + result_r2
            result_str = result_str + ' | Время обучения : {}'.format(end_time_learn)
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.txt')
            with open(filename, "a", encoding='utf-8') as file:
                file.write(result_str + "\n")
            break


def reset_parameters():
    """
    Сбрасывает параметры обучения, чтобы при использовании следующего метода обучения модель обучалась на тех же данных.
    """
    global learning_parameters
    learning_parameters = {
        'start_learn': None,  # Флаг запуска обучения
        'actual_dataset': pd.DataFrame(),  # Датасет, полученный с начала обучения (D)
        'last_reading_time_for_streaming_data_chart': None,
        # Время последнего считывания данных графиком потоковых данных
        'drift_indexes': [],  # Обнаруженные точки сдвига данных
        'drift_detector': None,  # Детектор сдвига данных
        'last_element_for_drift_detector': None, # Последний элемент, считанный при работе метода обнаружения сдвига данных
        'last_reading_time_for_learning_model': None,  # Время последнего считывания данных для обучения модели
    }


if __name__ == "__main__":
    # Создание и обучение модели на исторических данных
    create_new_model()
    # Запуск дообучения модели с использованием разных методов и с сохранением результатов в файл
    training_methods = ['online_learning', 'mini_batch_online_learning', 'transfer_learning']
    training_methods_with_data_shift = ['online_learning', 'mini_batch_online_learning', 'transfer_learning',
                                       'learning_from_scratch', 'autofit']
    for m1 in training_methods:
        for m2 in training_methods_with_data_shift:
            train_one_model(training_method=m1, training_method_with_data_shift=m2)
            reset_parameters()
    # Запуск проверки качества исходной модели
    check_original_model()
    # Сравнение предобученной и дообученой моделей
    comparison_models()
