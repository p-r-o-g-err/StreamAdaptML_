from MLModelTrainerTest import *
import DataHandler


learning_parameters = {
    'start_learn': None,  # Флаг запуска обучения
    'actual_dataset': pd.DataFrame(),  # Датасет, полученный с начала обучения (D)
    'last_reading_time_for_streaming_data_chart': None,  # Время последнего считывания данных графиком потоковых данных
    'drift_indexes': [],  # Обнаруженные точки сдвига данных

    'last_reading_time_for_learning_model': None,  # Время последнего считывания данных для обучения модели
    'data_window_size': 10  # Размер окна данных (S)
}


def updateData():
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
    # new_dataset = new_dataset[['date_time', temp_column_name]]
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
                                   ignore_index=True)

    learning_parameters['actual_dataset'] = actual_dataset
    return True


if __name__ == "__main__":
    # Считывание настроек обучения
    settings = DataHandler.read_settings()
    # Метод обучения при отсутствии сдвига данных
    training_method = settings.get('training_method')
    # Метод обучения при наличии сдвига данных
    training_method_with_data_shift = settings.get('training_method_with_data_shift')


    # DataHandler.update_dataset()
    # Считывание модели
    current_model = DataHandler.read_model()
    # Инициализация модели
    obj_model = ModelGeneration(model=current_model)
    while True:
        # Актуализация датасета
        result_update_dataset = updateData()  # DataHandler.update_dataset(logging=False) + read_dataset
        # Если были получены новые данные
        if result_update_dataset:
            # Подготовка актуального датасета для модели
            actual_dataset = learning_parameters['actual_dataset'].set_index('date_time')
            dataset = DataPreprocessing.normalize_dataset(actual_dataset)
            obj_model.set_dataset(dataset)
            # start_date
            if len(actual_dataset) >= learning_parameters["data_window_size"]:
                #obj_model.create_model()
                #obj_model.save_model()
                obj_model.train_from_scratch()

            # obj_model.save_model()
            s = 3

            # if training_method == 'online_learning':
            #     obj_model.train_online()
            # elif training_method == 'mini_batch_online_learning':
            #     obj_model.train_mini_batch_online()
            # elif training_method == 'transfer_learning':
            #     obj_model.train_transfer_learning()
            # else:
            #     print('Передан неверный метод обучения модели при отсутствии сдвига данных')
            #
            # if training_method_with_data_shift == 'online_learning':
            #     obj_model.train_online()
            # elif training_method_with_data_shift == 'mini_batch_online_learning':
            #     obj_model.train_mini_batch_online()
            # elif training_method_with_data_shift == 'transfer_learning':
            #     obj_model.train_transfer_learning()
            # elif training_method_with_data_shift == 'learning_from_scratch':
            #     obj_model.train_from_scratch()
            # elif training_method_with_data_shift == 'autofit':
            #     obj_model.train_autofit()
            # else:
            #     print('Передан неверный метод обучения модели при наличии сдвига данных')
