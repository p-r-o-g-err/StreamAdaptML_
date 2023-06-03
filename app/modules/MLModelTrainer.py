"""
    Модуль обучения модели машинного обучения.
    Функционал:
        1) Построить модель предсказания температуры в выбранном кабинете (обучить модель на исторических данных)
        2) Обучить модель на потоковых данных с выбором метода обнаружения сдвига данных,
            метода обучения при наличии и отсутствии сдвига данных
"""

import DataPreprocessing
import Visualization
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from datetime import datetime
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json


# Инициализация модели
def create_model(x_train, drop_out=0.2, activation='linear', optimizer='rmsprop'):
    model = Sequential()
    # 1 слой: принимает 3D тензор, возвращает всю выходную последовательность (many-to-many) с размерностью выходного пространства - 50
    model.add(LSTM(input_shape=[x_train.shape[-1], 1], units=50, return_sequences=True))
    # 2 слой: избегаем переобучения с помощью Dropout слоя
    model.add(Dropout(drop_out))
    # 3 слой: размерность определяется на основе предыдущего слоя, возвращает последний вывод в выходной последовательности (many-to-one)
    model.add(LSTM(units=100, return_sequences=False))
    # 4 слой: избегаем переобучения с помощью Dropout слоя
    model.add(Dropout(drop_out))
    # 5 слой: плотносвязанный слой, возвращающий прогнозы с размерностью выходного пространства - 1
    model.add(Dense(units=1, activation=activation))
    # model.add(Activation(activation))
    start = datetime.now()
    model.compile(loss='mse', optimizer=optimizer)
    print('Время компиляции :', datetime.now() - start)
    return model


# Обучение модели (офлайн)
def fit_model(model, x_train, y_train, batch_size=32, epochs=10):
    start = datetime.now()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    print('Время тренировки : {}'.format(datetime.now() - start))


# Прогнозирование
def model_predict(model, x_test):
    return model.predict(x_test)


# Ошибка прогноза RMSE
def print_error(testY, test_predict):
    test_rmse = mean_squared_error(testY, test_predict, squared=False)
    print('\tТест RMSE: %.3f' % (test_rmse))


# Сохранение модели
def save_model(model, model_name):
    if len(model_name.split('.')) == 1: model_name = model_name + '.keras'
    model.save(model_name)
    print(f"Модель \'{model_name}\' успешно сохранена")


# Загрузка модели
def load_model(model_name, loss='mse', optimizer='adam'):
    if len(model_name.split('.')) == 1: model_name = model_name + '.keras'
    loaded_model = keras.models.load_model(model_name)
    print(f"Модель \'{model_name}\' успешно загружена")
    # Компиляция модели 
    loaded_model.compile(loss=loss, optimizer=optimizer)
    return loaded_model


# Класс модели
class ModelGeneration(object):
    input_dataset = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    train_index = None
    test_index = None
    result_model = None
    predicted = None
    target_column = None

    def __init__(self, dataset, model=None):
        self.input_dataset = dataset
        self.target_column = list(dataset.filter(regex='wall_temp'))[0]
        if model is not None: self.result_model = model

    def get_train_test(self, test_size):
        self.x_train, self.y_train, self.x_test, self.y_test, self.train_index, self.test_index = DataPreprocessing.get_train_test(
            self.input_dataset, self.target_column, test_size)

    def fit_model(self, epochs_num):
        if self.result_model is None:
            print('Не инициализирована модель (Укажите модель)')
        elif (self.x_train is None) or (self.y_train is None):
            print('Отсутствует тренировочная выборка (Запустите get_train_test())')
        else:
            fit_model(self.result_model, self.x_train, self.y_train, epochs=epochs_num)

    def visual_learning(self):
        if self.result_model is None:
            print('Не инициализирована модель.')
        else:
            Visualization.plot_model(self.result_model)

    def model_predict(self):
        if self.x_test is None:
            print('Отсутствует тестовая выборка (Запустите get_train_test())')
        else:
            self.predicted = self.result_model.predict(self.x_test)

    def print_error(self):
        if self.y_test is None:
            print('Отсутствует тестовая выборка (Запустите get_train_test())')
        elif self.predicted is None:
            print('Предсказания отсутствуют (Запустите model_predict())')
        else:
            print_error(self.y_test, self.predicted.flatten())

    def plot_predicted(self, renderer='browser', plot_bat=False, plot_weather=False):
        if self.y_test is None:
            print('Отсутствует тестовая выборка (Запустите get_train_test())')
        elif self.predicted is None:
            print('Предсказания отсутствуют (Запустите model_predict())')
        else:
            temp_bat_col = None
            temp_outside_col = None
            if plot_bat:
                temp_bat_col = self.input_dataset[self.input_dataset.filter(regex='bat_temp').columns[0]]
            if plot_weather:
                temp_outside_col = self.input_dataset['temp']
            Visualization.plot_temps_series_predicted(self.test_index, self.y_test, self.predicted.flatten(), renderer,
                                                      temp_bat_col, temp_outside_col)

    def save_model(self, name):
        save_model(self.result_model, name)

    # Функция, запускающая все остальные
    def create_prediction_model(self, test_size=0.333, drop_out=0.2, activation='linear', optimizer='rmsprop',
                                epochs_num=20, find_anomalies=True, visual_mode='png', plot_bat=False,
                                plot_weather=False):
        if self.x_train is None:
            print("\tРазбиение датасета")
            self.get_train_test(test_size)
        print(
            f"\tx_train: {self.x_train.shape}\n\ty_train: {self.y_train.shape}\n\tx_test: {self.x_test.shape}\n\ty_test: {self.y_test.shape}")
        if self.result_model is None:
            print("Создание модели")
            self.result_model = create_model(self.x_train, drop_out, activation, optimizer=optimizer)
        print("Тренировка модели")
        self.fit_model(epochs_num)
        print("Отображение результатов")
        self.visual_learning()
        print("Прогнозирование значений")
        self.model_predict()
        print("Прогнозирование завершено:")
        self.print_error()
        if visual_mode not in ['none', 'png', 'browser']:
            print('Передан неизвестный режим отображения. Ожидалось одно из следующих значений: none, png, browser')
        else:
            self.plot_predicted(visual_mode, plot_bat=plot_bat, plot_weather=plot_weather)
            if find_anomalies:
                print("Обнаружение аномалий")
                self.find_anomalies('high_diff', visual_mode)


import matplotlib.pyplot as plt


def fit_on_batch(model, x_train, y_train):
    model.train_on_batch(x_train, y_train)


# def stream_learn(loaded_model,dataset):
#     adwin = drift.ADWIN()
#     relearn = False
#     print('Онлайн обучение модели')
#     time.sleep(1)
#     #loaded_model.compile(loss = 'mse', optimizer='adam')
#     bar = IncrementalBar('\t\tProgress', max = len(dataset))
#     for index, row in dataset.iterrows():
#         target_column = list(dataset.filter(regex='wall_temp'))[0]
#         Y = row[target_column]
#         X = np.array([list(row.drop(target_column))])
#         prediction = loaded_model.predict_on_batch(X)[0][0]
#         fit_on_batch(loaded_model, X, np.array([Y]))
#         #print(f'Y {Y} - prediction {prediction}')
#         in_drift, _ = adwin.update(Y)
#         if in_drift:
#             relearn = True
#             adwin.reset()
#         bar.next()
#     bar.finish()
#     time.sleep(1)
#     return loaded_model,relearn

def batch_stream_learn(loaded_model, dataset, batch_size=5, epochs=10, visual=True):
    print('Пакетное потоковое обучение модели')
    # loaded_model.compile(loss='mse', optimizer='adam')
    x_train, y_train = [], []
    # Взять последние 100 записей (для тестирования)
    dataset = dataset.tail(100)
    counter = 1
    loss = []
    target_column = list(dataset.filter(regex='wall_temp'))[0]
    for index, row in dataset.iterrows():
        Y = row[target_column]
        X = list(row.drop(target_column))
        x_train.append(X)
        y_train.append(Y)
        if len(x_train) == batch_size:
            print(f'Пакет #{counter}')
            loaded_model.result_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
            loss.append(loaded_model.result_model.history.history['loss'])
            x_train, y_train = [], []
            counter += 1
            # plot_temps_series_predicted

    # Визуализация (позже вынести в модуль Visualization.py)
    if visual == True:
        # Вывод потерь со всех эпох всех пакетов
        loss_all = sum(loss, [])
        plt.figure(figsize=(10, 5))
        plt.plot(loss_all, label='Потери')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.grid()
        plt.show()
        # Вывод средних потерь со всех пакетов
        loss_len = [sum(l) / len(l) for l in loss]
        plt.figure(figsize=(10, 5))
        plt.plot(loss_len, label='Потери')
        plt.xlabel('Пакеты')
        plt.ylabel('Потери')
        plt.grid()
        plt.show()

    return loaded_model, loss