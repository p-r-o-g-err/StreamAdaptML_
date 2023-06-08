"""
    Модуль обучения модели машинного обучения.
    Функционал:
        1) Построить модель предсказания температуры в выбранном кабинете (обучить модель на исторических данных)
        2) Обучить модель на потоковых данных с выбором метода обучения при наличии и отсутствии сдвига данных
"""
import pandas as pd
import DataPreprocessing
import Visualization
import keras.models
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.metrics.regression_metrics import mean_squared_error
import datetime
from sklearn.metrics import r2_score  #  mean_squared_error, accuracy_score
from keras.models import model_from_json

from app.modules import DataHandler


def create_model(x_train, drop_out=0.2, activation='linear', optimizer='rmsprop'):
    """
    Создает модель для прогнозирования временных рядов с использованием слоев LSTM
    :param x_train: входные данные для обучения. Должны быть трехмерным тензором, где последнее измерение представляет количество признаков.
    :param drop_out: вероятность отключения нейронов Dropout слоя для избежания переобучения (по умолчанию 0.2).
    :param activation: функция активации для последнего полносвязного слоя (по умолчанию linear).
    :param optimizer: оптимизатор для компиляции модели (по умолчанию rmsprop).
    :return: Созданная модель
    """
    model = Sequential()
    # 1 слой: принимает 3D тензор, возвращает всю выходную последовательность (many-to-many) с размерностью выходного пространства - 50
    model.add(LSTM(input_shape=[x_train.shape[-1], 1], units=50, return_sequences=True))
    # 2 слой: избегаем переобучения с помощью Dropout слоя
    model.add(Dropout(drop_out))
    # 3 слой: размерность определяется на основе предыдущего слоя, возвращает последний вывод в выходной последовательности (many-to-one)
    model.add(LSTM(units=100, return_sequences=False))
    # 4 слой: избегаем переобучения с помощью Dropout слоя
    model.add(Dropout(drop_out))
    # 5 слой: полносвязный слой, возвращающий прогнозы с размерностью выходного пространства - 1
    model.add(Dense(units=1, activation=activation))
    # model.add(Activation(activation))
    start = datetime.datetime.now()
    model.compile(loss='mse', optimizer=optimizer)
    print('Время компиляции :', datetime.datetime.now() - start)
    # print('Количество параметров сети', model.summary())
    return model


def compile_model(model, loss='mse', optimizer='rmsprop'):  # adam
    """
    Компиляция модели
    :param model: модель в формате .keras
    :param loss: функция потерь для компиляции модели (по умолчанию mse).
    :param optimizer: оптимизатор для компиляции модели (по умолчанию rmsprop).
    :return: Созданная модель
    """
    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_train_test(dataset, target_column, test_size=0.333, mode='train_from_scratch'):  # , self.train_index, self.test_index
    if mode == 'train_from_scratch':
        x_train, y_train, x_test, y_test, index_train, index_test = DataPreprocessing.get_train_test(dataset, target_column, test_size)
        return x_train, y_train, x_test, y_test, index_train, index_test
    elif mode == 'train_online':
        return None
    elif mode == 'train_mini_batch_online':
        return None
    elif mode == 'train_transfer_learning':
        return None
    else:
        raise 'Передан неизвестный режим в get_train_test()'

class ModelGeneration(object):
    current_model = None  # Текущая модель
    target_column = 'temp_audience'  # Название целевого столбца (Y)
    predicted = None  # Предсказания
    mse = None
    r2 = None
    history = None
    def __init__(self, model=None):
        """
        Инициализирует объект класса ModelGeneration.
        :param model: Загруженная модель (по умолчанию None).
        """
        if model is not None:
            # Компиляция модели
            self.current_model = compile_model(model=model)

    def create_model(self, dataset):
        """
        Создает новую модель
        :param dataset: входной датасет.
        """
        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(dataset, self.target_column, 0.333)
        self.current_model = create_model(x_train=x_train)

    def save_model(self):
        """
        Сохраняет текущую модель.
        """
        DataHandler.save_model(self.current_model)

    def train_online(self, dataset, epochs=1):  # validation_split=0.1
        """
        Метод онлайн-обучения (online learning).
        Обучает модель на потоковых данных путем последовательного обновления весов модели после каждого примера.
        :param dataset: входной датасет.
        :param epochs: количество эпох (по умолчанию 1).
        """
        x_train, y_train, x_test, y_test = \
            get_train_test(dataset, self.target_column, mode='train_online')
        start = datetime.datetime.now()
        for epoch in range(epochs):
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]
                x = x.reshape(1, *x.shape)  # Преобразование входных данных в форму (1, input_shape)
                self.current_model.train_on_batch(x, y)
            print(f"Эпоха {epoch + 1}/{epochs} - Обучение завершено")
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

    def train_mini_batch_online(self, dataset, batch_size=32, epochs=1):
        """
        Метод пакетного онлайн-обучения (mini-batch online learning).
        Обучает модель на потоковых данных путем обновления весов модели после каждого пакета данных.
        :param dataset: входной датасет.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 1).
        """
        x_train, y_train, x_test, y_test = \
            get_train_test(dataset, self.target_column, mode='train_mini_batch_online')
        start = datetime.datetime.now()
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                self.current_model.train_on_batch(x_batch, y_batch)
            print(f"Эпоха {epoch + 1}/{epochs} - Обучение завершено")
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

    def train_from_scratch(self, dataset, batch_size=32, epochs=10, test_size=0.333):
        """
        Метод обучения с нуля (from scratch).
        Обучает модель на обучающем наборе данных с нуля.
        :param dataset: входной датасет.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 10).
        :param test_size: доля тестовых данных в обучении (по умолчанию 0.333).
        """
        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(dataset, self.target_column, test_size, mode='train_from_scratch')
        start = datetime.datetime.now()
        self.history = self.current_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

        # Прогнозирование значений
        predicted_train = self.current_model.predict(x_train)
        predicted_test = self.current_model.predict(x_test)

        # Сохранение предсказаний для графика
        a1 = pd.DataFrame(index=index_train, data=predicted_train, columns=['temp'])
        a2 = pd.DataFrame(index=index_test, data=predicted_test, columns=['temp'])
        self.predicted = pd.concat([a1, a2])

        # Вычисление точности (MSE, R2)
        self.compute_mse(y_test, predicted_test.flatten())
        self.compute_r_squared(y_test, predicted_test.flatten())

        #history.history['loss']

    def train_transfer_learning(self, dataset, batch_size=32, epochs=10, test_size=0.333):
        """
        Метод трансферного обучения (transfer learning).
        Обучает модель на обучающем наборе данных, используя предварительно обученные веса модели.
        :param dataset: входной датасет.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 10).
        """
        # Заморозка весов предварительно обученной модели
        for layer in self.current_model.layers:
            layer.trainable = False

        # Компиляция модели после заморозки весов
        self.current_model.compile(loss='mse', optimizer='adam')

        start = datetime.datetime.now()

        x_train, y_train, x_test, y_test = \
            get_train_test(dataset, self.target_column, test_size, mode='train_transfer_learning')

        # Обучение модели
        self.current_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

    def train_autofit(self, dataset):
        # Инициализировать модели MS = {M1, M2, M3, M4}
        # Получить значение функции потерь для каждой модели
        # Определить модель с минимальным значением функции потерь
        # Заменить веса исходной модели M на веса модели с минимальным значением функции потерь
        pass

    def compute_mse(self, y_test, y_pred):
        self.mse = float(mean_squared_error(y_test, y_pred))
        print('\tТест MSE: %.3f' % self.mse)

    def compute_r_squared(self, y_true, y_pred):
        self.r2 = r2_score(y_true, y_pred)
        print('\tТест R2: %.3f' % self.r2)
        # numerator = np.sum(np.square(y_true - y_pred))
        # denominator = np.sum(np.square(y_true - y_mean))
        # result = 1 - numerator/denominator


    def visual_learning(self):
        if self.current_model is None:
            print('Не инициализирована модель.')
        else:
            Visualization.plot_model(self.current_model)


    # def plot_predicted(self, renderer='browser', plot_bat=False, plot_weather=False):
    #     if self.y_test is None:
    #         print('Отсутствует тестовая выборка (Запустите get_train_test())')
    #     elif self.predicted is None:
    #         print('Предсказания отсутствуют (Запустите model_predict())')
    #     else:
    #         temp_bat_col = None
    #         temp_outside_col = None
    #         if plot_bat:
    #             temp_bat_col = self.input_dataset[self.input_dataset.filter(regex='bat_temp').columns[0]]
    #         if plot_weather:
    #             temp_outside_col = self.input_dataset['temp']
    #         Visualization.plot_temps_series_predicted(self.test_index, self.y_test, self.predicted.flatten(), renderer,
    #                                                   temp_bat_col, temp_outside_col)
    #

    # Функция, запускающая все остальные
    def create_prediction_model(self, test_size=0.333, drop_out=0.2, activation='linear', optimizer='rmsprop',
                                epochs_num=20, visual_mode='png', plot_bat=False, plot_weather=False):
        if self.x_train is None:
            print("\tРазбиение датасета")
            self.get_train_test(test_size)
        print(
            f"\tx_train: {self.x_train.shape}\n\ty_train: {self.y_train.shape}\n\tx_test: {self.x_test.shape}\n\ty_test: {self.y_test.shape}")
        if self.current_model is None:
            print("Создание модели")
            self.current_model = create_model(self.x_train, drop_out, activation, optimizer=optimizer)
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
