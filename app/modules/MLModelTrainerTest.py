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
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
import datetime
from sklearn.metrics import mean_squared_error, accuracy_score
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
        x_train, y_train, x_test, y_test = DataPreprocessing.get_train_test(dataset, target_column, test_size)
        return x_train, y_train, x_test, y_test
    elif mode == 'train_online':
        return None
    elif mode == 'train_mini_batch_online':
        return None
    elif mode == 'train_transfer_learning':
        return None
    else:
        raise 'Передан неизвестный режим в get_train_test()'

class ModelGeneration(object):
    input_dataset = None  # Датасет
    current_model = None  # Текущая модель
    target_column = 'temp_audience'  # Название целевого столбца (Y)
    x_train = None  # Данные для обучения модели
    y_train = None  # Метки для обучения модели
    x_test = None  # Тестовые данные
    y_test = None  # Тестовые метки

    # train_index = None  # Индексы тренировочных данных
    # test_index = None  # Индексы тестовых данных
    predicted = None  # Предсказания

    def __init__(self, dataset=None, model=None):
        """
        Инициализирует объект класса ModelGeneration.
        :param dataset: Датасет для обучения модели.
        :param model: Загруженная модель (по умолчанию None).
        """
        if dataset is not None:
            self.input_dataset = dataset
        if model is not None:
            # Компиляция модели
            self.current_model = compile_model(model=model)

    def create_model(self, dataset=None):
        if dataset is not None:
            self.input_dataset = dataset
        if self.input_dataset is not None:
            self.x_train, self.y_train, self.x_test, self.y_test = \
                get_train_test(self.input_dataset, self.target_column, 0.333)
            self.current_model = create_model(x_train=self.x_train)
        else:
            raise 'Датасет не найден'

    def save_model(self):
        """
        Сохраняет текущую модель.
        """
        DataHandler.save_model(self.current_model)

    def set_dataset(self, dataset):
        self.input_dataset = dataset

    def train_online(self, x_train, y_train, epochs=1):  # validation_split=0.1
        """
        Метод онлайн-обучения (online learning).
        Обучает модель на потоковых данных путем последовательного обновления весов модели после каждого примера.
        :param x_train: входные данные для обучения.
        :param y_train: целевые значения для обучения.
        :param epochs: количество эпох (по умолчанию 1).
        """
        start = datetime.datetime.now()
        for epoch in range(epochs):
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]
                x = x.reshape(1, *x.shape)  # Преобразование входных данных в форму (1, input_shape)
                self.current_model.train_on_batch(x, y)
            print(f"Эпоха {epoch + 1}/{epochs} - Обучение завершено")
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

    def train_mini_batch_online(self, x_train, y_train, batch_size=32, epochs=1):
        """
        Метод пакетного онлайн-обучения (mini-batch online learning).
        Обучает модель на потоковых данных путем обновления весов модели после каждого пакета данных.
        :param x_train: входные данные для обучения.
        :param y_train: целевые значения для обучения.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 1).
        """
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                self.current_model.train_on_batch(x_batch, y_batch)
            print(f"Эпоха {epoch + 1}/{epochs} - Обучение завершено")

    def train_from_scratch(self, batch_size=32, epochs=10, test_size=0.333):
        """
        Метод обучения с нуля (from scratch).
        Обучает модель на обучающем наборе данных с нуля.
        :param x_train: входные данные для обучения.
        :param y_train: целевые значения для обучения.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 10).
        :param test_size: доля тестовых данных в обучении (по умолчанию 0.333).
        """
        self.x_train, self.y_train, self.x_test, self.y_test = \
            get_train_test(self.input_dataset, self.target_column, test_size)
        self.current_model.fit(x=self.x_train, y=self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def train_transfer_learning(self, x_train, y_train, batch_size=32, epochs=10):
        """
        Метод трансферного обучения (transfer learning).
        Обучает модель на обучающем наборе данных, используя предварительно обученные веса модели.
        :param x_train: входные данные для обучения.
        :param y_train: целевые значения для обучения.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 10).
        """
        # Заморозка весов предварительно обученной модели
        for layer in self.current_model.layers:
            layer.trainable = False

        # Компиляция модели после заморозки весов
        self.current_model.compile(loss='mse', optimizer='adam')

        # Обучение модели
        self.current_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # def train_autofit(self):
    #     pass

    def predict(self, x_test):
        """
        Прогнозирование на данных.
        :param x_test: входные данные для прогнозирования.
        :return: предсказанные значения.
        """
        if x_test is not None:
            self.x_test = x_test
        if self.x_test is None:
            print('Отсутствует тестовая выборка (Запустите get_train_test())')
        else:
            self.predicted = self.current_model.predict(self.x_test)

    def compute_error(self):
        """
        Вычисление ошибки прогноза.
        :param y_true: фактические значения.
        :param y_pred: предсказанные значения.
        :return: значение ошибки.
        """
        if self.y_test is None:
            print('Отсутствует тестовая выборка')
        elif self.predicted is None:
            print('Предсказания отсутствуют (Запустите predict())')
        else:
            accuracy = mean_squared_error(self.y_test, self.predicted.flatten(), squared=False)
            print('\tТест MSE: %.3f' % (accuracy))
            return accuracy

        # mse = mean_squared_error(y_true, y_pred, squared=False)
        # print('\tТест MSE: %.3f' % mse)
        # return mse

    def compute_accuracy(self):
        #MSE
        #R2
        pass

    # def compute_accuracy(self, y_true, y_pred, threshold=0.5):
    #     """
    #     Вычисление точности модели.
    #     :param y_true: фактические значения.
    #     :param y_pred: предсказанные значения.
    #     :param threshold: пороговое значение для классификации (по умолчанию 0.5).
    #     :return: значение точности.
    #     """
    #     if self.y_test is None: print('Отсутствует тестовая выборка (Запустите get_train_test())')
    #     elif self.predicted is None: print('Предсказания отсутствуют (Запустите model_predict())')
    #     else:
    #         accuracy = mean_squared_error(self.y_test, self.predicted.flatten(), squared=False)
    #         print('\tТест MSE: %.3f' % (accuracy))
    #         return accuracy

        # y_pred_classes = (y_pred > threshold).astype(int)
        # accuracy = accuracy_score(y_true, y_pred_classes)
        # return accuracy



    # def fit_model(self, epochs_num):
    #     if self.current_model is None:
    #         print('Не инициализирована модель (Укажите модель)')
    #     elif (self.x_train is None) or (self.y_train is None):
    #         print('Отсутствует тренировочная выборка (Запустите get_train_test())')
    #     else:
    #         fit_model(self.current_model, self.x_train, self.y_train, epochs=epochs_num)

    def visual_learning(self):
        if self.current_model is None:
            print('Не инициализирована модель.')
        else:
            Visualization.plot_model(self.current_model)

    # def model_predict(self):
    #     if self.x_test is None:
    #         print('Отсутствует тестовая выборка (Запустите get_train_test())')
    #     else:
    #         self.predicted = self.current_model.predict(self.x_test)

    # def print_error(self):
    #     if self.y_test is None:
    #         print('Отсутствует тестовая выборка (Запустите get_train_test())')
    #     elif self.predicted is None:
    #         print('Предсказания отсутствуют (Запустите model_predict())')
    #     else:
    #         print_error(self.y_test, self.predicted.flatten())

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
