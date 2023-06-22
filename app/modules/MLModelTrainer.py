"""
    Модуль обучения модели машинного обучения.
    Функционал:
        1) Построить модель предсказания температуры в выбранном кабинете (обучить модель на исторических данных)
        2) Обучить модель на потоковых данных с выбором метода обучения при наличии и отсутствии сдвига данных
"""
import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM
from keras.metrics.regression_metrics import mean_squared_error
import datetime
from sklearn.metrics import r2_score
from app.modules import DataHandler, DataPreprocessing


def create_model(x_train, drop_out=0.2, activation='linear', optimizer='adam'): # rmsprop
    """
    Создает модель для прогнозирования временных рядов с использованием слоев LSTM
    :param x_train: входные данные для обучения. Должны быть трехмерным тензором, где последнее измерение представляет количество признаков.
    :param drop_out: вероятность отключения нейронов Dropout слоя для избежания переобучения (по умолчанию 0.2).
    :param activation: функция активации для последнего полносвязного слоя (по умолчанию linear).
    :param optimizer: оптимизатор для компиляции модели (по умолчанию adam).
    :return: Созданная модель.
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
    start = datetime.datetime.now()
    model.compile(loss='mse', optimizer=optimizer)
    print('Время компиляции :', datetime.datetime.now() - start)
    # print('Количество параметров сети', model.summary())
    return model


def compile_model(model, loss='mse', optimizer='adam'):  # adam
    """
    Компиляция модели
    :param model: модель в формате .keras
    :param loss: функция потерь для компиляции модели (по умолчанию mse).
    :param optimizer: оптимизатор для компиляции модели (по умолчанию adam).
    :return: Созданная модель
    """
    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_train_test(dataset, target_column, test_size=0.333, mode='train_from_scratch'):
    """
    Разделяет датасет на обучающую и тестовую выборки для указанного метода обучения модели.
    :param dataset: Датасет для разделения.
    :param target_column: Имя целевого столбца.
    :param test_size: Процент данных для тестовой выборки. (по умолчанию 0.333 (33.3%)).
    :param mode: Метод обучения модели в виде строки.
    :return: Кортеж, содержащий x_train, y_train, x_test, y_test, index_train, index_test.
    """
    if mode == 'train_from_scratch':
        x_train, y_train, x_test, y_test, index_train, index_test = \
            DataPreprocessing.get_train_test_for_train_from_scratch(dataset, target_column, test_size)
        return x_train, y_train, x_test, y_test, index_train, index_test
    elif mode == 'train_online':
        x_train, y_train, x_test, y_test, index_train, index_test = \
            DataPreprocessing.get_train_test_for_online_learning(dataset, target_column)
        return x_train, y_train, x_test, y_test, index_train, index_test
    elif mode == 'train_mini_batch_online':
        x_train, y_train, x_test, y_test, index_train, index_test = \
            DataPreprocessing.get_train_test_for_online_learning(dataset, target_column)
        return x_train, y_train, x_test, y_test, index_train, index_test
    elif mode == 'train_transfer_learning':
        x_train, y_train, x_test, y_test, index_train, index_test = \
            DataPreprocessing.get_train_test_for_transfer_learning(dataset, target_column, test_size)
        return x_train, y_train, x_test, y_test, index_train, index_test
    else:
        raise 'Передан неизвестный режим в get_train_test()'


class ModelGeneration(object):
    """
    Класс генерации, обучения и оценки модели машинного обучения.
    """
    current_model = None  # Текущая модель машинного обучения
    target_column = 'temp_audience'  # Название целевого столбца (Y)
    predicted = None  # Предсказания
    mse = None  # Среднеквадратичная ошибка
    r2 = None  # Коэффициент детерминации
    history = None  # История обучения

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

    def train_online(self, dataset, epochs=1):
        """
        Метод онлайн-обучения (online learning).
        Обучает модель на потоковых данных путем последовательного обновления весов модели после каждого примера.
        :param dataset: входной датасет.
        :param epochs: количество эпох (по умолчанию 1).
        """
        print('Обучение модели (online learning)')

        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(dataset, self.target_column, mode='train_online')
        start = datetime.datetime.now()
        for epoch in range(epochs):
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]
                x = np.array([x])
                y = np.array([y])
                self.history = self.current_model.train_on_batch(x, y)
            print(f"Эпоха {epoch + 1}/{epochs} - Обучение завершено")
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

        # Прогнозирование значений
        predicted_test = self.current_model.predict(x_test)

        # Сохранение предсказаний для графика
        self.predicted = pd.DataFrame(index=index_test, data=predicted_test, columns=[self.target_column])

        # Вычисление точности (MSE, R2)
        self.compute_mse(y_test, predicted_test.flatten())
        self.compute_r_squared(y_test, predicted_test.flatten())

    def train_mini_batch_online(self, dataset, batch_size=32, epochs=1):
        """
        Метод пакетного онлайн-обучения (mini-batch online learning).
        Обучает модель на потоковых данных путем обновления весов модели после каждого пакета данных.
        :param dataset: входной датасет.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 1).
        """
        print('Обучение модели (mini-batch online learning)')

        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(dataset, self.target_column, mode='train_mini_batch_online')
        start = datetime.datetime.now()
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                self.history = self.current_model.train_on_batch(x_batch, y_batch)
            print(f"Эпоха {epoch + 1}/{epochs} - Обучение завершено")
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

        # Прогнозирование значений
        predicted_test = self.current_model.predict(x_test)

        # Сохранение предсказаний для графика
        self.predicted = pd.DataFrame(index=index_test, data=predicted_test, columns=[self.target_column])

        # Вычисление точности (MSE, R2)
        self.compute_mse(y_test, predicted_test.flatten())
        self.compute_r_squared(y_test, predicted_test.flatten())

    def train_from_scratch(self, dataset, batch_size=32, epochs=10, test_size=0.333):
        """
        Метод обучения с нуля (from scratch).
        Обучает модель на обучающем наборе данных с нуля.
        :param dataset: входной датасет.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 10).
        :param test_size: доля тестовых данных (по умолчанию 0.333).
        """
        print('Обучение модели (learning from scratch)')

        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(dataset, self.target_column, test_size, mode='train_from_scratch')

        start = datetime.datetime.now()
        self.history = self.current_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

        # Прогнозирование значений
        predicted_train = self.current_model.predict(x_train)
        predicted_test = self.current_model.predict(x_test)

        # Сохранение предсказаний для графика
        a1 = pd.DataFrame(index=index_train, data=predicted_train, columns=[self.target_column])
        a2 = pd.DataFrame(index=index_test, data=predicted_test, columns=[self.target_column])
        self.predicted = pd.concat([a1, a2])

        # Вычисление точности (MSE, R2)
        self.compute_mse(y_test, predicted_test.flatten())
        self.compute_r_squared(y_test, predicted_test.flatten())

    def train_transfer_learning(self, dataset, batch_size=32, epochs=10, test_size=0.2):
        """
        Метод трансферного обучения (transfer learning).
        Обучает модель на обучающем наборе данных, используя предварительно обученные веса модели.
        :param dataset: входной датасет.
        :param batch_size: размер пакета для обновления весов (по умолчанию 32).
        :param epochs: количество эпох (по умолчанию 10).
        :param test_size: доля тестовых данных (по умолчанию 0.2).
        """
        print('Обучение модели (transfer learning)')

        x_train, y_train, x_test, y_test, index_train, index_test = \
            get_train_test(dataset, self.target_column, test_size, mode='train_transfer_learning')

        # Обучение модели
        start = datetime.datetime.now()
        # Заморозка слоев LSTM и Dropout
        for layer in self.current_model.layers[:-1]:
            layer.trainable = False
        # Компиляция модели после заморозки слоев
        self.current_model.compile(loss='mse', optimizer='adam')
        print('Обучение после заморозки слоев')
        self.current_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        # Разморозка и дообучение некоторых слоев модели
        for layer in self.current_model.layers[:-1]:
            layer.trainable = True
        # Компиляция модели после разморозки c низким learning_rate
        self.current_model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.0001))  # Низкий learning rate
        # Дообучение модели на новых данных
        print('Дообучение после разморозки слоев')
        self.history = self.current_model.fit(x_test, y_test, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        print('Время обучения : {}'.format(datetime.datetime.now() - start))

        # Компиляция модели после обучения с целью восстановления исходной скорости обучения
        self.current_model.compile(loss='mse', optimizer='adam')

        # Прогнозирование значений
        predicted_train = self.current_model.predict(x_train)
        predicted_test = self.current_model.predict(x_test)

        # Сохранение предсказаний для графика
        a1 = pd.DataFrame(index=index_train, data=predicted_train, columns=[self.target_column])
        a2 = pd.DataFrame(index=index_test, data=predicted_test, columns=[self.target_column])
        self.predicted = pd.concat([a1, a2])

        # Вычисление точности (MSE, R2)
        self.compute_mse(y_test, predicted_test.flatten())
        self.compute_r_squared(y_test, predicted_test.flatten())

    def train_autofit(self, dataset):
        """
        Метод автоподбора (алгоритм адаптации).
        Инициализирует модели Ms = {M1, M2, M3, M4}.
        Получает значение функции потерь для каждой модели.
        Определяет модель с минимальным значением функции потерь.
        Заменяет параметры исходной модели M на параметры модели с минимальным значением функции потерь.
        :param dataset: входной датасет.
        """
        print('Обучение модели (adaptive algorithm)')
        # Создание копий модели
        model_1 = ModelGeneration(self.current_model)
        model_2 = ModelGeneration(self.current_model)
        model_3 = ModelGeneration(self.current_model)
        model_4 = ModelGeneration(self.current_model)

        # Обучение моделей на разных подходах
        model_1.train_online(dataset)
        model_2.train_mini_batch_online(dataset)
        model_3.train_from_scratch(dataset)
        model_4.train_transfer_learning(dataset)

        # Выбор модели с наименьшим значением функции потерь
        min_mse = min(model_1.mse, model_2.mse, model_3.mse, model_4.mse)
        if min_mse == model_1.mse:
            self.copy_model_data(model_1)
        elif min_mse == model_2.mse:
            self.copy_model_data(model_2)
        elif min_mse == model_3.mse:
            self.copy_model_data(model_3)
        elif min_mse == model_4.mse:
            self.copy_model_data(model_4)

    def compute_mse(self, y_true, y_pred):
        """
        Вычисляет значение метрики MSE.
        :param y_true: истинные значения целевого столбца.
        :param y_pred: предсказанные значения целевого столбца.
        """
        self.mse = float(mean_squared_error(y_true, y_pred))
        print('\tТест MSE: %.3f' % self.mse)

    def compute_r_squared(self, y_true, y_pred):
        """
        Вычисляет значение метрики MSE.
        :param y_true: истинные значения целевого столбца.
        :param y_pred: предсказанные значения целевого столбца.
        """
        self.r2 = float(r2_score(y_true, y_pred))
        print('\tТест R2: %.3f' % self.r2)

    def copy_model_data(self, copied_model):
        """
        Присваивает полям текущей модели значения полей копируемой модели.
        :param copied_model: Копируемая модель.
        """
        self.current_model = copied_model.current_model
        self.predicted = copied_model.predicted
        self.mse = copied_model.mse
        self.r2 = copied_model.r2
        self.history = copied_model.history
