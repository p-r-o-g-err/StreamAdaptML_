"""
    Модуль обнаружения сдвигов в данных.
    Включает функции по обнаружению сдвига в данных
"""
import matplotlib.pyplot as plt

from app.modules import DataPreprocessing


def stream_drift_detector(data_stream, drift_detector):
    """
    Функция для выявления дрейфа в потоке данных,
    на выходе возвращает список с индексами элементов, на которых произошла детекция дрейфа.

    data_stream - поток данных
    drift_detector - метод детекции дрейфа из библиотеки River
    reset_after_drift - очистить drift_detector после обнаружения дрейфа,
                        т.е. после детекции дрейфа все данные до дрейфа в дальнейшем анализе не используются.
    """
    drift_index = []
    #data_stream = DataPreprocessing.normalize_series(data_stream)
    for i, val in enumerate(data_stream):
        print('val = ', val)
        drift_detector.update(val)
        # Метод update добавляет значение элемента в окно, обновляет соответствующую статистику,
        # в данном случае общую сумму всех значений, среднее, ширину окна и общую дисперсию.
        if drift_detector.drift_detected:
            # Детектор дрейфа указывает после каждой выборки, есть ли дрейф в данных
            print(f'Зафиксирован сдвиг на индексе {i}')
            drift_index.append(i)

    return drift_index

