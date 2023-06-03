'''
    Описание конфигураций проекта
'''
import os

class Config(object):
    DEBUG = True  # Режим дебага
    HOST = '0.0.0.0'  # Здесь указываем адрес хоста
    PORT = 5000  # Здесь указываем номер порта
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads') # Путь к папке с загруженными файлами
    MODELS_FOLDER = os.path.join(UPLOAD_FOLDER, 'models') # Путь к папке с загруженными моделями
    DATA_FOLDER = os.path.join(UPLOAD_FOLDER, 'data') # Путь к папке с поступающими данными
    SETTINGS_FULLNAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json') # Путь к файлу с настройками
    #ALLOWED_EXTENSIONS = {'keras'}  # Разрешенные форматы моделей
    #MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # Максимальный размер загружаемой модели - 100 MB
    LOADED_MODEL = None  # Имя загруженной модели
    IS_LEARNING_STARTED = False  # Флаг, сигнализирующей о запуске обучения
    