"""
    Модуль получения данных датчиков.
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup


def url_is_valid(url):
    """
    Проверяет доступность URL-адреса.
    :param url: URL-адрес для проверки доступности (строка).
    :return: True, если URL-адрес доступен. False, если URL-адрес недоступен.
    """
    try:
        requests.get(url)
        return True
    except Exception as error:
        print(f"Не удается установить информацию о датчике по URL-адресу: {url} :")
        print(repr(error))
        return False


def get_sensor_info_from_url(url):
    """
    Получает данные датчика по указанной ссылке.
    :param url: URL-адрес, по которому необходимо получить данные датчика (строка).
    :return: Датафрейм с данными датчика или пустой датафрейм, если ссылка недействительна.
    """
    if url_is_valid(url):
        return pd.read_csv(url, sep=" ", header=None, names=["date", "time", "temp"])
    else:
        print('get_sensor_info_from_url вернул пустой DataFrame')
        return pd.DataFrame()


def get_sensor_info(start=None, url='https://sensors.mwlabs.ru/', res_period="10T", fill_in_the_gaps=True, logging=True):
    """
    Получает данные со всех датчиков.
    :param start: Начальная дата и время для фильтрации данных (по умолчанию None).
    :param url: URL-адрес главной страницы, откуда будут получены ссылки на датчики (по умолчанию 'https://sensors.mwlabs.ru/').
    :param res_period: Период округления данных (по умолчанию "10T").
    :param fill_in_the_gaps: Флаг для заполнения пропущенных значений (по умолчанию True).
    :param logging: Флаг для логирования процесса (по умолчанию True).
    :return: Датафрейм с данными со всех датчиков.
    """
    if logging:
        print("Получение данных со всех датчиков\n\tСчитывание ссылок и названий датчиков")
    # Получить ссылки на датчики с главной страницы
    r = requests.post(url)
    data = None
    if r.status_code in [200, 302]:
        data = r.text
    else: 
        print('get_sensor_info вернул пустой DataFrame')
        return pd.DataFrame()

    soup = BeautifulSoup(data, 'lxml') 
    name_sensors = []
    url_sensors = []

    # Получение названий датчиков и ссылок на их показания
    for tr in soup.table:
        # Сохраняем название аудитории
        if len(str(tr).split()) > 0:
            if ('wall' not in tr.text) and ('bat' not in tr.text):
                name_auditorium = tr.text 
            for child in tr:
                for child2 in child:
                    if 'href="/view/' in str(child2):
                        url = str(child2) 
                        url = url.removeprefix('<a href="/').removesuffix('</a>') 
                        url, name = url.split('" target="_blank">')
                        url_sensors.append('https://sensors.mwlabs.ru/' + url)
                        name_sensors.append(name_auditorium + '_' + name)

    # Получить данные с датчиков по полученным ссылкам
    sensors_info = dict()
    for i in range(len(name_sensors)):
        if logging:
            print(f"\tСчитывание данных датчика {name_sensors[i]}")
        sensors_info[name_sensors[i]] = get_sensor_info_from_url(url_sensors[i])
        # Переименовать столбец temp для каждого датчика
        sensors_info[name_sensors[i]] = sensors_info[name_sensors[i]].rename(columns={"temp": name_sensors[i] + "_temp"})

    if logging:
        print("\tФормирование итогового датафрейма")
    df = pd.concat(sensors_info.values(), sort=True)
    if logging:
        print("\t\tДобавление столбца date_time")
    df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%Y-%m-%d %H:%M:%S")
    if logging:
        print("\t\tУдаление столбцов date и time")
    df = df.drop(["date","time"],axis = 1)
    if start is not None:
        if logging:
            print("\t\tУдаление данных раньше", start)
        df = df[df['date_time'] >= start]
    if logging:
        print("\t\tНазначение столбца date_time в качестве индекса")
    df = df.set_index('date_time')
    if res_period is not None:
        if logging:
            print("\t\tОкругление данных до указанного периода")
        df = df.resample(res_period).mean()
    # Если включен режим устранения пропусков
    if fill_in_the_gaps:
        if logging:
            print("\t\tЗаполняем пропущенные значения интерполяцией")
        df = df.interpolate(method ='linear', limit_direction ='forward')     
        # Заполняем пропущенное значение первой записи следующим значением (интерполяция не захватывает первую строку)
        df = df.fillna(method ='bfill') 
    return df.round(2)
