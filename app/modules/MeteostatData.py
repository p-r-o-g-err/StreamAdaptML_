"""
    Модуль получения метеоданных из Meteostat.
"""


from datetime import datetime, timedelta
from meteostat import Point, Hourly, Daily, Monthly 
from geopy.geocoders import Nominatim
from app.modules import DataPreprocessing

# Получить координаты точки по адресу
def get_coords_of_address(address):
    """
    Получает координаты точки по адресу.
    :param address: Адрес точки (строка).
    :return: Кортеж с широтой и долготой точки (float).
    """
    #Указываем используемый geocoder 
    geolocator = Nominatim(user_agent="Tester")
    #Создаем переменную, которая состоит из данных о точке локации адреса
    location = geolocator.geocode(address)
    #Возвращаем широту и долготу в виде кортежа
    return location.latitude, location.longitude


# Получить данные о погоде в точке за заданный период с заданной частотой
def get_weather_info(start='2021-01-02 00:00:00', end=None, mode='Hourly', res_period=None,
                     address='улица Перекопская, 15а, Тюмень, Россия', logging=True):
    """
    Получает данные о погоде в заданной точке за заданный период с заданной частотой.
    :param start: Начальная дата и время в формате "ГГГГ-ММ-ДД ЧЧ:ММ:СС" (по умолчанию: '2021-01-02 00:00:00').
    :param end: Конечная дата и время в формате "ГГГГ-ММ-ДД ЧЧ:ММ:СС" (по умолчанию True).
    :param mode: Режим данных ('Hourly', 'Daily', 'Monthly') (по умолчанию: 'Hourly').
    :param res_period: Период округления данных ('T', 'H', 'D', 'M', 'Y', и т.д.) (по умолчанию: None (без округления)).
    :param address: Адрес точки, для которой требуется получить данные о погоде (строка) (по умолчанию: 'улица Перекопская, 15а, Тюмень, Россия').
    :param logging: Флаг для логирования процесса (по умолчанию True).
    :return: DataFrame с данными о погоде.
    """
    if logging:
        print("Получение данных о погоде из meteostat: \n\tПолучение координат заданного адреса")
    location = get_coords_of_address(address)  
    if logging:
        print("\tПреобразование точки локации в точку Meteostat")
    point = Point(location[0], location[1])
    if logging:
        print("\tПреобразование границ указанного периода времени к типу datetime")

    start = DataPreprocessing.str_to_date_time(start)
    if end is None:
        end = DataPreprocessing.str_to_date_time(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        end = DataPreprocessing.str_to_date_time(end)
    if logging:
        print("\tПолучение данных за указанный период с указанной частотой")
    if mode == "Hourly":
        data = Hourly(point, start, end)
    elif mode == "Daily":
        data = Daily(point, start, end)
    elif mode == "Monthly":
        data = Monthly(point, start, end)
    else: 
        print('Не удалось получить данные, так как указана неизвестная частота')
        return None
    # Извлечь DataFrame с данными
    data = data.fetch()
    # Переименовать столбец time в date_time
    data = data.reset_index()
    data = data.rename(columns={"time": "date_time"})
    data = data.set_index('date_time')
    if res_period is not None:
        if logging:
            print("\t\tОкругление данных до указанного периода")
        data = data.resample(res_period).mean().interpolate()
        if start is not None:
            data = data.reset_index()
            data = data[data['date_time'] > start]
            data = data.set_index('date_time')

    return data.round(2)
