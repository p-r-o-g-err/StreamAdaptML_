# """
#     Модуль визуализации.
#     Функционал:
#         1. Построение графика обучения
#         2.
# """
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import seaborn as sns
#
# # Построить линейную диаграмму по dataframe
# def plot_chart(data, y = [], subplots = False, split_by_columns = False, name = "", weight = 15, height = 15):
#     # Если столбцы указаны
#     if y != []:
#         # Создаем диаграмму с выбранными столбцами
#         try:
#             if split_by_columns:
#                 # Создаем диаграмму для каждого столбца
#                 for column in y:
#                     data[column].plot(subplots=False, figsize=(weight, height), legend=column, title=name)
#                     plt.show()
#             else:
#                 # Создаем диаграмму со всеми столбцами
#                 data.plot(y=y, subplots=subplots, figsize=(weight, height), title=name)
#         except:
#             print("Не все названия столбцов соответствуют DataFrame")
#     else:
#         if split_by_columns:
#             # Создаем диаграмму для каждого столбца
#             for column in data.columns:
#                 data[column].plot(subplots=False, figsize=(weight, height), legend=column, title=name)
#                 plt.show()
#         else:
#             # Создаем диаграмму со всеми столбцами
#             data.plot(subplots=subplots, figsize=(weight, height), title=name)
#     plt.show()
#
# # Построить корреляционную матрицу признаков
# def plot_correlation_matrix(data):
#     correlation = data.corr().abs()
#     plt.figure(figsize=(7,5))
#     sns.heatmap(correlation, vmax=1, annot=True, fmt= '.1f') #square=True,
#     plt.show()
#
# # Построить тепловую карту пропущенных значений в столбцах
# def plot_heatmap_of_missing_values(data, reset_index = True):
#     colours = ['#000099', '#ffff00']
#     if reset_index:
#         data1 = data.reset_index()
#         sns.heatmap(data1[data1.columns[1:]].isnull(), cmap=sns.color_palette(colours))
#     else:
#         sns.heatmap(data[data.columns].isnull(), cmap=sns.color_palette(colours))
#
#
# # Отобразить предсказанные, наблюдаемые значения, также опционально можно выбрать температуру батарей и улицы
# def plot_temps_series_predicted(dates, test, predicted, renderer='browser', temp_bat_col = None, temp_outside = None):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=dates, y=predicted, name="predicted",  mode='lines', marker={ 'color': 'red' },  hovertemplate='%{x|%d-%b-%Y %H:00}: %{y} °C'))
#     fig.add_trace(go.Scatter(x=dates, y=test, name="temperature", mode='lines', marker={ 'color': 'blue' }, hovertemplate='%{x|%d-%b-%Y %H:00}: %{y} °C'))
#     if temp_bat_col is not None:
#         fig.add_trace(go.Scatter(x=dates, y=temp_bat_col, name="battery", mode='lines', marker={ 'color': 'maroon' }, hovertemplate='%{x|%d-%b-%Y %H:00}: %{y} °C'))
#     if temp_outside is not None:
#         fig.add_trace(go.Scatter(x=dates, y=temp_outside, name="weather", mode='lines', marker={ 'color': 'teal' }, hovertemplate='%{x|%d-%b-%Y %H:00}: %{y} °C'))
#
#     fig.update_layout(
#       title="Результат прогнозирования модели",
#       font=dict(family="Franklin Gothic", size=18),
#       height=500,
#       width=1400)
#     fig.show(renderer=renderer)
#
# # Отобразить обучение модели
# def plot_model(model):
#     plt.figure(figsize = (10, 5))
#     plt.plot(model.history.history['loss'], label = 'Loss')
#     plt.plot(model.history.history['val_loss'], label = 'Val_Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.grid()
#     plt.legend()
#     plt.show()
#
# # Отобразить температуру
# def plot_temperature(dataset, columns = [], name = '', renderer='browser'):
#     fig = go.Figure()
#     if columns == []: columns = dataset.columns
#     for column in columns:
#         fig.add_trace(go.Scatter(x=dataset.index, y=dataset[column], name=column, hovertemplate='%{x|%d-%b-%Y %H:00}: %{y} °C'))
#     fig.update_layout(
#         title=name.replace("_", " "),
#         font=dict(family="Franklin Gothic", size=18),
#         height=500,
#         width=1400)
#     fig.show(renderer=renderer)
#
# # отрисовка найденных аномалий
# def plot_anomalies(dataset, target_column, renderer='browser', test_index = None):
#     anomalies = dataset[dataset["anomaly"] == 1]
#     fig = go.Figure()
#     hovertemplate = '%{x|%d-%b-%Y %H:00}: %{y} °C'
#     if test_index is not None and len(test_index) != len(dataset): dataset = dataset.loc[test_index]
#     fig.add_trace(go.Scatter(x=dataset.index,y=dataset[target_column],
#                            name="temperature", mode='lines', marker={ 'color': 'blue' },
#                            hovertemplate=hovertemplate))
#     fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[target_column],
#                            name="anomaly", mode='markers', marker={ 'color': 'red' },
#                            hovertemplate=hovertemplate))
#     fig.update_layout(
#         title="Результат обнаружения аномалий",
#         font=dict(family="Franklin Gothic", size=18),
#         height=500,
#         width=1400)
#     fig.show(renderer=renderer)
#
# def plot_data_distribution(data):
#     ax = sns.distplot(data, bins=50, kde=True, color='red', hist_kws={"linewidth": 15,'alpha':1})
#     ax.set(xlabel='Распределение', ylabel='Частота')
#     plt.show()