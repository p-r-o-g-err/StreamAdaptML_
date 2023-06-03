from river import drift
import numpy as np
import time
from progress.bar import IncrementalBar 
import matplotlib.pyplot as plt 

def fit_on_batch(model, x_train, y_train):
    model.train_on_batch(x_train, y_train)

def stream_learn(loaded_model,dataset):
    adwin = drift.ADWIN()
    relearn = False 
    print('Онлайн обучение модели')
    time.sleep(1)
    loaded_model.compile(loss = 'mse', optimizer='adam')
    bar = IncrementalBar('\t\tProgress', max = len(dataset)) 
    for index, row in dataset.iterrows():
        target_column = list(dataset.filter(regex='wall_temp'))[0]
        Y = row[target_column]  
        X = np.array([list(row.drop(target_column))]) 
        prediction = loaded_model.predict_on_batch(X)[0][0]  
        fit_on_batch(loaded_model, X, np.array([Y]))
        #print(f'Y {Y} - prediction {prediction}')
        in_drift, _ = adwin.update(Y)
        if in_drift:
            relearn = True
            adwin.reset()
        bar.next()
    bar.finish()
    time.sleep(1)
    return loaded_model,relearn 

def batch_stream_learn(loaded_model, dataset, batch_size):
    print('Пакетное потоковое обучение модели') 
    loaded_model.compile(loss='mse', optimizer='adam') 
    x_train, y_train = [], [] 
    # Взять первые 100 записей (для тестирования)
    dataset = dataset.head(100)
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
            loaded_model.fit(x_train, y_train, batch_size=batch_size, epochs=10)
            loss.append(loaded_model.history.history['loss'])
            x_train, y_train = [], [] 
            counter += 1 
    # Визуализация (позже вынести в модуль Visualization.py)
    # Вывод потерь со всех эпох всех пакетов

    # loss_all = sum(loss, [])
    # plt.figure(figsize=(10, 5))
    # plt.plot(loss_all, label='Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.show()
    # # Вывод средних потерь со всех пакетов
    # loss_len = [sum(l)/len(l) for l in loss]
    # plt.figure(figsize=(10, 5))
    # plt.plot(loss_len, label='Average loss')
    # plt.xlabel('Batches')
    # plt.ylabel('Average loss')
    # plt.grid()
    # plt.show() 
    return loaded_model, loss