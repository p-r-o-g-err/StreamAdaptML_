var ctx3 = null; // Переменная, содержащая контекст для рисования графика
var chart3 = null; // Переменная, содержащая объект графика

// Функция, создающая график
function createChart3() {
    if (chart3) {
        chart3.destroy(); // Уничтожить существующий график
    }
    ctx3 = document.getElementById('model-predictions').getContext('2d');
    chart3 = new Chart(ctx3, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Фактические значения',
                data: [],
                borderColor: 'blue',
                pointStyle: false,
                fill: false,
                borderWidth: 2
            }, {
                label: 'Предсказанные значения',
                data: [],
                borderColor: 'red',
                pointStyle: false,
                fill: false,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Время'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Температура'
                    }
                }
            }
        }
    });
}

// Функция, обновляющая график
function updateChart3(true_temp, pred_temp) {
    for (let i = 0; i < true_temp.length; i++) {
        const item = true_temp[i];
        chart3.data.labels.push(item.date_time);
        chart3.data.datasets[0].data.push(item.temp_audience);

        const item2 = pred_temp[i];
        chart3.data.datasets[1].data.push({x: item2.index, y: item2.temp_audience});
        console.log('Прогнозы модели - date_time: ' + item.date_time + ' true_temp: ' + item.temp_audience + ' pred_temp: ' + item2.temp_audience)
    }

    // Обновление графика
    chart3.update();
}

// Функция, выполняющая AJAX-запрос на получение данных и обновляющая график
function fetchData3() {
    // Выполнить AJAX запрос на маршрут Flask
    $.ajax({
        url: '/predictions_data',
        type: 'GET',
        success: function(response) {
            var true_temp = response.true_temp;
            var pred_temp = response.pred_temp;
            updateChart3(true_temp, pred_temp);
            // Сохранить состояние графика после обновления данных
            saveChartState3();
        },
        error: function(error) {
            console.log(error);
        }
    });
}

// Функция, инициирующая запуск и остановку получения обновлений для графика и параметров
function managementWork3() {
    $.get('/check_status', function(data) {
        // Если обучение запущено
        if (data === 'Фоновая работа запущена') {
            // Установить обновление данных и графика каждые n секунд
            interval = setInterval(function() {
                fetchData3();
            }, 1.5 * 1000);
            localStorage.setItem("predictionsDataUpdateInterval", interval);
        }
        // Иначе остановить обновление
        else {
            interval = localStorage.getItem("predictionsDataUpdateInterval");
            if (interval != null) {
                clearInterval(interval);
                interval = null;
                localStorage.setItem("predictionsDataUpdateInterval", interval);
            }
        }
    });
}

// Функция, восстанавливающая состояние графика
function restoreChartState3() {
    // Получить сохраненные данные графика из локального хранилища
    const chartData = localStorage.getItem('chartData3');
    const chartOptions = localStorage.getItem('chartOptions3');

    if (chartData && chartOptions) {
        // Восстановить данные графика
        chart3.data = JSON.parse(chartData);
        chart3.options = JSON.parse(chartOptions);
        chart3.update();
    }
}

// Функция, сохраняющая состояние графика
function saveChartState3() {
    if (chart3) {
        // Сохранить данные графика в локальное хранилище
        localStorage.setItem('chartData3', JSON.stringify(chart3.data));
        localStorage.setItem('chartOptions3', JSON.stringify(chart3.options));
    }
}

// При загрузке страницы создаем график, восстанавливаем состояние графика и параметров, обновляем данные
$(document).ready( function () {
    // Создание графика
    createChart3();
    // Восстановить состояние графика при загрузке страницы
    restoreChartState3();
    // Актуализация обновления данных
    managementWork3();
});
