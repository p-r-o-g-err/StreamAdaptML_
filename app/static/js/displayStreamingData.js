var ctx = null;
var chart = null;
function createChart() {
    ctx = document.getElementById('streaming-data').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Температура воздуха в кабинете',
                data: [],
                borderColor: 'green',
                pointStyle: false,
                fill: false,
                borderWidth: 2
            }, {
                label: 'Точки со сдвигом',
                data: [],
                borderColor: 'red',
                pointStyle: 'circle',
                backgroundColor: 'red',
                radius: 4,
                fill: false,
                showLine: false
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

// Отображение данных на графике
function updateChart(data, driftIndexes) {
    for (let i = 0; i < data.length; i++) {
        const item = data[i];
        chart.data.labels.push(item.date_time);
        chart.data.datasets[0].data.push(item.temp);
        console.log('driftIndexes: ' + driftIndexes + ' index: ' + item.index + ' result: ' + driftIndexes.includes(item.index))
        // Проверить, включен ли текущий элемент в список элементов со сдвигом
        if (driftIndexes.includes(item.index)) {
            chart.data.datasets[1].data.push({x: item.date_time, y: item.temp});
        }
    }
    // Обновление графика
    chart.update();

    // Обновление значения number-drift-points
    const numberDriftPointsElement = document.getElementById('number-drift-points');
    if (numberDriftPointsElement) {
        numberDriftPointsElement.textContent = driftIndexes.length.toString();

        // Сохранить значение driftIndexes.length в локальное хранилище
        localStorage.setItem('driftPointsCount', driftIndexes.length.toString());
    }
}


// Загрузка и отображение исходных данных
function fetchData() {
    // Выполнить AJAX запрос на маршрут Flask
    $.ajax({
        url: '/data',
        type: 'GET',
        success: function(response) {
            var data = response.data;
            var driftIndexes = response.driftIndexes;
            updateChart(data, driftIndexes);
        },
        error: function(error) {
            console.log(error);
        }
    });
}


function managementWork() {
    $.get('/check_status', function(data) {
        // Если обучение запущено
        if (data === 'Фоновая работа запущена') {
            console.log('Работа запущена')
            // Установить обновление данных и графика каждые n секунд
            interval = setInterval(function() {
                fetchData();
            }, 1 * 1000);
            localStorage.setItem("streamingDataUpdateInterval", interval);
        }
        // Иначе остановить обновление
        else {
            console.log('Работа остановлена')
            interval = localStorage.getItem("streamingDataUpdateInterval");
            if (interval != null) {
                clearInterval(interval);
                interval = null;
                localStorage.setItem("streamingDataUpdateInterval", interval);
            }
        }
    });
}


function restoreChartState() {
    // Выполнить AJAX запрос на маршрут Flask
    $.ajax({
        url: '/chart_streaming_data',
        type: 'GET',
        success: function(response) {
            var data = response.data;
            var driftIndexes = response.driftIndexes;
            updateChart(data, driftIndexes);
        },
        error: function(error) {
            console.log(error);
        }
    });
}


function restoreDriftPointsCount() {
    // Получить сохраненное значение driftPointsCount из локального хранилища
    const driftPointsCount = localStorage.getItem('driftPointsCount');
    if (driftPointsCount) {
        // Обновить значение number-drift-points
        const numberDriftPointsElement = document.getElementById('number-drift-points');
        if (numberDriftPointsElement) {
            numberDriftPointsElement.textContent = driftPointsCount;
        }
    }
}


// При загрузке страницы вызываем функцию восстановления состояния
$(document).ready( function () {
    // Создание графика
    createChart();
    // Восстанавление данных графика
    restoreChartState();
    // Восстановить значение счетчика сдвигов в данных
    restoreDriftPointsCount();
    // Актуализация обновления данных
    managementWork();
});
