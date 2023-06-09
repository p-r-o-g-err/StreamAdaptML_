var ctx1 = null;
var chart1 = null;
function createChart() {
    if (chart1) {
        chart1.destroy(); // Уничтожить существующий график
    }
    ctx1 = document.getElementById('streaming-data').getContext('2d');
    chart1 = new Chart(ctx1, {
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
                label: 'Зона детекции сдвига данных', // Точки со сдвигом
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
        chart1.data.labels.push(item.date_time);
        chart1.data.datasets[0].data.push(item.temp_audience);
        // Проверить, включен ли текущий элемент в список элементов со сдвигом
        if (driftIndexes.includes(item.index)) {
            chart1.data.datasets[1].data.push({x: item.date_time, y: item.temp_audience});
        }
    }
    // Обновление графика
    chart1.update();

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
        url: '/streaming_data',
        type: 'GET',
        success: function(response) {
            var data = response.data;
            var driftIndexes = response.driftIndexes;
            console.log('Количество driftIndexes: ' + driftIndexes.length.toString())
            updateChart(data, driftIndexes);
            saveChartState(); // Сохранить состояние графика после обновления данных
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
    // Получить сохраненные данные графика из локального хранилища
    const chartData = localStorage.getItem('chartData');
    const chartOptions = localStorage.getItem('chartOptions');

    if (chartData && chartOptions) {
        // Восстановить данные графика
        chart1.data = JSON.parse(chartData);
        chart1.options = JSON.parse(chartOptions);
        chart1.update();
    }
}

function saveChartState() {
    if (chart1) {
        // Сохранить данные графика в локальное хранилище
        localStorage.setItem('chartData', JSON.stringify(chart1.data));
        localStorage.setItem('chartOptions', JSON.stringify(chart1.options));
    }
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

// Очистка параметров при первом запуске
window.onbeforeunload = function() {
    // Выполнить AJAX запрос на маршрут Flask
    $.ajax({
        url: '/first_run',
        type: 'GET',
        success: function(response) {
            var first_run_status = response.first_run_status;
            if (first_run_status == true){
                localStorage.clear();
            }
        },
        error: function(error) {
            console.log(error);
        }
    });
}

// При загрузке страницы восстанавливаем состояния
$(document).ready( function () {
    // Создание графика
    createChart();
     // Восстановить состояние графика при загрузке страницы
    restoreChartState();
    // Восстановить значение счетчика сдвигов в данных
    restoreDriftPointsCount();
    // Актуализация обновления данных
    managementWork();
});
