var ctx2 = null;
var chart2 = null;
function createChart2() {
    if (chart2) {
        chart2.destroy(); // Уничтожить существующий график
    }
    ctx2 = document.getElementById('model-performance').getContext('2d');
    chart2 = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Потери',
                data: [],
                borderColor: 'blue',
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
                        text: 'Пакеты'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Потери'
                    }
                }
            }
        }
    });
}

function updateParameters(training_time) {
    // Обновление значения number-drift-points
    const timeValueElement = document.getElementById('time-value');
    if (timeValueElement) {
        timeValueElement.textContent = training_time;

        // Сохранить значение driftIndexes.length в локальное хранилище
        localStorage.setItem('timeValue', training_time);
    }
}

// Загрузка и отображение исходных данных
function fetchData2() {
    // Выполнить AJAX запрос на маршрут Flask
    $.ajax({
        url: '/training_data',
        type: 'GET',
        success: function(response) {
            var training_time = response.training_time;
            updateParameters(training_time);
        },
        error: function(error) {
            console.log(error);
        }
    });
}

function managementWork2() {
    $.get('/check_status', function(data) {
        // Если обучение запущено
        if (data === 'Фоновая работа запущена') {
            // Установить обновление данных и графика каждые n секунд
            interval = setInterval(function() {
                fetchData2();
            }, 1 * 1000);
            localStorage.setItem("trainingDataUpdateInterval", interval);
        }
        // Иначе остановить обновление
        else {
            interval = localStorage.getItem("trainingDataUpdateInterval");
            if (interval != null) {
                clearInterval(interval);
                interval = null;
                localStorage.setItem("trainingDataUpdateInterval", interval);
            }
        }
    });
}

function restoreParameters() {
    // Получить сохраненное значение параметров из локального хранилища
    const timeValue = localStorage.getItem('timeValue');
    if (timeValue) {
        // Обновить значение timeValue
        const timeValueElement = document.getElementById('time-value');
        if (timeValueElement) {
            timeValueElement.textContent = timeValue;
        }
    }
}

function clearChart2() {
    if (chart2) {
        // Очистить данные графика
        chart2.data.labels = [];
        chart2.data.datasets.forEach(dataset => {
            dataset.data = [];
        });
        // Обновить график
        chart2.update();
    }
}


// При загрузке страницы восстанавливаем состояния
$(document).ready( function () {
    // Создание графика
    createChart2();
    // Восстанавление данных графика
//    restoreChartState();
    // Восстановить значения параметров
    restoreParameters();
    // Актуализация обновления данных
    managementWork2();
});
