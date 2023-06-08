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
                borderColor: 'orange',
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
                        text: 'Потери'
                    }
                }
            }
        }
    });
}



// Отображение данных на графике
function updateChart2(loss) {
//    console.log('Сработал updateChart2')
//    for (let i = 0; i < true_temp.length; i++) {
//        const item = true_temp[i];
//        chart2.data.labels.push(item.date_time);
//        chart2.data.datasets[0].data.push(item.temp_audience);
//        console.log('date_time: ' + item.date_time + ' true_temp: ' + item.temp_audience)
//    }
//
//    // Обновление графика
//    chart3.update();
}

function updateParameters(training_time, mse, r2) {
    // Обновление значения number-drift-points
    const timeValueElement = document.getElementById('time-value');
    if (timeValueElement) {
        timeValueElement.textContent = training_time;

        // Сохранить значение в локальное хранилище
        localStorage.setItem('timeValue', training_time);
    }

    const lossValueElement = document.getElementById('loss-value');
    if (lossValueElement) {
        if (mse != null){
            lossValueElement.textContent = mse;

            // Сохранить значение в локальное хранилище
            localStorage.setItem('lossValue', mse);
        }
    }

    const accuracyValueElement = document.getElementById('accuracy-value');
    if (accuracyValueElement) {
        if (r2 != null){
            accuracyValueElement.textContent = r2;

            // Сохранить значение в локальное хранилище
            localStorage.setItem('accuracyValue', r2);
        }
    }
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

    const lossValue = localStorage.getItem('lossValue');
    if (lossValue) {
        const lossValueElement = document.getElementById('loss-value');
        if (lossValueElement) {
            lossValueElement.textContent = lossValue;
        }
    }

    const accuracyValue = localStorage.getItem('accuracyValue');
    if (accuracyValue) {
        const accuracyValueElement = document.getElementById('accuracy-value');
        if (accuracyValueElement) {
            accuracyValueElement.textContent = accuracyValue;
        }
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
            var mse = response.mse;
            var r2 = response.r2;
            console.log('training_time: ' + training_time + ' mse: ' + mse + ' r2: ' + r2)
            updateParameters(training_time, mse, r2);
            //updateChart2(loss);
            saveChartState2(); // Сохранить состояние графика после обновления данных
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


function restoreChartState2() {
    // Получить сохраненные данные графика из локального хранилища
    const chartData = localStorage.getItem('chartData2');
    const chartOptions = localStorage.getItem('chartOptions2');

    if (chartData && chartOptions) {
        // Восстановить данные графика
        chart2.data = JSON.parse(chartData);
        chart2.options = JSON.parse(chartOptions);
        chart2.update();
    }
}


function saveChartState2() {
    if (chart2) {
        // Сохранить данные графика в локальное хранилище
        localStorage.setItem('chartData2', JSON.stringify(chart2.data));
        localStorage.setItem('chartOptions2', JSON.stringify(chart2.options));
    }
}



// При загрузке страницы восстанавливаем состояния
$(document).ready( function () {
    // Создание графика
    createChart2();
    // Восстанавление данных графика
    restoreChartState2();
    // Восстановить значения параметров
    restoreParameters();
    // Актуализация обновления данных
    managementWork2();

    //console.log('Сработал displayTraining')
});
