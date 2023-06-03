'''
    Запуск приложения
'''
from app import app

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])