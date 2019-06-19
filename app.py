from keras.models import load_model
import flask
from flask import render_template
from flask import request
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf

app = flask.Flask(__name__)


class ClassifierModel:
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        self.model = None

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

    def load(self, path):
        with self.graph.as_default():
            with self.session.as_default():
                self.model = load_model(path)


models = {'Нейросеть':'ann',
              'K ближайших соседей':'knn',
              'Наивный байес':'nb',
              'Случайный лес':'rf'}

features =  {'Форма шляпки':{'Колокольчатая':'b', 'Коническая':'c', 'Выпуклая':'x', 'Плоская':'f', 'Плоская с выпирающим бугорком':'k', 'Воронковидная':'s'},
                 'Поверхность шляпки':{'Волокнистая':'f', 'Бороздчатая':'g', 'Чешуйчатая':'y', 'Гладкая':'s'},
                 'Цвет шляпки':{'Коричневый':'n', 'Охровый':'b', 'Коричный':'c', 'Серый':'g', 'Зеленый':'r', 'Розовый':'p', 'Фиолетовый':'u', 'Красный':'e', 'Белый':'w', 'Желтый':'y'},
                 'Повреждения':{'Есть':'t', 'Нет':'f'},
                 'Запах': {'Миндальный':'a', 'Анисовый':'l', 'Креозот':'c', 'Рыбный':'y', 'Неприятный':'f', 'Затхлый':'m', 'Нет':'n', 'Едкий':'p', 'Острый':'s'},
                 'Крепление нижней части шляпки': {'Прикрепленное':'a', 'Спускающееся':'d', 'Свободное':'f', 'Зубчатое':'n'},
                 'Пространство в нижней части шляпки': {'Близкое':'c', 'Тесное':'w', 'Удаленное':'d'},
                 'Размер нижней части шляпки': {'Широкий':'b', 'Узкий':'n'},
                 'Цвет нижней части шляпки': {'Черный':'k', 'Коричневый':'n', 'Охровый':'b', 'Шоколадный':'h', 'Серый':'g', 'Зеленый':'r', 'Оранжевый':'o', 'Розовый':'p', 'Фиолетовый':'u', 'Красный':'e', 'Белый':'w', 'Желтый':'y'},
                 'Форма ножки': {'Сужающаяся к низу':'e', 'Расширяющаяся к низу':'t'},
                 'Основание ножки': {'Луковичное':'b', 'Булавообразное':'c', 'Чашевидное':'u', 'Равное':'e', 'Ризоморфное':'z', 'Укорененное':'r', 'Отсутствует':'?'},
                 'Поверхность ножки ниже кольца': {'Волокнистая':'f', 'Чешуйчатая':'y', 'Шелковая':'k', 'Гладкая':'s'},
                 'Поверхность ножки выше кольца': {'Волокнистая':'f', 'Чешуйчатая':'y', 'Шелковая':'k', 'Гладкая':'s'},
                 'Цвет ножки выше кольца': {'Коричневый': 'n', 'Охровый': 'b', 'Коричный': 'c', 'Серый': 'g', 'Оранжевый': 'o', 'Розовый': 'p','Красный': 'e', 'Белый': 'w', 'Желтый': 'y'},
                 'Цвет ножки ниже кольца': {'Коричневый': 'n', 'Охровый': 'b', 'Коричный': 'c', 'Серый': 'g', 'Оранжевый': 'o', 'Розовый': 'p','Красный': 'e', 'Белый': 'w', 'Желтый': 'y'},
                 'Тип покрытия': {'Частичное':'p', 'Полное':'u'},
                 'Цвет покрытия': {'Коричневый':'n', 'Оранжевый':'o', 'Белый':'w', 'Желтый':'y'},
                 'Количество колец': {'Нет':'n', 'Одно':'o', 'Два':'t'},
                 'Вид колец': {'Паутинчатый':'c', 'Прозрачный':'e', 'Расширяющийся':'f', 'Большой':'l', 'Нет':'n', 'Висячий':'p', 'Укутывающий':'s', 'Зональный':'z'},
                 'Цвет спор': {'Черный':'k','Коричневый': 'n', 'Охровый': 'b', 'Шоколадный': 'h', 'Зеленый': 'r', 'Оранжевый': 'o', 'Фиолетовый': 'u', 'Белый': 'w','Желтый': 'y'},
                 'Популяция': {'Большое скопление':'a', 'В группах':'c', 'Много':'n', 'Разбросанная':'s', 'Несколько':'v', 'Единичный':'y'},
                 'Среда обитания': {'Трава':'g', 'Листья':'l', 'Поля':'m', 'Тропинки':'p', 'Город':'u', 'Свалка':'w', 'Лес':'d'}
                 }


@app.route("/", methods=["POST","GET"])
def main():
    res = ""
    if flask.request.method == "POST":
        responses = []
        for i,f in enumerate(features):
            responses.append(request.form.get(str(i+1)))

        m = request.form.get('model')
        print(m)
        model = ClassifierModel()
        if m == 'Нейросеть':
            model.load('ann.h5')
        elif m == 'K ближайших соседей':
            model = joblib.load('knn.pkl')
        elif m == 'Наивный байес':
            model = joblib.load('nb.pkl')
        elif m == 'Случайный лес':
            model = joblib.load('rf.pkl')

        X = pd.DataFrame(responses)
        le_x = joblib.load('le.pkl')
        for c in X.columns:
            X[c] = le_x.fit_transform(X[c])
        sc = StandardScaler()
        X = sc.fit_transform(X).transpose()

        preds = model.predict(X)
        preds = preds > 0.5
        res = 'Съедобный' if preds[0] else 'Ядовитый'

    return render_template('main.html', features=features, models=models, response=res)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='0.0.0.0')
