from flask import Flask
from flask import request
from flask import render_template
import numpy as np
import sklearn.datasets
from sklearn import neighbors
import datetime
import calendar
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/earnings')
def index2():
    global day_season_means
    global radius
    global results
    global dates
    global weekdays

    day_season_means = []
    radius = []
    results = []
    dates = []
    weekdays = []

    #------------------Recibiendo parámetross del formulario
    cf = float(request.args.get('cf'))
    c = float(request.args.get('c'))
    ci = float(request.args.get('ci'))
    pr_day = datetime.date.fromisoformat(request.args.get('pr-day'))
    flag = 0
    today = datetime.date.today()
    today = today + datetime.timedelta(days=-1)
    cf_all = [cf]; c_all = [c]; ci_all = [ci]


    #--------------arreglos de tempradas (según meses) y días de la semana
    seasons_idx = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    dias = ['x', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    #---------------Carga de datos y asignación de tamaño para entrenamiento (todos los datos).
    salida = np.loadtxt(fname='C:/wamp64/www/day-predictor/data/venta-neta-all.txt')
    entrada = np.loadtxt(fname='C:/wamp64/www/day-predictor/data/day-season-caps-all.csv', delimiter=',')
    dataset = sklearn.datasets.base.Bunch(data=entrada, target=salida)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset['data'], dataset['target'], test_size=0)

    #----------------------separación de días por temporadas----------------------
    s1 = entrada[entrada[:, 1] == 1]
    s2 = entrada[entrada[:, 1] == 2]
    s3 = entrada[entrada[:, 1] == 3]
    s4 = entrada[entrada[:, 1] == 4]
    seasons = [s1, s2, s3, s4]
    # print(s1.shape, s2.shape, s3.shape, s4.shape)

    #----------------matriz de días de la semana x temporada--------------
    for i in range(0, 4):
        day = []
        for j in range(1, 8):
            data = []

            #-------------promedio de capital final[2], compras [3] y capital inicial [4]

            data.append(np.mean(seasons[i][seasons[i][:, 0] == j][:, 2]))
            data.append(np.mean(seasons[i][seasons[i][:, 0] == j][:, 3]))
            data.append(np.mean(seasons[i][seasons[i][:, 0] == j][:, 4]))

            # -------------desviación estándar de capital final[2], compras [3] y capital inicial [4]
            data.append(np.std(seasons[i][seasons[i][:, 0] == j][:, 2]))
            data.append(np.std(seasons[i][seasons[i][:, 0] == j][:, 3]))
            data.append(np.std(seasons[i][seasons[i][:, 0] == j][:, 4]))

            #-----------------radio = venta neta promedio - venta neta dispersa-------------
            rad = data[0] + data[1] - data[2] - data[3] - data[4] + data[5]
            radius.append(rad)
            day.append(data)
        day_season_means.append(day)

    #-----------------Radius Neighbors Regressor para predicciones continuas-------------
    from sklearn.neighbors import RadiusNeighborsRegressor
    neigh = RadiusNeighborsRegressor(radius=np.mean(radius), weights='distance')
    neigh.fit(X_train, y_train)

    #knn = neighbors.KNeighborsRegressor()
    #knn.fit(X_train, y_train)

    add_days = 1
    future_date = today + datetime.timedelta(days=add_days)
    dates = [future_date]
    weekdays.append(future_date.weekday()+1)
    #--------------------diferencia entre lo especificado en el formulario y el promedio.
    #---------Si está dentro de la desviación estándar, se considera, sino se toma el promedio y la diferencia es 0.

    diff_cf = cf - day_season_means[seasons_idx[today.month]][today.weekday()][0]
    diff_c = c - day_season_means[seasons_idx[today.month]][today.weekday()][1]
    diff_ci = ci - day_season_means[seasons_idx[today.month]][today.weekday()][2]

    if(abs(diff_cf) > day_season_means[seasons_idx[today.month]][today.weekday()][3]):
        diff_cf = 0
    if (abs(diff_c) > day_season_means[seasons_idx[today.month]][today.weekday()][4]):
        diff_c = 0
    if (abs(diff_ci) > day_season_means[seasons_idx[today.month]][today.weekday()][5]):
        diff_ci = 0

    while future_date <= pr_day:
        flag+=1
        monthday = future_date.day
        day = future_date.weekday() + 1
        season = seasons_idx[future_date.month]
        pmnt = 0
        if (monthday == 1 | monthday == 2 | monthday == 15 | monthday == 16 | monthday ==
                calendar.monthrange(future_date.year, future_date.month)[1]):
            pmnt = 1

        #------------ ocupar los datos generados con el entrenamiento para volver a entrenar ---------------
        if(add_days != 1):
            cf = day_season_means[season - 1][day - 1][0] + diff_cf
            c = day_season_means[season - 1][day - 1][1] + diff_c
            ci = day_season_means[season - 1][day - 1][2] + diff_ci
            cf_all = np.append(cf_all,cf); c_all = np.append(c_all,c); ci_all = np.append(ci_all,ci)

        results.append(neigh.predict([[day, season, cf, c, ci, pmnt]]))
        np.append(X_train,[[day, season, cf, c, ci, pmnt]])
        np.append(y_train,[results[-1]])
        neigh.fit(X_train,y_train)

        add_days+=1
        future_date = today + datetime.timedelta(days=add_days)
        dates = np.append(dates,future_date)
        weekdays.append(future_date.weekday()+1)

    if(pr_day <= today):
        flag = -1
        pmnt = 0
        monthday = pr_day.day
        if (monthday == 1 | monthday == 2 | monthday == 15 | monthday == 16 | monthday ==
                calendar.monthrange(future_date.year, future_date.month)[1]):
            pmnt = 1
        results.append(neigh.predict([[pr_day.weekday()+1, seasons_idx[pr_day.month], cf+diff_cf, c+diff_c, ci+diff_ci, pmnt]]))



    return render_template('result.html',cfs = cf_all, cs = c_all, cis = ci_all, dia = pr_day, res = results, f = flag, today = today, dates = dates)



@app.route('/graphic')
def graphics():
    wkds = weekdays; res = results
    salida = np.loadtxt(fname='C:/wamp64/www/day-predictor/data/venta-neta-all.txt')
    entrada = np.loadtxt(fname='C:/wamp64/www/day-predictor/data/day-season-caps-all.csv', delimiter=',')
    dataset = sklearn.datasets.base.Bunch(data=entrada, target=salida)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset['data'], dataset['target'], test_size=0.00)

    plt.scatter(X_train[:, 0], y_train, c='r', label='data')
    plt.scatter(wkds[:-1], res, c='g', label='prediction', alpha=1)
    plt.axis('tight')
    plt.legend()
    plt.title("Radius neighbors approach.")
    plt.show()

    return "Grafica"
if __name__ == '__main__':
    app.run(debug = True)