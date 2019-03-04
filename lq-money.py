#!C:\Users\Alonso Alavez\PycharmProjects\CursoPy\venv\Scripts\python.exe
import os,sys
import cgi, cgitb
cgitb.enable()
import numpy as np
import sklearn.datasets
import matplotlib
import datetime
import calendar
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
os.environ['HOME'] = '/tmp'
import matplotlib.pyplot as plt

form = cgi.FieldStorage()

seasons = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
dias = ['x', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

cf= float(form.getvalue('cf'))
c= float(form.getvalue('c'))
ci=float(form.getvalue('ci'))
des_date = datetime.date.fromisoformat(form.getvalue('pr-day'))
add_days = 1


future_date = datetime.date.today() + datetime.timedelta(days=add_days)
monthday = future_date.day
day= future_date.weekday()+1
season=seasons[future_date.month]
pmnt = 0
if(monthday == 1 | monthday == 2 | monthday == 15 | monthday== 16 | monthday == calendar.monthrange(future_date.year,future_date.month)[1]):
    pmnt = 1


salida = np.loadtxt(fname = 'C:/wamp64/www/day-predictor/data/venta-neta-all.txt')
entrada = np.loadtxt(fname = 'C:/wamp64/www/day-predictor/data/day-season-caps-all.csv', delimiter=',')
dataset = sklearn.datasets.base.Bunch(data=entrada, target=salida)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], test_size=.95, random_state=1)

knn = neighbors.KNeighborsRegressor()
knn.fit(X_train, y_train)
knnr_res = knn.predict([[day,season,cf,c,ci,pmnt]])
#plt.scatter(X_test[:,0], knnr_res, c='g', label='prediction',alpha=0.35)
#plt.axis('tight')
#plt.legend()
#plt.title("Radius neighbors approach.")

#plt.show()


print("Content-Type: text/html")
print(f"""
	<TITLE>Resultado </TITLE> 
    <form action="http://localhost/day-predictor/formu.html?">
        <input type="submit" value="Atrás" ">
    </form>
    <div>
    <h1>Si hoy tuviste ${cf} como capital final, ${c} de compras y ${ci} de capital inicial:</h1> <br/>
    </div> 
    <h2> Probablemente, mañana vendas ${knnr_res[0]} </h2>
    <br/>
    {des_date.day}
"""
)


