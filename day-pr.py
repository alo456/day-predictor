#!C:\Users\Alonso Alavez\PycharmProjects\CursoPy\venv\Scripts\python.exe
import os,sys
import cgi, cgitb
cgitb.enable()
import numpy as np
import sklearn.datasets
import matplotlib

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
os.environ['HOME'] = '/tmp'

import matplotlib.pyplot as plt

form = cgi.FieldStorage()

cf= float(form.getvalue('cf'))
c= float(form.getvalue('c'))
ci=float(form.getvalue('ci'))
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

salida = np.loadtxt(fname = 'C:/wamp64/www/day-predictor/data/pytarget.txt')
entrada = np.loadtxt(fname = 'C:/wamp64/www/day-predictor/data/pydata-15-16.csv', delimiter=',')
dataset = sklearn.datasets.base.Bunch(data=entrada, target=salida)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], test_size=.60, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train,y_train)

nb_res = gnb.predict([[cf, c, ci]])
nb_proba = gnb.predict_proba([[cf,c,ci]])


X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], test_size=.65, random_state=1)

from sklearn import tree
cfr = tree.DecisionTreeClassifier()
cfr.fit(X_train, y_train)

dtc_res = cfr.predict([[cf,c,ci]])
dtc_proba = cfr.predict_proba([[cf,c,ci]])


X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], test_size=.67, random_state=1)

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
knc_res = clf.predict([[cf,c,ci]])


X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], test_size=.58, random_state=1)

knn = neighbors.KNeighborsRegressor()
knn.fit(X_train, y_train)
knnr_res = clf.predict([[cf,c,ci]])

#plt.plot(dias, pred_proba[0,:])
#plt.show()

print("Content-Type: text/html")
print(f"""
	<TITLE>Resultado </TITLE> 
    <form action="http://localhost/day-predictor/formu.html">
        <input type="submit" value="Atrás" ">
    </form>
    <div>
    <h1>Día más probable con ${cf} como capital final, ${c} de compras y ${ci} de capital inicial:</h1> <br/>
    </div> 
    <h2>{dias[int(nb_res[0]) - 1]}, según "Naive Bayes", con una probabilidad del {round(nb_proba[0][int(nb_res[0]) - 1]*100,2)}%. </h2> <br/>
    <h2>{dias[int(dtc_res[0]) - 1]}, según "Decision Tree", con una probabilidad del {round(dtc_proba[0][int(dtc_res[0])-1]*100,2)}%. </h2> <br/>
    <h2>{dias[int(knc_res[0]) - 1]}, según "K Neighbors Classifier". </h2> <br/>
    <h2>{dias[int(knnr_res[0]) - 1]}, según "K Neighbors Regressor". </h2> <br/>
"""
)
