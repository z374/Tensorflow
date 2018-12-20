In **maniac_regression.py vengono:

1) Generate coppie di numeri casuali (x,y), ben descritte da una relazione linare y = mx + q. I dati sono affetti da eteroschedasticità.
2) Generato un grafo in Tensorflow per il calcolo dei coefficienti m e q secondo la minimizzazione dei quadrati. 
3) Effettuato un confronto con il modello lineare presente all'interno di sklearn.

**Principali funzioni utilizzate:

- test_train_split (sklearn)
- StandarScaler (sklearn)
- LinearRegression (sklearn)

**Struttura del grafo:

![alt text](https://raw.githubusercontent.com/z374/Tensorflow/master/Linear_Regression/struttura.PNG)

**Esempio output:

![alt text](https://raw.githubusercontent.com/z374/Tensorflow/master/Linear_Regression/output.PNG)
