# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:27:01 2018

@author: z374

Uso di tensorflow per costruire un regressore lineare unidimensoinale.
Confronto con regressione lineare.

"""

import numpy as np
import matplotlib.pyplot as plt

# In[1]: Creazione dei dati casuali e suddivisione

# N : numero di sample
# modificabili : coefficiente angolare e sparpagliamento.


N = 200
X = np.float32(np.arange(N))
y = np.arange(N)
y = X*np.random.randint(10) + np.random.randn(N)*X

#riscaliamo un pochetto va'
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()
X= sc.fit_transform(X.reshape(-1,1))
y = sc.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X,y, shuffle= True, test_size=0.2)

plt.scatter(X_tr,y_tr, color = 'blue', facecolors ='none', label = 'train salmple');
plt.scatter(X_te,y_te, color = 'blue', label = 'test sample'); 
plt.title("Input data")




# In[2]: Creazione di un grafo tensorflow per la regressione (low-level)

# modello : z = mx + q
# note # se non standardizzi l'input, devi fare attenzione al learning_rate


import tensorflow as tf

## Grafo
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name = 'X')
    y = tf.placeholder(dtype=tf.float32, shape=(None), name = 'y')
    m = tf.Variable(initial_value = .1, name = 'm')
    q = tf.Variable(initial_value = .1, name = 'q')
    z = m*x + q
    
    scarto_medio = tf.reduce_mean(tf.square(y-z),axis=0)
    minimizza = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    minimizza = minimizza.minimize(scarto_medio)
    init = tf.global_variables_initializer()
    
## Sessione
with tf.Session(graph = g) as sess:
    sess.run(init)
    for _ in range(800):
        sess.run(minimizza, feed_dict= {x:X_tr, y:y_tr})

    
    ippsilonne = sess.run(z, feed_dict={x:X_te})     
    plt.plot(X_te,ippsilonne, color = 'r', label = 'fit', lw=1) ; plt.legend(loc = 'upper left') ; plt.show()

    print("\nControllo qualità dei poveri:")
    print("Scarto medio su training: %10.3f" % sess.run(scarto_medio,feed_dict={x:X_tr, y:y_tr}))
    print("Scarto medio su test:     %10.3f" % sess.run(scarto_medio,feed_dict={x:X_te, y:y_te}))
    print("Stima della retta(tensorflow): y = %.5f x + %.5f" % (sess.run(m), sess.run(q)))


# In[3]: Regressione lineare a confronto:
    
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_tr, y_tr)
print("LinearRegression(sklearn):     y = %.5f x + %.5f" %(lr.coef_, lr.intercept_))

