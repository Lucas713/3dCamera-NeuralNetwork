import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))

import pydot
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
K.set_session(sess)

from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.utils import plot_model



# fonction générant une liste de n tableaux de taille x par y
# chaque tableau est fonction de deux variables selectionnées aléatoirement
def gen_data(n,sizex,sizey):
    ab_train = np.random.randint(100,size=(n,2))/10
    
    imgtot = []
    for i in range(n):
        ab = ab_train[i]
        a = ab[0]
        b = ab[1]
        img = []
        for x in range(sizex):
            xrow = []
            add = 0
            for y in range(sizey):
                #add = x + y
                add = a*b*(1+np.cos(a*x))+a*(1+np.sin(b+y))
                xrow.append(add)
            img.append(xrow)
        imgtot.append(img)
        #On représente les surfaces en 3D
#    k = 500
#    for i in range(1, n, k):
#        f = lambda x,y: a[i]*a[i]*np.sin(b[i]*x)-b[i]*np.cos(x+b[i]*y)
#        #f = lambda x,y: x*x/(a[i]*a[i])-y*y/(b[i]*b[i])
#        plt.subplot(n,2,i)
#        X = np.linspace(0,sizeX, 20)
#        Y = np.linspace(0, sizeY, 20)
#        X,Y = np.meshgrid(X,Y)
#        ax = plt.axes(projection = '3d')
#        Z = f(X,Y)
#        ax.plot_surface(X,Y,Z, cmap='plasma')
#        plt.savefig('surface_{}'.format(i))
#        plt.show()
    return ab_train, imgtot

# Fonction permettant d'afficher une matrice sous forme de graph
def graph_data(M):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sizeX = np.shape(M)[1]
    sizeY = np.shape(M)[0]
    X = np.linspace(0, sizeX-1, sizeX)
    Y = np.linspace(0, sizeY-1, sizeY)
    #ax = plt.axes(projection = '3d')
    X,Y = np.meshgrid(X,Y)
    ax.plot_surface(X,Y,np.array(M), cmap='plasma')
    #plt.savefig('surface_{}'.format(i))
    plt.show()

# Fonction permettant d'afficher une matrice sous forme d'image    
def img_data(M):
    fig = plt.figure()
    plt.imshow(np.array(M), cmap='hot')
    plt.show()
    
def calc_data(ab):
    a = ab[0]
    b = ab[1]
    img = []
    for x in range(sizex):
        xrow = []
        add = 0
        for y in range(sizey):
            #add = x + y
            add = a*b*(1+np.cos(a*x))+a*(1+np.sin(b+y))
            xrow.append(add)
        img.append(xrow)
    return img

N = 1000
sizex = 20
sizey = 10
Y_train, X_train = gen_data(N,sizex,sizey)
Y_test, X_test = gen_data(N,sizex,sizey)



###  MODEL  ###

model = Sequential()
#model.add(Flatten(input_shape=(sizex,sizey)))
#model.add(Dense(32, activation='relu'))
model.add(Dense(input_shape=(sizex,sizey), output_dim=32, activation='relu'))

model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dense(12,  activation='tanh'))
model.add(Dense(6, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='linear'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='rmsprop',
          metrics=['acc'])

history = model.fit(np.array(X_train), np.array(Y_train),
          batch_size=20, epochs=20,
          verbose=2,
          validation_data=(np.array(X_test), np.array(Y_test)))

#print(np.mean(history.history['acc'][40:]))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

###  TEST  ###

#print(model.predict(np.array(X_train), batch_size=1))
#print(model.evaluate(np.array(X_test),np.array(Y_test),batch_size=1, 
#               verbose=1))


### plot the model ###
#plot_model(model, to_file='/home/nvidia/Desktop/Codes/model.png')




# Save the model
model.save('/home/nvidia/Desktop/Codes/saved_model_batch20_epoch50.h5')

# Recreate the exact same model purely from the file
new_model = load_model('/home/nvidia/Desktop/Codes/saved_model.h5')

# affichage de la prediction pour l'entrée d'indice i
i = 100

Y_1 = new_model.predict(np.array(X_train), batch_size=1)[i]
print(Y_1)
graph_data(calc_data(Y_1))
img_data(calc_data(Y_1))
print(Y_train[i])
graph_data(X_train[i])
img_data(X_train[i])













