import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from sklearn.metrics import log_loss
import tensorflow as tf
from data.Container import Container as Data
from data import half_moon_loader
from models.bnn import bnn
### Script to run a BNN experiment over the moons data

seed = np.random.randint(1,23523452)

## load data 
n_train, n_test = 250, 1000
moon_data = half_moon_loader.download(n_train, n_test)
data = Data(moon_data[0][0], moon_data[0][1], x_test=moon_data[1][0], y_test=moon_data[1][1], dataset='moons', seed=seed)

n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES

## Specify model parameters
lr = (3e-4,)
n_hidden = [100,100]
n_epochs, batchsize = 50, 128
initVar, eval_samps = -10.0, None
batchnorm = 'standard' 

model = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
model.train(data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=False)

preds_test = model.predict_new(data.data['x_test'].astype('float32'))
acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

## Visualize predictions
range_x = np.arange(-2.,3.,.1)
range_y = np.arange(-1.5,2.,.1)
X,Y = np.mgrid[-2.:3.:.1, -1.5:2.:.1]
xy = np.vstack((X.flatten(), Y.flatten())).T

print('Starting plotting work')
predictions = model.predict_new(xy.astype('float32'))

zi = np.zeros(X.shape)
for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zi[i,j] = predictions[idx[0],0] * 100

plt.figure()
plt.contourf(X,Y,zi,cmap=plt.cm.coolwarm)
print('Done with heat map')

preds = np.argmax(preds_test, axis=1)
x0, x1 = data.data['x_test'][np.where(preds==0)], data.data['x_test'][np.where(preds==1)]

plt.scatter(x0[:,0], x0[:,1], color='g', s=1)
plt.scatter(x1[:,0], x1[:,1], color='m', s=1)
plt.savefig('./what', bbox_inches='tight')
