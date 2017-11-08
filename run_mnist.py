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
from data.mnist import mnist 
from models.bnn import bnn
### Script to run a BNN experiment over the moons data

seed = np.random.randint(1,23523452)
threshold = 0.1

## load data 
data = mnist(threshold=threshold)
data = Data(data.x_train, data.y_train, x_test=data.x_test, y_test=data.y_test, dataset='mnist', seed=seed)

n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES

## Specify model parameters
lr = (3e-4,)
n_hidden = [512,512]
n_epochs, batchsize = 30, 64
initVar, eval_samps = -10.0, None
batchnorm = 'standard' 

model = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
model.train(data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=True)

preds_test = model.predict_new(data.data['x_test'].astype('float32'))
acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

