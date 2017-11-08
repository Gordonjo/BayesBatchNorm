Implementation and experimentation of Batch Normalization for variational Bayesian neural networks
=======
This repository implements supervised Bayesian neural networks according to `Blundell et al.' (2015) <https://arxiv.org/abs/1505.05424>`_. Networks are implemented with straightforward Batch normalization as in `Ioffe and Szegedy, (2015) <https://arxiv.org/abs/1502.03167>`_, as well as with an adapted version for variational BNNs. 

The models are implementated in `TensorFlow  1.3 <https://www.tensorflow.org/api_docs/>`_.


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.


**Install**


.. code-block:: bash

  pip install scipy
  pip install numpy
  pip install matplotlib
  pip install tensorflow(-gpu)

Examples
-------------
Run run_{moons, mnist}.py without any arguments to train a Bayesian neural network on one of these datasets. Will print training process to screen, as well as save a contour plot for the moons data. In the scripts, you can toggle between no batchnorm (batchnorm='None'), standard batchnorm (batchnorm='standard') and Bayesian batchnorm (batchnorm='bayes').
