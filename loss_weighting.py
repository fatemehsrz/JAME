import keras.backend as K
from keras.layers import Layer
from keras.constraints import Constraint, max_norm, min_max_norm, unit_norm, maxnorm, nonneg
from keras.initializers import Constant
import tensorflow as tf
import numpy as np


class LossWeighter(Layer):
    def __init__(self, nb_outputs=3,**kwargs):

        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        self.act=tf.nn.relu
        super(LossWeighter, self).__init__(**kwargs)


    def build(self, inputShape=None):

        self.loss_weights = []

        for i in range(self.nb_outputs):

            self.loss_weights+= [self.add_weight(name='weighted_loss' + str(i), shape=(1,), initializer=Constant(1.0), trainable=True, constraint= difficulty_control()) ]

        super(LossWeighter,self).build(inputShape)



    def call(self,inputs):

        firstLoss, secondLoss, thirdLoss = inputs
        return self.act((self.loss_weights[0]*firstLoss) + (self.loss_weights[1]*secondLoss) + (self.loss_weights[2]*thirdLoss))


    def compute_output_shape(self,inputShape):

        return inputShape[0]


class difficulty_control(object):

    def __init__(self ):
        self.min_value = 0
        self.max_value = 1
        self.rate=0.001
        self.alpha= 1.0


    def __call__(self, w):

        norms = K.sqrt(K.sum(K.square(w), axis=0, keepdims=True))
        clipping = (self.rate * K.clip(norms, self.min_value, self.max_value) +(1 - self.rate) * norms)
        ratio = (clipping / (K.epsilon() + norms))** self.alpha
        w *= K.softmax(ratio, axis=-1)

        return w








